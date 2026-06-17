//! The game store: a websocket client that turns the server's event stream into
//! reactive UI state.
//!
//! The protocol is event-sourced, so this store mirrors it: a `DEAL` resets the
//! hand, `AWAITING` marks whose turn it is, `UPDATE` applies each public action,
//! and `TRICK_WON` sweeps the table. Hidden information the wire omits is
//! reconstructed locally — most notably the dealer's picked-up up-card, which is
//! folded into our hand exactly when the server asks us to discard.
//!
//! Because the server (and its bots) emit events far faster than a human can
//! follow, messages are not applied as they arrive: they go through a small
//! render queue that paces them out (see the pacing constants below), applying
//! one at a time in order so each action appears with a legible gap.

import type {
  ActionBubble,
  ConnStatus,
} from './game-types';
import type {
  CardCode,
  ClientMsg,
  HandScore,
  PublicAction,
  Scores,
  Seat,
  SeatedPlayer,
  ServerMsg,
  Suit,
  Team,
  TurnHint,
} from './protocol';
import { SUIT_SYMBOL, parseCard, sortHand } from './cards';

// --- Render pacing (all milliseconds) ---------------------------------------
//
// The server fires events as fast as the bots decide, which is too quick to
// follow, so the client paces them out. Incoming messages are queued and
// rendered strictly in order, one at a time. The delays below are *minimums*:
// time a message already spent in flight counts toward them, so an action that
// takes 1.2s to arrive shows at once, while one that arrives in 0.2s waits out
// the remaining 0.3s. Your own actions are exempt — they render the instant you
// make them (you cannot act before your turn unlocks, which is itself gated
// behind the previous action).

/** Minimum gap between two consecutive actions (bid, pass, discard, play). */
const ACTION_GAP_MS = 500;
/** How long a completed trick rests on the table before being swept up. */
const TRICK_LINGER_MS = 1000;
/** How long a hand's result lingers before the next hand is dealt. */
const HAND_END_PAUSE_MS = 1500;

const SEATS: Seat[] = ['North', 'East', 'South', 'West'];

function seatRecord<T>(value: () => T): Record<Seat, T> {
  return { North: value(), East: value(), South: value(), West: value() };
}

function partnerOf(seat: Seat): Seat {
  switch (seat) {
    case 'North':
      return 'South';
    case 'South':
      return 'North';
    case 'East':
      return 'West';
    case 'West':
      return 'East';
  }
}

function teamShort(team: Team): string {
  return team === 'NorthSouth' ? 'N/S' : 'E/W';
}

function describeHand(score: HandScore): string {
  const [team, points] = score.points_awarded;
  const who = teamShort(team);
  if (score.euchred) return `${who} euchred the makers for ${points}!`;
  if (score.march) return `${who} swept the hand${score.alone ? ' alone' : ''} for ${points}!`;
  return `${who} made it for ${points}.`;
}

/** Whether rendering this message advances the pacing clock (a visible beat). */
function paceSetting(msg: ServerMsg): boolean {
  switch (msg.type) {
    case 'UPDATE': // any action: bid, pass, discard, or play
    case 'TRICK_WON':
    case 'HAND_COMPLETE':
    case 'DEAL': // so the first bid sits a beat after the hand is dealt
      return true;
    default:
      return false;
  }
}

export class GameStore {
  // --- connection ---------------------------------------------------------
  status = $state<ConnStatus>('idle');
  error = $state<string | null>(null);

  // --- table identity -----------------------------------------------------
  mySeat = $state<Seat | null>(null);
  players = $state<Record<Seat, SeatedPlayer | null>>(seatRecord<SeatedPlayer | null>(() => null));
  dealer = $state<Seat | null>(null);

  // --- bidding ------------------------------------------------------------
  upCard = $state<CardCode | null>(null);
  upCardLive = $state(false);

  // --- contract -----------------------------------------------------------
  trump = $state<Suit | null>(null);
  maker = $state<Seat | null>(null);
  alone = $state(false);

  // --- the play -----------------------------------------------------------
  hand = $state<CardCode[]>([]);
  /** Cards resting on the felt for the trick in progress. */
  table = $state<{ seat: Seat; card: CardCode }[]>([]);
  /** A finished trick mid-sweep toward its winner (kept separate so the new
   *  trick can start building underneath the animation). */
  collecting = $state<{ plays: { seat: Seat; card: CardCode }[]; winner: Seat } | null>(null);

  // --- turn ---------------------------------------------------------------
  whoseTurn = $state<Seat | null>(null);
  hint = $state<TurnHint | null>(null);
  legal = $state<CardCode[] | null>(null);

  // --- bookkeeping --------------------------------------------------------
  scores = $state<Scores>({ north_south: 0, east_west: 0 });
  /** The match target; the server uses 10 by default and does not send it. */
  readonly targetScore = 10;
  tricksWon = $state<Record<Seat, number>>(seatRecord(() => 0));
  cardsLeft = $state<Record<Seat, number>>(seatRecord(() => 5));
  bubbles = $state<Record<Seat, ActionBubble | null>>(seatRecord<ActionBubble | null>(() => null));

  // --- transient notices --------------------------------------------------
  banner = $state<string | null>(null);
  toast = $state<string | null>(null);
  gameOver = $state<{ winner: Team; scores: Scores } | null>(null);

  // --- private ------------------------------------------------------------
  private ws: WebSocket | null = null;
  private name = 'You';
  private preferredSeat: Seat | null = null;
  private lastBidUp = true;
  private bubbleSeq = 0;
  private timers: ReturnType<typeof setTimeout>[] = [];
  private collectTimers: ReturnType<typeof setTimeout>[] = [];

  // --- render queue -------------------------------------------------------
  /** Server messages awaiting their paced turn to be rendered, in order. */
  private queue: ServerMsg[] = [];
  /** Pending timer for the head of the queue, or null when idle/running. */
  private pumpTimer: ReturnType<typeof setTimeout> | null = null;
  /** When the last pace-setting event (a play, sweep, or hand result) rendered. */
  private lastRenderAt = Number.NEGATIVE_INFINITY;

  // --- derived (used by the UI) ------------------------------------------
  get sittingOut(): Seat | null {
    return this.alone && this.maker ? partnerOf(this.maker) : null;
  }
  get myTurn(): boolean {
    return this.mySeat !== null && this.whoseTurn === this.mySeat;
  }
  get awaitingBid(): boolean {
    return this.myTurn && this.hint?.kind === 'BID';
  }
  get awaitingDiscard(): boolean {
    return this.myTurn && this.hint?.kind === 'DISCARD';
  }
  get awaitingPlay(): boolean {
    return this.myTurn && this.hint?.kind === 'PLAY';
  }

  // --- connection ---------------------------------------------------------
  connect(url: string, name: string, seat: Seat | null): void {
    this.name = name.trim() || 'You';
    this.preferredSeat = seat;
    this.error = null;
    this.status = 'connecting';
    this.resetQueue();

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      this.status = 'error';
      this.error = `Not a valid websocket URL: ${url}`;
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.send({ type: 'HELLO', name: this.name, seat: this.preferredSeat });
    };
    ws.onmessage = (event) => {
      try {
        this.enqueue(JSON.parse(event.data as string) as ServerMsg);
      } catch (err) {
        console.error('failed to handle server message', err);
      }
    };
    ws.onerror = () => {
      if (this.status !== 'joined') {
        this.status = 'error';
        this.error = 'Could not reach the server. Is euchre-server running?';
      }
    };
    ws.onclose = () => {
      this.clearTimers();
      this.resetQueue();
      if (this.status === 'joined') {
        this.status = 'closed';
      } else if (this.status !== 'error') {
        this.status = 'error';
        this.error ??= 'The connection closed before the table was joined.';
      }
    };
  }

  // --- outgoing actions ---------------------------------------------------
  orderUp(alone: boolean): void {
    if (!this.upCard) return;
    this.bid(parseCard(this.upCard).suit, alone);
  }
  bid(suit: Suit, alone: boolean): void {
    this.send({ type: 'BID', suit, alone });
    this.endMyTurn();
  }
  pass(): void {
    this.send({ type: 'PASS' });
    this.endMyTurn();
  }
  discardCard(card: CardCode): void {
    this.send({ type: 'DISCARD', card });
    // The rebroadcast UPDATE hides the buried card, so drop it locally now.
    this.hand = this.hand.filter((c) => c !== card);
    this.endMyTurn();
  }
  playCard(card: CardCode): void {
    this.send({ type: 'PLAY', card });
    this.endMyTurn();
  }

  // --- render queue -------------------------------------------------------
  /** Queue a freshly-arrived server message and keep the pump running. */
  private enqueue(msg: ServerMsg): void {
    this.queue.push(msg);
    if (this.pumpTimer === null) this.drain();
  }

  /**
   * Render as many queued messages as are due right now, then arm a timer for
   * the next one that still has to wait. The minimum nature falls out of
   * comparing against the wall clock: a message overdue by the time we reach it
   * renders immediately.
   */
  private drain = (): void => {
    this.pumpTimer = null;
    for (;;) {
      const msg = this.queue[0];
      if (!msg) return;
      const due = this.lastRenderAt + this.delayBefore(msg);
      const wait = due - performance.now();
      if (wait > 0) {
        this.pumpTimer = setTimeout(this.drain, wait);
        return;
      }
      this.queue.shift();
      this.apply(msg);
      if (paceSetting(msg)) this.lastRenderAt = performance.now();
    }
  };

  /** The minimum delay, since the last paced render, before showing `msg`. */
  private delayBefore(msg: ServerMsg): number {
    switch (msg.type) {
      case 'UPDATE':
        // Opponents' actions are paced; our own render the instant we make them.
        return msg.player !== this.mySeat ? ACTION_GAP_MS : 0;
      case 'TRICK_WON':
        return TRICK_LINGER_MS;
      case 'DEAL':
      case 'GAME_OVER':
        return HAND_END_PAUSE_MS;
      default:
        return 0;
    }
  }

  private resetQueue(): void {
    if (this.pumpTimer !== null) clearTimeout(this.pumpTimer);
    this.pumpTimer = null;
    this.queue = [];
    this.lastRenderAt = Number.NEGATIVE_INFINITY;
  }

  // --- incoming messages --------------------------------------------------
  private apply(msg: ServerMsg): void {
    switch (msg.type) {
      case 'JOINED': {
        this.mySeat = msg.your_seat;
        this.dealer = msg.first_dealer;
        const players = seatRecord<SeatedPlayer | null>(() => null);
        for (const p of msg.players) players[p.seat] = p;
        this.players = players;
        this.status = 'joined';
        break;
      }
      case 'SYNC': {
        const v = msg.view;
        this.dealer = v.dealer;
        this.scores = v.scores;
        if (v.contract) {
          this.trump = v.contract.trump;
          this.maker = v.contract.maker;
          this.alone = v.contract.alone;
        }
        this.hand = sortHand(v.hand, this.trump);
        this.table = v.current_trick.plays.map((p) => ({ seat: p.seat, card: p.card }));
        const won = seatRecord(() => 0);
        for (const [, winner] of v.completed_tricks) won[winner] += 1;
        this.tricksWon = won;
        const left = seatRecord(() => 5 - v.completed_tricks.length);
        for (const p of v.current_trick.plays) left[p.seat] -= 1;
        if (this.sittingOut) left[this.sittingOut] = 0;
        this.cardsLeft = left;
        break;
      }
      case 'DEAL': {
        this.clearTimers();
        this.dealer = msg.dealer;
        this.upCard = msg.up_card;
        this.upCardLive = true;
        this.hand = sortHand(msg.hand, null);
        this.trump = null;
        this.maker = null;
        this.alone = false;
        this.table = [];
        this.collecting = null;
        this.whoseTurn = null;
        this.hint = null;
        this.legal = null;
        this.tricksWon = seatRecord(() => 0);
        this.cardsLeft = seatRecord(() => 5);
        this.bubbles = seatRecord<ActionBubble | null>(() => null);
        this.banner = null;
        break;
      }
      case 'AWAITING': {
        this.whoseTurn = msg.player;
        this.hint = msg.hint;
        this.legal = msg.player === this.mySeat ? (msg.legal ?? null) : null;
        if (msg.hint.kind === 'BID') {
          this.lastBidUp = msg.hint.up;
          if (!msg.hint.up) this.upCardLive = false; // round two: up-card turned down
        }
        if (msg.hint.kind === 'DISCARD' && msg.player === this.mySeat && this.upCard) {
          // The dealer has taken the up-card; fold it in so we can choose to bury.
          if (!this.hand.includes(this.upCard)) {
            this.hand = sortHand([...this.hand, this.upCard], this.trump);
          }
        }
        break;
      }
      case 'UPDATE': {
        this.applyAction(msg.player, msg.action);
        break;
      }
      case 'TRICK_WON': {
        this.tricksWon[msg.player] += 1;
        // The trick has already lingered (the queue held this message), so sweep now.
        if (this.table.length > 0) this.sweep(msg.player);
        break;
      }
      case 'HAND_COMPLETE': {
        if (msg.result === 'PassedOut') {
          this.flashBanner('Hand passed out — no one scored.');
        } else {
          const score = msg.result.Played;
          const [team, points] = score.points_awarded;
          if (team === 'NorthSouth') {
            this.scores = { ...this.scores, north_south: this.scores.north_south + points };
          } else {
            this.scores = { ...this.scores, east_west: this.scores.east_west + points };
          }
          this.flashBanner(describeHand(score));
        }
        break;
      }
      case 'GAME_OVER': {
        this.scores = msg.scores;
        this.gameOver = { winner: msg.winner, scores: msg.scores };
        break;
      }
      case 'ERROR': {
        this.flashToast(msg.message);
        break;
      }
    }
  }

  private applyAction(player: Seat, action: PublicAction): void {
    switch (action.type) {
      case 'BID': {
        this.trump = action.suit;
        this.maker = player;
        this.alone = action.alone;
        this.upCardLive = false;
        const verb = this.lastBidUp ? 'orders up' : 'calls';
        this.setBubble(player, `${verb} ${SUIT_SYMBOL[action.suit]}${action.alone ? ' alone' : ''}`);
        if (this.sittingOut) this.cardsLeft[this.sittingOut] = 0;
        break;
      }
      case 'PASS': {
        this.setBubble(player, 'pass');
        break;
      }
      case 'DISCARD': {
        this.setBubble(player, 'discards');
        break;
      }
      case 'PLAY': {
        this.table = [...this.table, { seat: player, card: action.card }];
        if (player === this.mySeat) {
          this.hand = this.hand.filter((c) => c !== action.card);
        } else {
          this.cardsLeft[player] = Math.max(0, this.cardsLeft[player] - 1);
        }
        break;
      }
    }
  }

  // --- helpers ------------------------------------------------------------
  /** Lift the finished trick off the felt and fly it to the winner. */
  private sweep(winner: Seat): void {
    this.clearCollectTimers();
    this.collecting = { plays: this.table, winner };
    this.table = [];
    // Let the cards mount in place, then trigger their `out` transition toward
    // the winner; tear the layer down once that animation has run.
    this.collectTimers.push(
      setTimeout(() => {
        if (this.collecting) this.collecting.plays = [];
      }, 30),
    );
    this.collectTimers.push(
      setTimeout(() => {
        this.collecting = null;
      }, 30 + 460),
    );
  }

  private send(msg: ClientMsg): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  private endMyTurn(): void {
    // Disable controls immediately; the next AWAITING re-enables when due.
    this.whoseTurn = null;
    this.hint = null;
    this.legal = null;
  }

  private setBubble(seat: Seat, text: string): void {
    const key = ++this.bubbleSeq;
    this.bubbles[seat] = { text, key };
    this.timers.push(
      setTimeout(() => {
        if (this.bubbles[seat]?.key === key) this.bubbles[seat] = null;
      }, 1800),
    );
  }

  private flashBanner(text: string): void {
    this.banner = text;
    this.timers.push(
      setTimeout(() => {
        if (this.banner === text) this.banner = null;
      }, 2800),
    );
  }

  private flashToast(text: string): void {
    this.toast = text;
    this.timers.push(
      setTimeout(() => {
        if (this.toast === text) this.toast = null;
      }, 3200),
    );
  }

  private clearCollectTimers(): void {
    for (const t of this.collectTimers) clearTimeout(t);
    this.collectTimers = [];
  }

  private clearTimers(): void {
    for (const t of this.timers) clearTimeout(t);
    this.timers = [];
    this.clearCollectTimers();
  }
}

// Keep the seat list exported for any future per-seat iteration needs.
export { SEATS };
