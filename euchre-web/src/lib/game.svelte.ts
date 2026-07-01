//! The game store: a websocket client that turns the server's event stream into
//! reactive UI state.
//!
//! The protocol is event-sourced, so this store mirrors it: a `DEAL` resets the
//! hand, `AWAITING` marks whose turn it is, `UPDATE` applies each public action,
//! and `TRICK_WON` sweeps the table. Hidden information the wire omits is
//! reconstructed locally — most notably the dealer's picked-up up-card, which is
//! folded into our hand exactly when the server asks us to discard.
//!
//! ## Identity
//!
//! The store works entirely in fixed table positions ([`Player`]: `0` = North …
//! `3` = West), which is what every top-level wire field already carries. The
//! one exception is a `SYNC` snapshot, whose trick history names seats
//! *relative to the dealer* ([`Seat`]); those are converted to a `Player`
//! with [`seatToPlayer`] as they are read in.
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
  Player,
  PublicAction,
  ScoredAction,
  Seat,
  SeatInfo,
  ServerMsg,
  SuggestedAction,
  Suit,
  TeamId,
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
/** How long the "game over" result lingers before returning to the lobby. */
const GAME_OVER_LINGER_MS = 4000;

/** Fixed-team cumulative score, indexed by team (`0` = North/South). */
type TeamScores = { north_south: number; east_west: number };

/** A fresh per-player array (`0` = North … `3` = West). */
function playerArray<T>(value: () => T): T[] {
  return [value(), value(), value(), value()];
}

/** A player's partner, sitting across the table. */
function partnerOf(player: Player): Player {
  return (player + 2) % 4;
}

/** A player's fixed team (`0` = North/South, `1` = East/West). */
function teamOf(player: Player): TeamId {
  return player % 2;
}

/** Clockwise steps from the dealer to each dealer-relative seat. */
const SEAT_OFFSET: Record<Seat, number> = { First: 1, Second: 2, Third: 3, Dealer: 0 };

/** Resolves a dealer-relative wire seat to its fixed table position. */
function seatToPlayer(seat: Seat, dealer: Player): Player {
  return (dealer + SEAT_OFFSET[seat]) % 4;
}

function teamShort(team: TeamId): string {
  return team === 0 ? 'N/S' : 'E/W';
}

/**
 * The hover tooltip for one assist option: the probability it is the best move
 * (the headline number), with the raw network score underneath. Returns
 * undefined when there is no hint, so no `title` attribute is rendered.
 */
function assistTip(scored: ScoredAction | undefined): string | undefined {
  if (!scored) return undefined;
  const pct = scored.probability * 100;
  const shown = pct >= 1 ? `${Math.round(pct)}%` : '<1%';
  return `Best play: ${shown}\nRaw score: ${scored.score.toFixed(2)}`;
}

/** Whether rendering this message advances the pacing clock (a visible beat). */
function paceSetting(msg: ServerMsg): boolean {
  switch (msg.type) {
    case 'UPDATE': // any action: bid, pass, discard, or play
    case 'TRICK_WON':
    case 'HAND_COMPLETE':
    case 'GAME_OVER': // so the lobby return is timed from the result showing
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
  /** The joined table's short code, shown in the lobby. */
  tableCode = $state<string | null>(null);
  /** This client's seat, or `null` while it has not sat down. */
  mySeat = $state<Player | null>(null);
  /** Who occupies each seat, used by the lobby. */
  seats = $state<SeatInfo[]>(playerArray<SeatInfo>(() => ({ type: 'Empty' })));
  /** Per-seat display name (or `null` if empty), used by the game board. */
  players = $state<({ name: string } | null)[]>(playerArray<{ name: string } | null>(() => null));
  dealer = $state<Player | null>(null);

  // --- bidding ------------------------------------------------------------
  upCard = $state<CardCode | null>(null);
  upCardLive = $state(false);

  // --- contract -----------------------------------------------------------
  trump = $state<Suit | null>(null);
  maker = $state<Player | null>(null);
  alone = $state(false);

  // --- the play -----------------------------------------------------------
  hand = $state<CardCode[]>([]);
  /** Cards resting on the felt for the trick in progress. */
  table = $state<{ seat: Player; card: CardCode }[]>([]);
  /** A finished trick mid-sweep toward its winner (kept separate so the new
   *  trick can start building underneath the animation). */
  collecting = $state<{ plays: { seat: Player; card: CardCode }[]; winner: Player } | null>(null);

  // --- turn ---------------------------------------------------------------
  whoseTurn = $state<Player | null>(null);
  hint = $state<TurnHint | null>(null);
  legal = $state<CardCode[] | null>(null);

  // --- assist mode --------------------------------------------------------
  // Populated by a server `SUGGEST` (only when the server runs with assist mode
  // on, and only for our own turn). The UI outlines the recommended option and
  // shows each option's raw network score on hover. Both stay null with assist
  // off, leaving the board unchanged.
  /** The move the neural agent recommends this turn, or null. */
  suggestRecommended = $state<SuggestedAction | null>(null);
  /** Every option the agent scored this turn, or null. */
  suggestScores = $state<ScoredAction[] | null>(null);

  // --- bookkeeping --------------------------------------------------------
  scores = $state<TeamScores>({ north_south: 0, east_west: 0 });
  /** The match target; the server uses 10 by default and does not send it. */
  readonly targetScore = 10;
  tricksWon = $state<number[]>(playerArray(() => 0));
  cardsLeft = $state<number[]>(playerArray(() => 5));
  bubbles = $state<(ActionBubble | null)[]>(playerArray<ActionBubble | null>(() => null));

  // --- transient notices --------------------------------------------------
  banner = $state<string | null>(null);
  toast = $state<string | null>(null);
  gameOver = $state<{ winner: TeamId; scores: TeamScores } | null>(null);

  // --- private ------------------------------------------------------------
  private ws: WebSocket | null = null;
  private name = 'You';
  /** The table to join on HELLO, or `null` to create a fresh one. */
  private joinTable: string | null = null;
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
  get sittingOut(): Player | null {
    return this.alone && this.maker !== null ? partnerOf(this.maker) : null;
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
  /** Connect and either create a table (`table === null`) or join `table`. */
  connect(url: string, name: string, table: string | null): void {
    this.name = name.trim() || 'You';
    this.joinTable = table;
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
      this.send({ type: 'HELLO', name: this.name, table: this.joinTable ?? undefined });
    };
    ws.onmessage = (event) => {
      try {
        this.enqueue(JSON.parse(event.data as string) as ServerMsg);
      } catch (err) {
        console.error('failed to handle server message', err);
      }
    };
    ws.onerror = () => {
      if (this.status !== 'lobby' && this.status !== 'playing') {
        this.status = 'error';
        this.error = 'Could not reach the server. Is euchre-server running?';
      }
    };
    ws.onclose = () => {
      this.clearTimers();
      this.resetQueue();
      if (this.status === 'playing') {
        this.status = 'closed';
      } else if (this.status !== 'error') {
        this.status = 'error';
        this.error ??= 'The connection closed.';
      }
    };
  }

  // --- lobby actions ------------------------------------------------------
  /** Take `seat` for yourself. */
  sit(seat: Player): void {
    this.send({ type: 'SEAT', seat, player: { type: 'Self' } });
  }
  /** Fill the empty `seat` with a bot. */
  addBot(seat: Player): void {
    this.send({ type: 'SEAT', seat, player: { type: 'Bot' } });
  }
  /** Empty `seat` (your own or a bot's). */
  vacate(seat: Player): void {
    this.send({ type: 'SEAT', seat, player: { type: 'Empty' } });
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
      case 'TABLE_STATE':
        // After a match, let the result sit before returning to the lobby; at
        // join time (no result showing) switch immediately.
        return this.gameOver ? GAME_OVER_LINGER_MS : 0;
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
      case 'TABLE_STATE': {
        this.tableCode = msg.table;
        this.mySeat = msg.your_seat;
        this.seats = msg.seats;
        this.players = msg.seats.map((s) => (s.type === 'Empty' ? null : { name: s.name }));
        this.gameOver = null;
        this.status = 'lobby';
        break;
      }
      case 'START_GAME': {
        this.dealer = msg.first_dealer;
        this.scores = { north_south: 0, east_west: 0 };
        this.gameOver = null;
        this.status = 'playing';
        break;
      }
      case 'SYNC': {
        const v = msg.view;
        this.dealer = v.dealer;
        this.scores = this.fixedScores(v.scores.us, v.scores.them);
        if (v.contract) {
          this.trump = v.contract.trump;
          this.maker = seatToPlayer(v.contract.maker, v.dealer);
          this.alone = v.contract.alone;
        }
        this.hand = sortHand(v.hand, this.trump);
        this.table = v.current_trick.plays.map((p) => ({
          seat: seatToPlayer(p.seat, v.dealer),
          card: p.card,
        }));
        const won = playerArray(() => 0);
        for (const [, winner] of v.completed_tricks) won[seatToPlayer(winner, v.dealer)] += 1;
        this.tricksWon = won;
        const left = playerArray(() => 5 - v.completed_tricks.length);
        for (const p of v.current_trick.plays) left[seatToPlayer(p.seat, v.dealer)] -= 1;
        if (this.sittingOut !== null) left[this.sittingOut] = 0;
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
        this.tricksWon = playerArray(() => 0);
        this.cardsLeft = playerArray(() => 5);
        this.bubbles = playerArray<ActionBubble | null>(() => null);
        this.banner = null;
        this.clearSuggestion();
        break;
      }
      case 'AWAITING': {
        this.whoseTurn = msg.player;
        this.hint = msg.hint;
        this.legal = msg.player === this.mySeat ? (msg.legal ?? null) : null;
        // Drop any previous hint; a fresh SUGGEST (if any) follows immediately.
        this.clearSuggestion();
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
          this.addScore(score.points_awarded);
          this.flashBanner(this.describeHand(score));
        }
        break;
      }
      case 'GAME_OVER': {
        const scores = { north_south: msg.scores[0], east_west: msg.scores[1] };
        this.scores = scores;
        this.gameOver = { winner: msg.winner, scores };
        break;
      }
      case 'ERROR': {
        this.flashToast(msg.message);
        break;
      }
      case 'SUGGEST': {
        // Always arrives right after our own AWAITING; ignore any stray hint for
        // another seat.
        if (msg.player === this.mySeat) {
          this.suggestRecommended = msg.recommended;
          this.suggestScores = msg.scores;
        }
        break;
      }
    }
  }

  private applyAction(player: Player, action: PublicAction): void {
    switch (action.type) {
      case 'BID': {
        this.trump = action.suit;
        this.maker = player;
        this.alone = action.alone;
        this.upCardLive = false;
        const verb = this.lastBidUp ? 'orders up' : 'calls';
        this.setBubble(player, `${verb} ${SUIT_SYMBOL[action.suit]}${action.alone ? ' alone' : ''}`);
        if (this.sittingOut !== null) this.cardsLeft[this.sittingOut] = 0;
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

  // --- scoring ------------------------------------------------------------
  /** Folds a point-of-view (us/them) score into the fixed-team representation. */
  private fixedScores(us: number, them: number): TeamScores {
    const mine = this.mySeat === null ? 0 : teamOf(this.mySeat);
    return mine === 0
      ? { north_south: us, east_west: them }
      : { north_south: them, east_west: us };
  }

  /** Adds a hand's net points (signed toward our team) to the running score. */
  private addScore(points: number): void {
    const mine = this.mySeat === null ? 0 : teamOf(this.mySeat);
    // Exactly one team scores a hand: positive points go to ours, negative to theirs.
    const gain = mine === 0 ? points : -points;
    if (gain >= 0) {
      this.scores = { ...this.scores, north_south: this.scores.north_south + gain };
    } else {
      this.scores = { ...this.scores, east_west: this.scores.east_west - gain };
    }
  }

  /** A short caption for how the just-finished hand scored. */
  private describeHand(score: HandScore): string {
    const points = Math.abs(score.points_awarded);
    const euchred = score.maker_tricks < 3;
    const march = score.maker_tricks === 5;
    const makerTeam = this.maker === null ? 0 : teamOf(this.maker);
    if (euchred) return `${teamShort((makerTeam + 1) % 2)} euchred the makers for ${points}!`;
    if (march) return `${teamShort(makerTeam)} swept the hand${this.alone ? ' alone' : ''} for ${points}!`;
    return `${teamShort(makerTeam)} made it for ${points}.`;
  }

  // --- helpers ------------------------------------------------------------
  /** Lift the finished trick off the felt and fly it to the winner. */
  private sweep(winner: Player): void {
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
    this.clearSuggestion();
  }

  // --- assist mode --------------------------------------------------------
  private clearSuggestion(): void {
    this.suggestRecommended = null;
    this.suggestScores = null;
  }

  /** The assist entry for a card (play or discard), or undefined if none. */
  private scoredForCard(code: CardCode): ScoredAction | undefined {
    return this.suggestScores?.find(
      (s) => (s.action.type === 'PLAY' || s.action.type === 'DISCARD') && s.action.card === code,
    );
  }

  /** The assist entry for naming `suit` (with/without `alone`), or undefined. */
  private scoredForBid(suit: Suit, alone: boolean): ScoredAction | undefined {
    return this.suggestScores?.find(
      (s) => s.action.type === 'BID' && s.action.suit === suit && s.action.alone === alone,
    );
  }

  /** The assist entry for passing, or undefined. */
  private scoredForPass(): ScoredAction | undefined {
    return this.suggestScores?.find((s) => s.action.type === 'PASS');
  }

  /** Whether the agent's recommended move is to play or bury `code`. */
  isRecommendedCard(code: CardCode): boolean {
    const r = this.suggestRecommended;
    return !!r && (r.type === 'PLAY' || r.type === 'DISCARD') && r.card === code;
  }

  /** Whether the agent's recommended move is to name `suit` (matching `alone`). */
  isRecommendedBid(suit: Suit, alone: boolean): boolean {
    const r = this.suggestRecommended;
    return !!r && r.type === 'BID' && r.suit === suit && r.alone === alone;
  }

  /** Whether the agent's recommended move is to pass. */
  isRecommendedPass(): boolean {
    return this.suggestRecommended?.type === 'PASS';
  }

  /** A hover tooltip for a card's assist hint, or undefined when there is none. */
  cardTip(code: CardCode): string | undefined {
    return assistTip(this.scoredForCard(code));
  }

  /** A hover tooltip for a bid button's assist hint, or undefined. */
  bidTip(suit: Suit, alone: boolean): string | undefined {
    return assistTip(this.scoredForBid(suit, alone));
  }

  /** A hover tooltip for the pass button's assist hint, or undefined. */
  passTip(): string | undefined {
    return assistTip(this.scoredForPass());
  }

  private setBubble(seat: Player, text: string): void {
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
