//! TypeScript mirror of the euchre-server wire protocol.
//
// These types match the JSON produced by `serde` in `euchre-server`'s
// `protocol.rs` (and the engine-interface types it reuses):
//
//   * enums tagged with a `"type"` (or, for `TurnHint`, `"kind"`) field in
//     `SCREAMING_SNAKE_CASE`;
//   * `Card`s as the compact two-letter codes (e.g. `"JS"`, `"TH"`);
//   * `Suit` as its Rust variant name (`"Hearts"`);
//   * `HandResult` as an externally-tagged enum (`"PassedOut"` or
//     `{ "Played": HandScore }`).
//
// ## Two identities on the wire
//
// The server speaks two seat identities, and the distinction matters:
//
//   * `Player` — a **fixed** table position (`0` = North … `3` = West), stable
//     across the whole match. Every top-level message field that names a
//     participant (`your_seat`, `dealer`, `player`, `maker` of the played card)
//     is a `Player`. This is the identity the UI works in.
//   * `Seat` — the engine's **dealer-relative** seat (`First` is the
//     dealer's left, around to `Dealer`), which rotates each hand. It appears
//     only *inside* the trick history of a `SYNC` snapshot (`Play.seat`,
//     `completed_tricks` winners, and a `Contract.maker`). Convert these to a
//     `Player` with the snapshot's `dealer` before use.

/** A suit, spelled as its Rust variant name. */
export type Suit = 'Clubs' | 'Diamonds' | 'Hearts' | 'Spades';

/**
 * A fixed table position, stable for the whole match: `0` = North, `1` = East,
 * `2` = South, `3` = West. Partners are `0`/`2` and `1`/`3`.
 */
export type Player = number;

/**
 * The engine's dealer-relative seat, used only inside a `SYNC` snapshot's trick
 * history. `First` is the dealer's immediate left, then `Second`, `Third`, and
 * `Dealer`. Resolve to a {@link Player} via the snapshot's `dealer`.
 */
export type Seat = 'First' | 'Second' | 'Third' | 'Dealer';

/** A fixed team identity: `0` = North/South, `1` = East/West. */
export type TeamId = number;

/** A card's compact two-letter wire code, e.g. `"JS"`, `"TH"`, `"9C"`. */
export type CardCode = string;

/** Who occupies a seat, as listed in a `TABLE_STATE` message. */
export type SeatInfo =
  | { type: 'Empty' }
  | { type: 'Bot'; name: string }
  | { type: 'Human'; name: string };

/** What a `SEAT` request asks the server to put at a seat. */
export type SeatRequest =
  | { type: 'Self' } // the sender takes the seat
  | { type: 'Bot' } // fill it with a bot
  | { type: 'Empty' }; // empty it

/**
 * Cumulative match score, told from the receiving client's point of view: `us`
 * is the client's own team, `them` the opponents'.
 */
export interface Scores {
  us: number;
  them: number;
}

export interface GameRules {
  stick_the_dealer: boolean;
}

export interface Contract {
  trump: Suit;
  /** Dealer-relative inside a `SYNC` snapshot; resolve with the snapshot's `dealer`. */
  maker: Seat;
  alone: boolean;
}

export interface Play {
  seat: Seat;
  card: CardCode;
}

export interface Trick {
  plays: Play[];
}

/** What kind of decision the active seat must make (`ServerMsg.AWAITING.hint`). */
export type TurnHint =
  | { kind: 'BID'; up: boolean; may_pass: boolean }
  | { kind: 'DISCARD' }
  | { kind: 'PLAY'; lead: Suit | null };

/** A player's action, as broadcast to everyone (a discard hides its card). */
export type PublicAction =
  | { type: 'BID'; suit: Suit; alone: boolean }
  | { type: 'PASS' }
  | { type: 'DISCARD' }
  | { type: 'PLAY'; card: CardCode };

/**
 * A move the assist net can recommend or score, in this client's own action
 * vocabulary. Unlike {@link PublicAction}, a discard names its card: an assist
 * suggestion is private to the seat it is sent to.
 */
export type SuggestedAction =
  | { type: 'BID'; suit: Suit; alone: boolean }
  | { type: 'PASS' }
  | { type: 'DISCARD'; card: CardCode }
  | { type: 'PLAY'; card: CardCode };

/**
 * One option the assist net weighed. `score` is the raw logit (higher is
 * better, but unbounded); `probability` is the softmax of the scores over the
 * legal options — the chance this option is the best move — and the values
 * across one suggestion sum to 1.
 */
export interface ScoredAction {
  action: SuggestedAction;
  score: number;
  probability: number;
}

/**
 * How a played hand scored, told from the receiving client's point of view.
 * `points_awarded` is the net points to the client's own team: positive if it
 * scored, negative if the opponents did.
 */
export interface HandScore {
  maker_tricks: number;
  points_awarded: number;
}

/** Externally-tagged: the string `"PassedOut"` or `{ Played: HandScore }`. */
export type HandResult = 'PassedOut' | { Played: HandScore };

export interface PlayerView {
  /** Fixed table position of the viewer. */
  seat: Player;
  /** Fixed table position of the dealer. */
  dealer: Player;
  hand: CardCode[];
  contract: Contract | null;
  current_trick: Trick;
  /** `[trick, winner]` pairs, oldest first; winners are dealer-relative seats. */
  completed_tricks: [Trick, Seat][];
  scores: Scores;
  rules: GameRules;
}

/** A message from the server to this client. */
export type ServerMsg =
  | { type: 'TABLE_STATE'; table: string; your_seat: Player | null; seats: SeatInfo[] }
  | { type: 'START_GAME'; first_dealer: Player }
  | { type: 'DEAL'; dealer: Player; hand: CardCode[]; up_card: CardCode }
  | { type: 'AWAITING'; player: Player; hint: TurnHint; legal?: CardCode[] }
  | { type: 'UPDATE'; player: Player; action: PublicAction }
  | { type: 'TRICK_WON'; player: Player }
  | { type: 'HAND_COMPLETE'; result: HandResult }
  | { type: 'GAME_OVER'; winner: TeamId; scores: [number, number] }
  | { type: 'ERROR'; message: string }
  | { type: 'SYNC'; view: PlayerView }
  | {
      type: 'SUGGEST';
      player: Player;
      recommended: SuggestedAction;
      scores: ScoredAction[];
    };

/** A message from this client to the server. */
export type ClientMsg =
  | { type: 'HELLO'; name: string; table?: string | null }
  | { type: 'SEAT'; seat: Player; player: SeatRequest }
  | { type: 'BID'; suit: Suit; alone: boolean }
  | { type: 'PASS' }
  | { type: 'DISCARD'; card: CardCode }
  | { type: 'PLAY'; card: CardCode };
