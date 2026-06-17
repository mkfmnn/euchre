//! TypeScript mirror of the euchre-server wire protocol.
//
// These types match the JSON produced by `serde` in `euchre-server`'s
// `protocol.rs` (and the engine-interface types it reuses):
//
//   * enums tagged with a `"type"` (or, for `TurnHint`, `"kind"`) field in
//     `SCREAMING_SNAKE_CASE`;
//   * `Card`s as the compact two-letter codes (e.g. `"JS"`, `"TH"`);
//   * `Suit` / `Seat` / `Team` as their Rust variant names (`"Hearts"`,
//     `"North"`, `"NorthSouth"`);
//   * `HandResult` as an externally-tagged enum (`"PassedOut"` or
//     `{ "Played": HandScore }`), and tuples (`points_awarded`) as arrays.

/** A suit, spelled as its Rust variant name. */
export type Suit = 'Clubs' | 'Diamonds' | 'Hearts' | 'Spades';

/** A seat at the table, clockwise N → E → S → W. */
export type Seat = 'North' | 'East' | 'South' | 'West';

/** A partnership: North/South or East/West. */
export type Team = 'NorthSouth' | 'EastWest';

/** A card's compact two-letter wire code, e.g. `"JS"`, `"TH"`, `"9C"`. */
export type CardCode = string;

export interface SeatedPlayer {
  seat: Seat;
  name: string;
  /** Whether this seat is filled by a server-side bot. */
  bot: boolean;
}

export interface Scores {
  north_south: number;
  east_west: number;
}

export interface GameRules {
  stick_the_dealer: boolean;
}

export interface Contract {
  trump: Suit;
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

export interface HandScore {
  makers: Team;
  maker_tricks: number;
  euchred: boolean;
  march: boolean;
  alone: boolean;
  /** `[team, points]`. */
  points_awarded: [Team, number];
}

/** Externally-tagged: the string `"PassedOut"` or `{ Played: HandScore }`. */
export type HandResult = 'PassedOut' | { Played: HandScore };

export interface PlayerView {
  seat: Seat;
  dealer: Seat;
  hand: CardCode[];
  contract: Contract | null;
  current_trick: Trick;
  /** `[trick, winner]` pairs, oldest first. */
  completed_tricks: [Trick, Seat][];
  scores: Scores;
  rules: GameRules;
}

/** A message from the server to this client. */
export type ServerMsg =
  | { type: 'JOINED'; players: SeatedPlayer[]; your_seat: Seat; first_dealer: Seat }
  | { type: 'DEAL'; dealer: Seat; hand: CardCode[]; up_card: CardCode }
  | { type: 'AWAITING'; player: Seat; hint: TurnHint; legal?: CardCode[] }
  | { type: 'UPDATE'; player: Seat; action: PublicAction }
  | { type: 'TRICK_WON'; player: Seat }
  | { type: 'HAND_COMPLETE'; result: HandResult }
  | { type: 'GAME_OVER'; winner: Team; scores: Scores }
  | { type: 'ERROR'; message: string }
  | { type: 'SYNC'; view: PlayerView };

/** A message from this client to the server. */
export type ClientMsg =
  | { type: 'HELLO'; name: string; seat?: Seat | null }
  | { type: 'BID'; suit: Suit; alone: boolean }
  | { type: 'PASS' }
  | { type: 'DISCARD'; card: CardCode }
  | { type: 'PLAY'; card: CardCode };
