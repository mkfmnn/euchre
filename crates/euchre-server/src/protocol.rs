//! The websocket wire protocol: the JSON messages exchanged between a client and
//! the server.
//!
//! The protocol is **event-sourced**. A client learns its hand from a [`Deal`]
//! and then derives the rest of the game state from a stream of [`Update`] and
//! [`TrickWon`] events. A single [`Sync`] snapshot exists only for join /
//! reconnect / spectator resync — the one thing an event stream handles poorly.
//!
//! [`Deal`]: ServerMsg::Deal
//! [`Update`]: ServerMsg::Update
//! [`TrickWon`]: ServerMsg::TrickWon
//! [`Sync`]: ServerMsg::Sync
//!
//! ## The action vocabulary
//!
//! Players bid, pass, discard, and play. The client expresses these with the
//! action variants of [`ClientMsg`]; the server rebroadcasts what happened with
//! [`PublicAction`]. The two are the same vocabulary seen from two sides — the
//! key difference is the **discard**: the client's discard names the buried
//! card, but the rebroadcast [`PublicAction::Discard`] carries nothing, so the
//! buried card stays secret. This mirrors hidden-information euchre exactly.
//!
//! All messages are tagged JSON: a `"type"` field selects the variant, in
//! `SCREAMING_SNAKE_CASE`. Cards are the compact two-letter codes from
//! [`Card`](euchre_interface::Card) (e.g. `"JS"`, `"TH"`).

use euchre_interface::{Card, Contract, GameRules, HandResult, Scores, Seat, Suit, Trick};
use serde::{Deserialize, Serialize};

/// A fixed table position, stable across the whole match: `0` = North, `1` = East,
/// `2` = South, `3` = West (partners are 0/2 and 1/3). Unlike the engine's
/// dealer-relative [`Seat`], a player keeps the same `Player` for every hand, so
/// it is the right identity to put on the wire.
pub type Player = u8;

/// A fixed team identity on the wire: `0` = North/South, `1` = East/West.
pub type TeamId = u8;

/// A message from a client to the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ClientMsg {
    /// The first message a client must send: announce a display name and,
    /// optionally, the code of a table to join. When `table` is omitted the
    /// server creates a fresh table; otherwise it joins the named one (or
    /// replies with an [`ServerMsg::Error`] if no such table exists). Either way
    /// the server replies with a [`ServerMsg::TableState`].
    Hello {
        name: String,
        #[serde(default)]
        table: Option<String>,
    },
    /// In the lobby, request a change to `seat`: take it yourself, fill it with a
    /// bot, or empty it. Broadcasts a fresh [`ServerMsg::TableState`] to everyone.
    Seat { seat: Player, player: SeatRequest },
    /// Bid: in the first round, order up the up-card's suit; in the second,
    /// name `suit` as trump. `alone` requests going it alone.
    Bid { suit: Suit, alone: bool },
    /// Decline to bid (in either round).
    Pass,
    /// As dealer, bury `card` after taking the up-card.
    Discard { card: Card },
    /// Play `card` to the current trick.
    Play { card: Card },
}

/// What a [`ClientMsg::Seat`] asks the server to put at a seat. `Bot` is a
/// struct-style variant so a future difficulty selector can be added without a
/// protocol break.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SeatRequest {
    /// The sender takes the seat, leaving any seat they currently hold.
    #[serde(rename = "Self")]
    Me,
    /// Fill the (empty) seat with a server-side bot.
    Bot,
    /// Empty the seat (only one's own seat or a bot seat).
    Empty,
}

/// A message from the server to a client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ServerMsg {
    /// The lobby snapshot: the table's code, which seat (if any) this connection
    /// holds, and who occupies each of the four seats. Broadcast (per-connection,
    /// so `your_seat` is right for each) whenever seating changes, and again when
    /// a match ends and the table returns to the lobby.
    TableState {
        table: String,
        your_seat: Option<Player>,
        seats: [SeatInfo; 4],
    },
    /// The lobby filled and a match is beginning. `first_dealer` deals the first
    /// hand; the usual [`Deal`](ServerMsg::Deal) stream follows.
    StartGame { first_dealer: Player },
    /// A fresh hand has been dealt. Sent privately — `hand` is this client's
    /// cards only.
    Deal {
        dealer: Player,
        hand: Vec<Card>,
        up_card: Card,
    },
    /// It is `player`'s turn. Broadcast to everyone so all clients can show
    /// whose turn it is; `legal`, the cards this seat may play, is populated
    /// only in the active player's own copy (and only when a card is due).
    Awaiting {
        player: Player,
        hint: TurnHint,
        #[serde(skip_serializing_if = "Option::is_none")]
        legal: Option<Vec<Card>>,
    },
    /// `player` took the action `action`. Broadcast to everyone.
    Update {
        player: Player,
        action: PublicAction,
    },
    /// `player` won the trick just completed.
    TrickWon { player: Player },
    /// The hand ended; `result` is how it scored (or that it was passed out),
    /// told from the receiving client's own team's point of view.
    HandComplete { result: HandResult },
    /// The match ended. `winner` is the winning team and `scores` is the final
    /// score by team (index 0 = North/South).
    GameOver { winner: TeamId, scores: [u8; 2] },
    /// Something the client sent could not be applied (bad message, illegal
    /// move, out of turn). Advisory — the server re-sends [`Awaiting`] so the
    /// client can try again.
    ///
    /// [`Awaiting`]: ServerMsg::Awaiting
    Error { message: String },
    /// A full snapshot of the game from this client's seat, for join/reconnect
    /// resync. Defined for forward use; the walking skeleton sends it on join.
    Sync { view: PlayerView },
    /// An assist hint for the active player: the move the neural agent
    /// recommends and the raw network score of every option it weighed. Sent —
    /// only when the server runs with assist mode enabled (the `EUCHRE_ASSIST`
    /// environment variable) — privately to the seat on turn, right after the
    /// matching [`Awaiting`](ServerMsg::Awaiting). A client outlines the
    /// `recommended` option and surfaces each option's `score` on hover; with
    /// assist disabled no `Suggest` is ever sent.
    Suggest {
        player: Player,
        recommended: SuggestedAction,
        scores: Vec<ScoredAction>,
    },
}

/// A move the assist net can recommend or score, spelled in the client's own
/// action vocabulary so the UI can match it straight to a button or card.
///
/// Unlike the public [`PublicAction::Discard`], a [`SuggestedAction::Discard`]
/// names its card: a suggestion is private to the seat it is sent to, so there
/// is no hidden information to protect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SuggestedAction {
    /// Name `suit` as trump (order up in round one, call in round two),
    /// optionally alone.
    Bid { suit: Suit, alone: bool },
    /// Decline to bid.
    Pass,
    /// Bury `card` as the dealer.
    Discard { card: Card },
    /// Play `card` to the trick.
    Play { card: Card },
}

/// One option the assist net weighed, paired with its raw logit. Higher is
/// better; the scores are not probabilities, so only their ordering and
/// relative gaps are meaningful.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredAction {
    pub action: SuggestedAction,
    pub score: f32,
}

/// What kind of decision the active seat must make, carried by
/// [`ServerMsg::Awaiting`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TurnHint {
    /// A bid is due. `up` is true in the first round (order up the up-card) and
    /// false in the second (name a suit). `may_pass` is false only when "stick
    /// the dealer" forces the dealer to name trump.
    Bid { up: bool, may_pass: bool },
    /// The dealer must discard a card after taking the up-card.
    Discard,
    /// A card is due. `lead` is the led suit of the current trick, or `None` if
    /// this seat leads.
    Play { lead: Option<Suit> },
}

/// A player action as broadcast to everyone — the public face of a
/// [`ClientMsg`] action. Crucially, [`PublicAction::Discard`] hides which card
/// was buried.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PublicAction {
    /// Named `suit` as trump (an order-up in round one or a call in round two),
    /// optionally going alone.
    Bid { suit: Suit, alone: bool },
    /// Declined to bid.
    Pass,
    /// Buried a card (which card is intentionally not revealed).
    Discard,
    /// Played `card` to the trick.
    Play { card: Card },
}

/// Who occupies a seat, as listed in a [`ServerMsg::TableState`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SeatInfo {
    /// Nobody is in the seat.
    Empty,
    /// A server-side bot, with its display name.
    Bot { name: String },
    /// A connected human, with their display name.
    Human { name: String },
}

/// An owned snapshot of everything a seat may see, for [`ServerMsg::Sync`].
///
/// The owned analogue of the engine's borrowing
/// [`GameView`](euchre_interface::GameView). The top-level `seat` and `dealer` are
/// fixed table positions; the seats *inside* the trick history
/// (`current_trick`, `completed_tricks`) are the engine's dealer-relative
/// [`Seat`]s, and `scores` is told from this client's point of view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerView {
    pub seat: Player,
    pub dealer: Player,
    pub hand: Vec<Card>,
    pub contract: Option<Contract>,
    pub current_trick: Trick,
    pub completed_tricks: Vec<(Trick, Seat)>,
    pub scores: Scores,
    pub rules: GameRules,
}
