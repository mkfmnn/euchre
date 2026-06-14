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

use euchre_interface::{Card, Contract, GameRules, HandResult, Scores, Seat, Suit, Team, Trick};
use serde::{Deserialize, Serialize};

/// A message from a client to the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ClientMsg {
    /// The first message a client must send: announce a display name and,
    /// optionally, a preferred seat. The server replies with [`ServerMsg::Joined`].
    Hello {
        name: String,
        #[serde(default)]
        seat: Option<Seat>,
    },
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

/// A message from the server to a client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ServerMsg {
    /// Acknowledges a [`ClientMsg::Hello`]: the table roster, the seat assigned
    /// to this client, and who deals the current hand.
    Joined {
        players: Vec<SeatedPlayer>,
        your_seat: Seat,
        first_dealer: Seat,
    },
    /// A fresh hand has been dealt. Sent privately — `hand` is this client's
    /// cards only.
    Deal {
        dealer: Seat,
        hand: Vec<Card>,
        up_card: Card,
    },
    /// It is `player`'s turn. Broadcast to everyone so all clients can show
    /// whose turn it is; `legal`, the cards this seat may play, is populated
    /// only in the active player's own copy (and only when a card is due).
    Awaiting {
        player: Seat,
        hint: TurnHint,
        #[serde(skip_serializing_if = "Option::is_none")]
        legal: Option<Vec<Card>>,
    },
    /// `player` took the action `action`. Broadcast to everyone.
    Update { player: Seat, action: PublicAction },
    /// `player` won the trick just completed.
    TrickWon { player: Seat },
    /// The hand ended; `result` is how it scored (or that it was passed out).
    HandComplete { result: HandResult },
    /// The match ended.
    GameOver { winner: Team, scores: Scores },
    /// Something the client sent could not be applied (bad message, illegal
    /// move, out of turn). Advisory — the server re-sends [`Awaiting`] so the
    /// client can try again.
    ///
    /// [`Awaiting`]: ServerMsg::Awaiting
    Error { message: String },
    /// A full snapshot of the game from this client's seat, for join/reconnect
    /// resync. Defined for forward use; the walking skeleton sends it on join.
    Sync { view: PlayerView },
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

/// A seat's occupant, as listed in [`ServerMsg::Joined`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeatedPlayer {
    pub seat: Seat,
    pub name: String,
    /// Whether this seat is filled by a server-side bot.
    pub bot: bool,
}

/// An owned snapshot of everything a seat may see, for [`ServerMsg::Sync`].
///
/// The owned analogue of the engine's borrowing
/// [`GameView`](euchre_interface::GameView).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerView {
    pub seat: Seat,
    pub dealer: Seat,
    pub hand: Vec<Card>,
    pub contract: Option<Contract>,
    pub current_trick: Trick,
    pub completed_tricks: Vec<(Trick, Seat)>,
    pub scores: Scores,
    pub rules: GameRules,
}
