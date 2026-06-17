//! # euchre-interface
//!
//! Shared types and the [`Agent`] trait that define the interface between a
//! Euchre game engine and an AI bot that plays the game.
//!
//! This crate is deliberately engine-agnostic: it describes *what* an agent is
//! asked and *what* it answers, not how the game is run. A driver (such as the
//! `euchre-engine` crate) deals cards, enforces the rules, tracks score, and
//! calls into an [`Agent`] at each decision point.
//!
//! ## The card game
//!
//! Euchre is a four-player, two-versus-two, trick-taking game played with a
//! 24-card deck (Nine through Ace in each suit). Partners sit across from each
//! other. Each hand has a *bidding* phase that fixes the trump suit and a
//! *play* phase of five tricks. The defining quirk is the trump ordering: the
//! Jack of the trump suit (the **right bower**) and the Jack of the same color
//! (the **left bower**) outrank everything else, and the left bower plays as a
//! trump rather than as its printed suit.
//!
//! ## Implementing an agent
//!
//! Implement [`Agent`] and supply a decision at each callback:
//!
//! ```
//! use euchre_interface::{Agent, Bid, CallBid, Card, GameView, Suit, UpcardBid};
//!
//! struct PassiveBot;
//!
//! impl Agent for PassiveBot {
//!     fn bid_upcard(&mut self, _view: &GameView<'_>, _up_card: Card) -> UpcardBid {
//!         UpcardBid::Pass
//!     }
//!
//!     fn bid_call(&mut self, _view: &GameView<'_>, turned_down: Suit) -> CallBid {
//!         // Forced to call on the last seat? Pick any legal suit; otherwise pass.
//!         CallBid::Pass
//!     }
//!
//!     fn discard(&mut self, view: &GameView<'_>) -> Card {
//!         view.hand[0]
//!     }
//!
//!     fn play_card(&mut self, _view: &GameView<'_>, legal: &[Card]) -> Card {
//!         legal[0]
//!     }
//! }
//! ```

pub mod agent;
pub mod card;
pub mod game;

pub use agent::{Agent, Bid, CallBid, HandResult, HandScore, UpcardBid};
pub use card::{Card, Color, Rank, Suit};
pub use game::{Contract, GameRules, GameView, Play, Scores, Seat, Team, Trick};
