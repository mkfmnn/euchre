//! # euchre-engine
//!
//! A Euchre game engine in two layers:
//!
//! * The [`game`] module is the **core**: a deterministic state machine
//!   ([`Game`]) that holds the authoritative match state, reports what decision
//!   is needed next, and applies decisions back in. It is engine-only — it
//!   never reads input or prints — which makes it equally suitable for driving a
//!   terminal loop or a websocket server that fans requests out to up to four
//!   connected clients.
//!
//! * The [`driver`] module is the terminal **driver**: a [`Driver`] that wires
//!   the core to four [`Player`]s (each an AI [`Agent`] or a human at the
//!   keyboard) and runs a whole match, narrating at a chosen [`Verbosity`].
//!
//! Decisions are expressed through the [`Agent`] trait and value types of the
//! companion [`euchre_interface`] crate; this crate adds the rules, scoring, and
//! orchestration. It deliberately ships **no agents** — concrete strategies live
//! in a separate crate that plugs into the [`Player::Bot`] slot.
//!
//! ## Running four bots
//!
//! ```no_run
//! use euchre_engine::{Driver, GameConfig, Player, Verbosity};
//! # use euchre_interface::{Agent, Card, CallBid, GameView, UpcardBid};
//! # struct MyBot;
//! # impl Agent for MyBot {
//! #     fn bid_upcard(&mut self, _v: &GameView<'_>) -> UpcardBid { UpcardBid::Pass }
//! #     fn bid_call(&mut self, _v: &GameView<'_>) -> CallBid { CallBid::Pass }
//! #     fn discard(&mut self, v: &GameView<'_>) -> Card { v.hand[0] }
//! #     fn play_card(&mut self, _v: &GameView<'_>, legal: &[Card]) -> Card { legal[0] }
//! # }
//! let mut bots = [MyBot, MyBot, MyBot, MyBot];
//! let [a, b, c, d] = &mut bots;
//! let players = [Player::Bot(a), Player::Bot(b), Player::Bot(c), Player::Bot(d)];
//!
//! let outcome = Driver::headless(
//!     GameConfig::default(),
//!     players,
//!     Verbosity::Hand,
//!     std::io::stdout(),
//! )
//! .run()
//! .unwrap();
//! println!("{:?} won", outcome.winner);
//! ```
//!
//! ## Driving the core directly (e.g. a server)
//!
//! ```
//! use euchre_engine::{Action, Decision, Game, GameConfig};
//! use euchre_interface::{CallBid, Card, UpcardBid};
//!
//! let deck: [Card; 24] = Card::deck().try_into().unwrap();
//! let mut game = Game::new(GameConfig::default(), deck);
//! loop {
//!     match game.next_action() {
//!         Action::BidUpcard { seat, .. } => {
//!             let _view = game.view(seat); // send to the client for `seat`
//!             game.apply(Decision::Upcard(UpcardBid::Pass)).unwrap();
//!         }
//!         Action::BidCall { .. } => {
//!             // Everyone passes, so the hand is thrown in (stick-the-dealer off).
//!             game.apply(Decision::Call(CallBid::Pass)).unwrap();
//!         }
//!         Action::HandComplete { .. } => break,
//!         _ => unreachable!("no cards are played when every seat passes"),
//!     }
//! }
//! ```

pub mod driver;
pub mod game;
pub mod shuffle;

pub use driver::{Driver, Outcome, Player, Verbosity};
pub use game::{Action, ApplyError, Decision, Game, GameConfig};
pub use shuffle::deal;

// Re-export the interface so downstream code can use one crate. The whole crate
// is available as `euchre_engine::interface`, with the most common types lifted
// to the top level for convenience.
pub use euchre_interface as interface;
pub use euchre_interface::{Agent, Card, GameRules, GameView, Scores, Seat, Suit};
