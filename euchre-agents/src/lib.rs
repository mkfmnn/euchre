//! # euchre-agents
//!
//! Concrete [`Agent`] implementations that plug into the `euchre-engine`
//! [`Player::Bot`] slot. The engine and interface ship no strategies of their
//! own; this crate supplies them.
//!
//! Two agents are provided, from simplest to least simple:
//!
//! * [`RandomAgent`] — picks uniformly at random among the legal options at
//!   every decision. A baseline opponent and a fuzz source.
//! * [`HeuristicAgent`] — plays a handful of common-sense rules of thumb for
//!   bidding, discarding, leading, and following. No search, but a recognizably
//!   sensible game that reliably beats the random agent.
//!
//! Both implement the [`Agent`] trait, so either drops into a driver:
//!
//! ```no_run
//! use euchre_agents::{HeuristicAgent, RandomAgent};
//! use euchre_engine::{Driver, GameConfig, Player, Verbosity};
//!
//! let mut north = HeuristicAgent::new();
//! let mut east = RandomAgent::new();
//! let mut south = HeuristicAgent::new();
//! let mut west = RandomAgent::new();
//! let players = [
//!     Player::Bot(&mut north),
//!     Player::Bot(&mut east),
//!     Player::Bot(&mut south),
//!     Player::Bot(&mut west),
//! ];
//!
//! let outcome = Driver::headless(
//!     GameConfig::default(),
//!     players,
//!     Verbosity::Silent,
//!     std::io::stdout(),
//! )
//! .run()
//! .unwrap();
//! println!("{:?} won", outcome.winner);
//! ```
//!
//! [`Agent`]: euchre_interface::Agent
//! [`Player::Bot`]: ../euchre_engine/enum.Player.html

pub mod heuristic;
pub mod random;

pub use heuristic::HeuristicAgent;
pub use random::RandomAgent;
