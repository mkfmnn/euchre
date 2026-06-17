//! # euchre-agents
//!
//! Concrete [`Agent`] implementations that plug into the `euchre-engine`
//! [`Player::Bot`] slot. The engine and interface ship no strategies of their
//! own; this crate supplies them.
//!
//! Five agents are provided:
//!
//! * [`RandomAgent`] — picks uniformly at random among the legal options at
//!   every decision. A baseline opponent and a fuzz source.
//! * [`HeuristicAgent`] — plays a handful of common-sense rules of thumb for
//!   bidding, discarding, leading, and following. No search, but a recognizably
//!   sensible game that reliably beats the random agent.
//! * [`AdvancedAgent`] — a stronger heuristic player that counts cards, bids by
//!   position with the "next"/"green" calling conventions, evaluates hands in
//!   estimated tricks, and plays score-aware. Still no search or learning, but it
//!   beats the plain heuristic.
//! * [`MonteCarloAgent`] — the first *searching* agent. At each card it samples
//!   full deals of the hidden cards consistent with what it has seen, solves each
//!   to a double-dummy optimum, and plays the card that scores best on average
//!   (Perfect-Information Monte Carlo). It reuses [`AdvancedAgent`] for bidding
//!   and beats it in the play.
//! * [`NeuralAgent`] — a *learned*, search-free agent. Four small policy networks
//!   (one per decision) are trained by behavioural cloning of a strong teacher and
//!   then fine-tuned by self-play reinforcement learning, so each move is a single
//!   forward pass yet the agent plays *better* than the teacher it was cloned from.
//!   See the [`neural`] module for the model, the training loop, and the design
//!   rationale.
//!
//! All implement the [`Agent`] trait, so any of them drops into a driver:
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

pub mod advanced;
pub mod heuristic;
pub mod montecarlo;
pub mod neural;
pub mod random;
mod solver;

pub use advanced::AdvancedAgent;
pub use heuristic::HeuristicAgent;
pub use montecarlo::MonteCarloAgent;
pub use neural::NeuralAgent;
pub use random::RandomAgent;
