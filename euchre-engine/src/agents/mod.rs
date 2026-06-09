//! Ready-to-use [`Agent`](euchre_interface::Agent) implementations.
//!
//! Three agents ship with the engine, spanning the spectrum from trivial to
//! interactive:
//!
//! * [`RandomAgent`] — makes legal but uninformed choices. Useful as a sparring
//!   partner, a fuzz source for the engine, and a baseline to measure other
//!   agents against.
//! * [`HeuristicAgent`] — a competent rule-based bot. It evaluates its hand to
//!   decide whether to make trump, when to go alone, and which card to lead or
//!   follow with. Strong enough to give a human a real game.
//! * [`HumanAgent`] — drives the decisions from a text prompt, letting a person
//!   sit at any seat. See [`human`] for wiring it to a terminal.

pub mod heuristic;
pub mod human;
pub mod random;

pub use heuristic::HeuristicAgent;
pub use human::{HumanAgent, Prompter, TerminalPrompter};
pub use random::RandomAgent;
