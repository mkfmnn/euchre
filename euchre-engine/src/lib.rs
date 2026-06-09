//! # euchre-engine
//!
//! A complete, self-contained engine for the card game **Euchre**, built on the
//! engine-agnostic types in the [`euchre_interface`] crate. The engine deals,
//! runs the two-round auction, plays out the five tricks, enforces every rule,
//! and keeps score to the end of a match — calling into [`Agent`]s for the
//! decisions a player must make.
//!
//! It supports both AI and human players interchangeably: any value
//! implementing [`Agent`] can occupy any seat. Three implementations ship in
//! [`agents`]:
//!
//! * [`RandomAgent`](agents::RandomAgent) — a legal-but-uninformed baseline.
//! * [`HeuristicAgent`](agents::HeuristicAgent) — a competent rule-based bot.
//! * [`HumanAgent`](agents::HumanAgent) — a person playing through a text
//!   prompt (or any custom [`Prompter`](agents::Prompter)).
//!
//! ## Running a match between bots
//!
//! ```
//! use euchre_engine::{Engine, EngineConfig};
//! use euchre_engine::agents::{HeuristicAgent, RandomAgent};
//! use euchre_engine::rng::Rng;
//! use euchre_interface::Agent;
//!
//! let agents: [Box<dyn Agent>; 4] = [
//!     Box::new(HeuristicAgent::new()),
//!     Box::new(RandomAgent::new(Rng::seed_from_u64(2))),
//!     Box::new(HeuristicAgent::new()),
//!     Box::new(RandomAgent::new(Rng::seed_from_u64(4))),
//! ];
//! let mut engine = Engine::new(agents, EngineConfig { seed: Some(1), ..Default::default() });
//! let outcome = engine.play_match();
//! println!("{:?} win {}–{}", outcome.winner,
//!     outcome.scores.north_south, outcome.scores.east_west);
//! ```
//!
//! ## Seating a human
//!
//! Drop a [`HumanAgent`](agents::HumanAgent) into any seat; the others can be
//! bots. The included `euchre` binary (`cargo run -p euchre-engine`) wires a
//! human into the South seat against three [`HeuristicAgent`](agents::HeuristicAgent)s.

pub mod agents;
pub mod engine;
pub mod rng;

pub use engine::{
    Engine, EngineConfig, EngineError, HandOutcome, MatchOutcome, legal_plays, score_hand,
};

// Re-export the interface types so downstream users need only depend on this
// crate to build and run a game.
pub use euchre_interface::{
    Agent, Bid, CallBid, Card, Color, Contract, GameView, HandResult, Play, Rank, Scores, Seat,
    Suit, Team, Trick, UpcardBid,
};
