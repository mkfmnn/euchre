//! # euchre-eval
//!
//! An evaluation harness for Euchre [`Agent`](euchre_interface::Agent)s. It
//! exists to answer one question reliably: **did this change make the agent win
//! more matches?**
//!
//! ## What it measures
//!
//! The objective is *match-win probability*, not points. Skilled Euchre is
//! score-aware — an agent should sandbag a lead (trading expected points for
//! lower variance) and gamble from behind — so scoring on points would punish
//! exactly the play you want. Matches are therefore run to their real finish
//! line (the configured target score, conventionally 10) and scored purely on who
//! won.
//!
//! ## How it fights variance
//!
//! A single match is mostly luck. The [`runner`] uses **duplicate dealing**:
//! every deck is played twice with the agents on opposite sides, so deal luck is
//! shared and cancels. The [`stats`] module then reports:
//!
//! * a Wilson confidence interval on the headline win rate;
//! * McNemar's paired test on the duplicate pairs (the variance-reduced verdict);
//! * an optional SPRT that stops as soon as the result is decided.
//!
//! ## Example
//!
//! ```no_run
//! use euchre_eval::{builtin, runner::{HeadToHead, run_pair}};
//! use euchre_engine::GameConfig;
//! use euchre_eval::stats::{mcnemar, wilson_interval, Z_95};
//!
//! let a = builtin("heuristic").unwrap();
//! let b = builtin("random").unwrap();
//! let mut h2h = HeadToHead::default();
//! for seed in 0..500 {
//!     h2h.record(run_pair(GameConfig::default(), seed, &a.factory, &b.factory));
//! }
//! let (lo, hi) = wilson_interval(h2h.a_wins(), h2h.games(), Z_95);
//! let test = mcnemar(h2h.a_better, h2h.b_better);
//! println!("{} win rate {:.1}% ({:.1}–{:.1}%), McNemar p = {:.4}",
//!     a.name, 100.0 * h2h.a_win_rate(), 100.0 * lo, 100.0 * hi, test.p_value);
//! ```
//!
//! ## Ranking a whole pool
//!
//! To compare more than two agents at once, [`tournament`] runs a round-robin —
//! every agent against every other, reusing the same duplicate-dealing pairs —
//! and [`elo`] fits BayesElo-style Bradley-Terry ratings (with uncertainties) to
//! the aggregate, producing a leaderboard. The `euchre-tournament` binary wraps
//! both into a CLI with table, CSV, and JSON output.

pub mod elo;
pub mod runner;
pub mod stats;
pub mod tournament;

use euchre_agents::{
    AdvancedAgent, HeuristicAgent, MonteCarloAgent, NeuralAgent, OpenAiAdvancedAgent, RandomAgent, StrongAgent,
};
use euchre_interface::Agent;
use runner::{AgentFactory, Contestant};

/// Builds a contestant for a built-in agent by name, or `None` if unknown.
///
/// Recognised names are `"random"`, `"heuristic"`, `"advanced"`, `"montecarlo"`,
/// `"montecarlo-play"` (the Monte-Carlo agent with bidding search disabled, so it
/// delegates bidding to the advanced agent), `"neural"` (the learned, search-free
/// policy-network agent), and `"strong"` (a wider, RL-tuned policy network trained
/// to beat the neural champion). New agents should be added here so the CLI and any
/// tooling can name them.
pub fn builtin(name: &str) -> Option<Contestant> {
    let factory: AgentFactory = match name {
        "random" => Box::new(|seed| Box::new(RandomAgent::with_seed(seed)) as Box<dyn Agent>),
        "heuristic" => Box::new(|_seed| Box::new(HeuristicAgent::new()) as Box<dyn Agent>),
        "advanced" => Box::new(|_seed| Box::new(AdvancedAgent::new()) as Box<dyn Agent>),
        "montecarlo" => {
            Box::new(|seed| Box::new(MonteCarloAgent::with_seed(seed)) as Box<dyn Agent>)
        }
        "montecarlo-play" => Box::new(|seed| {
            Box::new(MonteCarloAgent::with_seed(seed).play_only()) as Box<dyn Agent>
        }),
        "neural" => Box::new(|_seed| Box::new(NeuralAgent::pretrained()) as Box<dyn Agent>),
        "strong" => Box::new(|_seed| Box::new(StrongAgent::pretrained()) as Box<dyn Agent>),
        "openai-advanced" => {
            Box::new(|seed| Box::new(OpenAiAdvancedAgent::with_seed(seed)) as Box<dyn Agent>)
        }
        _ => return None,
    };
    Some(Contestant::new(name, factory))
}

/// The names of every built-in agent, for help text and listings.
pub const BUILTIN_AGENTS: &[&str] = &[
    "random",
    "heuristic",
    "advanced",
    "montecarlo",
    "montecarlo-play",
    "neural",
    "openai-advanced",
    "strong",
];
