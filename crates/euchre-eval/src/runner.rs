//! Running matches and pairing them for variance reduction.
//!
//! A single Euchre match is mostly luck, so comparing two agents naively needs a
//! great many games. This module uses **duplicate dealing**: every deal is played
//! twice from the same shuffled deck with the agents on opposite sides, so the
//! luck of the deal is shared between the two games and largely cancels. The unit
//! of comparison stays the whole match at its real finish line (race to the
//! configured target score), because score-aware play is only correct relative to
//! that finish line — shortening matches would change optimal strategy.

use euchre_engine::{Driver, GameConfig, Player, Verbosity};
use euchre_interface::Agent;

/// Builds a fresh agent for one seat, seeded for reproducibility.
///
/// Matches need a clean agent per seat — a stateful learner must not carry state
/// between matches, and a randomised agent needs its own stream — so contestants
/// are supplied as factories rather than instances. The `u64` is a per-seat seed
/// derived by the runner; deterministic agents may ignore it.
pub type AgentFactory = Box<dyn Fn(u64) -> Box<dyn Agent>>;

/// A named contestant: a display name and a factory that builds it.
pub struct Contestant {
    /// How the contestant is shown in results.
    pub name: String,
    /// Builds a fresh instance for a seat.
    pub factory: AgentFactory,
}

impl Contestant {
    /// Pairs a name with its factory.
    pub fn new(name: impl Into<String>, factory: AgentFactory) -> Self {
        Contestant {
            name: name.into(),
            factory,
        }
    }
}

/// Derives an independent per-seat seed from a match seed (a SplitMix64 step), so
/// the four agents in a match draw from disjoint, reproducible streams.
fn seat_seed(match_seed: u64, salt: u64) -> u64 {
    let mut z = match_seed
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt.wrapping_mul(0x2545_F491_4F6C_DD1D));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Plays one match, with `ns` seated North/South and `ew` seated East/West, and
/// returns the winning team index (0 = North/South, 1 = East/West).
///
/// The two seeds are kept separate on purpose. `deck_seed` drives the shuffle and
/// is held fixed across a duplicate pair so deal luck cancels; `agent_seed` drives
/// the agents' own randomness and should differ between the two games of a pair,
/// so that two identically-seeded stochastic agents don't play in lockstep.
/// Deterministic agents ignore `agent_seed`.
pub fn run_match(
    config: GameConfig,
    deck_seed: u64,
    agent_seed: u64,
    ns: &AgentFactory,
    ew: &AgentFactory,
) -> usize {
    let mut north = ns(seat_seed(agent_seed, 0));
    let mut east = ew(seat_seed(agent_seed, 1));
    let mut south = ns(seat_seed(agent_seed, 2));
    let mut west = ew(seat_seed(agent_seed, 3));
    let players = [
        Player::Bot(&mut *north),
        Player::Bot(&mut *east),
        Player::Bot(&mut *south),
        Player::Bot(&mut *west),
    ];
    Driver::with_seed(
        config,
        players,
        Verbosity::Silent,
        std::io::empty(),
        std::io::sink(),
        deck_seed,
    )
    .run()
    .expect("a headless match never fails on I/O")
    .winner
}

/// The result of one duplicate-dealing pair: the same deck played with each agent
/// taking the North/South side once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairResult {
    /// Whether the first agent, seated North/South, won the first game.
    pub a_ns_won: bool,
    /// Whether the second agent, seated North/South, won the second game.
    pub b_ns_won: bool,
}

impl PairResult {
    /// The first agent's match wins across the pair (0, 1, or 2). It plays
    /// North/South in game one and East/West in game two.
    pub fn a_wins(&self) -> u64 {
        u64::from(self.a_ns_won) + u64::from(!self.b_ns_won)
    }

    /// Whether the same cards were played to a better result by the first agent —
    /// a discordant pair favouring it in McNemar's sense.
    pub fn a_better(&self) -> bool {
        self.a_ns_won && !self.b_ns_won
    }

    /// Whether the same cards were played to a better result by the second agent.
    pub fn b_better(&self) -> bool {
        !self.a_ns_won && self.b_ns_won
    }
}

/// Plays a deck twice with the sides swapped, holding the deal fixed.
///
/// Both games use the same shuffled deck (so deal luck cancels) but independent
/// agent-randomness seeds (so stochastic agents are sampled afresh each game).
pub fn run_pair(config: GameConfig, seed: u64, a: &AgentFactory, b: &AgentFactory) -> PairResult {
    let a_ns_won = run_match(config, seed, seat_seed(seed, 10), a, b) == 0;
    let b_ns_won = run_match(config, seed, seat_seed(seed, 11), b, a) == 0;
    PairResult { a_ns_won, b_ns_won }
}

/// Running totals from a head-to-head, in the currency of match wins.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeadToHead {
    /// Duplicate-dealing pairs played.
    pub pairs: u64,
    /// Discordant pairs decided in the first agent's favour (for McNemar): the
    /// same deck swept 2–0 by the first agent.
    pub a_better: u64,
    /// Discordant pairs decided in the second agent's favour (for McNemar).
    pub b_better: u64,
}

impl HeadToHead {
    /// Folds one pair's result into the totals. Every pair is either a sweep for
    /// one side (a discordant pair) or a 1–1 split (a tie), so these three counts
    /// determine every other tally.
    pub fn record(&mut self, pair: PairResult) {
        self.pairs += 1;
        self.a_better += u64::from(pair.a_better());
        self.b_better += u64::from(pair.b_better());
    }

    /// Total games played (`2 * pairs`).
    pub fn games(&self) -> u64 {
        2 * self.pairs
    }

    /// Pairs split 1–1, where each agent won once on the same deck.
    pub fn ties(&self) -> u64 {
        self.pairs - self.a_better - self.b_better
    }

    /// Match wins for the first agent. A swept pair is worth two; a tie, one.
    pub fn a_wins(&self) -> u64 {
        2 * self.a_better + self.ties()
    }

    /// Match wins for the second agent.
    pub fn b_wins(&self) -> u64 {
        2 * self.b_better + self.ties()
    }

    /// The first agent's overall match-win rate.
    pub fn a_win_rate(&self) -> f64 {
        if self.pairs == 0 {
            0.0
        } else {
            self.a_wins() as f64 / self.games() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_agents::{HeuristicAgent, RandomAgent};
    use euchre_interface::Agent;

    fn heuristic() -> AgentFactory {
        Box::new(|_seed| Box::new(HeuristicAgent::new()) as Box<dyn Agent>)
    }

    fn random() -> AgentFactory {
        Box::new(|seed| Box::new(RandomAgent::with_seed(seed)) as Box<dyn Agent>)
    }

    #[test]
    fn seat_seeds_are_distinct_and_reproducible() {
        let seeds: Vec<u64> = (0..4).map(|s| seat_seed(42, s)).collect();
        assert_eq!(seeds, (0..4).map(|s| seat_seed(42, s)).collect::<Vec<_>>());
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(seeds[i], seeds[j]);
            }
        }
    }

    #[test]
    fn a_pair_swaps_the_sides() {
        // The same deck played both ways yields a well-formed pair, and a_wins is
        // in range.
        let pair = run_pair(GameConfig::default(), 7, &heuristic(), &random());
        assert!(pair.a_wins() <= 2);
    }

    #[test]
    fn identical_stochastic_agents_are_not_locked_in_step() {
        // Two independently-seeded random agents must actually disagree on some
        // deals; if the mirror reused the same agent seeds they would move in
        // lockstep and produce zero discordant pairs, defeating A/A testing.
        let (a, b) = (random(), random());
        let mut h2h = HeadToHead::default();
        for seed in 0..200 {
            h2h.record(run_pair(GameConfig::default(), seed, &a, &b));
        }
        assert!(h2h.a_better + h2h.b_better > 0, "no discordant pairs");
        // Roughly a coin flip, with no systematic edge for either side.
        assert!(
            (h2h.a_win_rate() - 0.5).abs() < 0.1,
            "rate {}",
            h2h.a_win_rate()
        );
    }

    #[test]
    fn heuristic_beats_random_in_a_head_to_head() {
        let (a, b) = (heuristic(), random());
        let mut h2h = HeadToHead::default();
        for seed in 0..150 {
            h2h.record(run_pair(GameConfig::default(), seed, &a, &b));
        }
        assert!(
            h2h.a_win_rate() > 0.75,
            "heuristic win rate was only {:.3}",
            h2h.a_win_rate()
        );
        // The duplicate design should overwhelmingly favour the heuristic.
        assert!(h2h.a_better > h2h.b_better);
    }
}
