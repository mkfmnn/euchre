//! Round-robin tournaments over a pool of agents.
//!
//! Where [`runner`](crate::runner) compares two agents, this module runs every
//! agent against every other and aggregates the lot into a win matrix ready for
//! [`elo::fit`](crate::elo::fit). Each pairing reuses the duplicate-dealing
//! [`run_pair`](crate::runner::run_pair) machinery — so deal luck cancels within
//! a pairing exactly as in a head-to-head — and each pairing draws from a
//! disjoint band of deck seeds so the results across pairings are independent,
//! which is what the rating model assumes.
//!
//! ```no_run
//! use euchre_eval::{builtin, tournament::run_round_robin};
//! use euchre_eval::elo::{fit, leaderboard, EloOptions};
//! use euchre_engine::GameConfig;
//!
//! let pool = ["random", "heuristic", "advanced"]
//!     .map(|n| builtin(n).unwrap());
//! let results = run_round_robin(GameConfig::default(), &pool, 200, 0);
//! for r in leaderboard(fit(&results.names, &results.wins_matrix(), &EloOptions::default())) {
//!     println!("{:<12} {:+.0}", r.name, r.elo);
//! }
//! ```

use euchre_engine::GameConfig;

use crate::runner::{Contestant, HeadToHead, run_pair};

/// One completed pairing: the two contestants (by index into
/// [`TournamentResults::names`]) and their head-to-head record, with `a` seated
/// as the first agent.
#[derive(Debug, Clone, Copy)]
pub struct Pairing {
    /// Index of the first agent.
    pub a: usize,
    /// Index of the second agent.
    pub b: usize,
    /// The duplicate-dealing record, `a` as the first agent and `b` the second.
    pub record: HeadToHead,
}

/// The aggregated outcome of a round-robin: who played, and how every pairing
/// went.
#[derive(Debug, Clone)]
pub struct TournamentResults {
    /// Contestant names, indexed as referenced by every [`Pairing`].
    pub names: Vec<String>,
    /// Every unordered pairing's record, with the lower index seated first.
    pub pairings: Vec<Pairing>,
}

impl TournamentResults {
    /// Builds the square win matrix for [`elo::fit`](crate::elo::fit), where
    /// entry `[i][j]` is the number of matches `i` won against `j`.
    pub fn wins_matrix(&self) -> Vec<Vec<u64>> {
        let n = self.names.len();
        let mut wins = vec![vec![0u64; n]; n];
        for p in &self.pairings {
            wins[p.a][p.b] += p.record.a_wins();
            wins[p.b][p.a] += p.record.b_wins();
        }
        wins
    }

    /// The number of contestants.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the tournament had no contestants.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Enumerates the unordered pairings `(i, j)` with `i < j` for `n` contestants,
/// in a stable order (the order in which a round-robin plays them).
pub fn matchups(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).flat_map(move |i| (i + 1..n).map(move |j| (i, j)))
}

/// Plays one pairing: `pairs` duplicate deals of `a` (first) against `b`.
///
/// `base_seed` is the deck seed of the pairing's first pair; successive pairs
/// step it by one, so a pairing occupies the contiguous seed band
/// `[base_seed, base_seed + pairs)`. The round-robin gives each pairing a
/// disjoint band so no two pairings share decks.
pub fn play_pairing(
    config: GameConfig,
    base_seed: u64,
    pairs: u64,
    a: &Contestant,
    b: &Contestant,
) -> HeadToHead {
    let mut h2h = HeadToHead::default();
    for p in 0..pairs {
        h2h.record(run_pair(
            config,
            base_seed.wrapping_add(p),
            &a.factory,
            &b.factory,
        ));
    }
    h2h
}

/// Runs a full round-robin: every contestant against every other for `pairs`
/// duplicate-dealing pairs apiece (so `2 * pairs` matches per pairing).
///
/// Pairings are seeded into disjoint deck-seed bands derived from `base_seed`,
/// keeping their results independent. With fewer than two contestants there is
/// nothing to play and the result has no pairings.
pub fn run_round_robin(
    config: GameConfig,
    contestants: &[Contestant],
    pairs: u64,
    base_seed: u64,
) -> TournamentResults {
    let names = contestants.iter().map(|c| c.name.clone()).collect();
    let mut pairings = Vec::new();
    for (idx, (i, j)) in matchups(contestants.len()).enumerate() {
        let pairing_seed = base_seed.wrapping_add(idx as u64 * pairs);
        let record = play_pairing(
            config,
            pairing_seed,
            pairs,
            &contestants[i],
            &contestants[j],
        );
        pairings.push(Pairing { a: i, b: j, record });
    }
    TournamentResults { names, pairings }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::AgentFactory;
    use euchre_agents::{HeuristicAgent, RandomAgent};
    use euchre_interface::Agent;

    fn contestant(name: &str) -> Contestant {
        let factory: AgentFactory = match name {
            "heuristic" => Box::new(|_| Box::new(HeuristicAgent::new()) as Box<dyn Agent>),
            _ => Box::new(|seed| Box::new(RandomAgent::with_seed(seed)) as Box<dyn Agent>),
        };
        Contestant::new(name, factory)
    }

    #[test]
    fn matchups_are_the_unordered_pairs() {
        assert_eq!(matchups(1).count(), 0);
        assert_eq!(
            matchups(4).collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        );
    }

    #[test]
    fn round_robin_plays_every_pair_once() {
        let pool = [
            contestant("random"),
            contestant("heuristic"),
            contestant("r2"),
        ];
        let results = run_round_robin(GameConfig::default(), &pool, 5, 0);
        assert_eq!(results.len(), 3);
        assert_eq!(results.pairings.len(), 3); // C(3, 2)
        // Every pairing played the requested number of games.
        for p in &results.pairings {
            assert_eq!(p.record.games(), 10);
        }
    }

    #[test]
    fn win_matrix_conserves_games() {
        let pool = [
            contestant("random"),
            contestant("heuristic"),
            contestant("r2"),
        ];
        let results = run_round_robin(GameConfig::default(), &pool, 8, 1);
        let wins = results.wins_matrix();
        // Each pairing contributes 2 * pairs = 16 games split between its two
        // cells, so every off-diagonal pair of cells sums to 16.
        for (i, j) in matchups(3) {
            assert_eq!(wins[i][j] + wins[j][i], 16);
        }
        // The diagonal stays empty.
        for (i, row) in wins.iter().enumerate() {
            assert_eq!(row[i], 0);
        }
    }

    #[test]
    fn heuristic_tops_a_field_of_random() {
        // In a pool of one heuristic and two random agents, the heuristic should
        // win clearly more of its games than either random agent.
        let pool = [
            contestant("heuristic"),
            contestant("random"),
            contestant("r2"),
        ];
        let results = run_round_robin(GameConfig::default(), &pool, 60, 7);
        let wins = results.wins_matrix();
        let total: Vec<u64> = (0..3).map(|i| (0..3).map(|j| wins[i][j]).sum()).collect();
        assert!(
            total[0] > total[1] && total[0] > total[2],
            "totals {total:?}"
        );
    }
}
