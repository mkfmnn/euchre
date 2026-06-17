//! Integration tests that run whole matches through the real engine, pitting
//! the [`HeuristicAgent`] against the [`RandomAgent`].
//!
//! These double as a sanity check on the bidding thresholds: a heuristic that
//! tuned itself into never bidding (or always bidding) would stop beating noise,
//! and these tests would catch it.

use euchre_agents::{HeuristicAgent, RandomAgent};
use euchre_engine::{Driver, GameConfig, Player, Team, Verbosity};

/// Runs one match with the heuristic team (North/South) against the random team
/// (East/West) for a given seed, returning the winning team.
fn play_match(seed: u64) -> Team {
    let mut north = HeuristicAgent::new();
    let mut south = HeuristicAgent::new();
    let mut east = RandomAgent::with_seed(seed ^ 0xE45);
    let mut west = RandomAgent::with_seed(seed ^ 0x357);
    let players = [
        Player::Bot(&mut north),
        Player::Bot(&mut east),
        Player::Bot(&mut south),
        Player::Bot(&mut west),
    ];
    Driver::with_seed(
        GameConfig::default(),
        players,
        Verbosity::Silent,
        std::io::empty(),
        std::io::sink(),
        seed,
    )
    .run()
    .expect("a headless match never fails on I/O")
    .winner
}

#[test]
fn heuristic_team_dominates_random_team() {
    let matches: usize = 200;
    let heuristic_wins = (0..matches)
        .filter(|&seed| play_match(seed as u64) == Team::NorthSouth)
        .count();

    // Random play is weak; the heuristic should win the lion's share. We assert
    // a conservative 75% so the test is not flaky, but in practice it wins more.
    assert!(
        heuristic_wins * 100 >= matches * 75,
        "heuristic won only {heuristic_wins}/{matches} matches"
    );
}

#[test]
fn a_full_heuristic_table_completes() {
    // Four heuristic agents must still always make legal moves and finish a
    // match (no panics from the driver's `expect`s on legality).
    let mut bots = [
        HeuristicAgent::new(),
        HeuristicAgent::new(),
        HeuristicAgent::new(),
        HeuristicAgent::new(),
    ];
    let [a, b, c, d] = &mut bots;
    let players = [
        Player::Bot(a),
        Player::Bot(b),
        Player::Bot(c),
        Player::Bot(d),
    ];
    let outcome = Driver::with_seed(
        GameConfig::default(),
        players,
        Verbosity::Silent,
        std::io::empty(),
        std::io::sink(),
        2024,
    )
    .run()
    .expect("a headless match never fails on I/O");
    assert!(outcome.scores.for_team(outcome.winner) >= GameConfig::default().target_score);
}
