//! Integration tests that run whole matches through the real engine, pitting
//! the [`HeuristicAgent`] against the [`RandomAgent`].
//!
//! These double as a sanity check on the bidding thresholds: a heuristic that
//! tuned itself into never bidding (or always bidding) would stop beating noise,
//! and these tests would catch it.

use euchre_agents::{HeuristicAgent, OpenAiAdvancedAgent, RandomAgent};
use euchre_engine::{Driver, GameConfig, Player, Verbosity};

/// Runs one match with the heuristic team (North/South) against the random team
/// (East/West) for a given seed, returning the winning team.
fn play_match(seed: u64) -> usize {
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
        .filter(|&seed| play_match(seed as u64) == 0)
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
    assert!(outcome.scores[outcome.winner] >= GameConfig::default().target_score);
}

fn play_openai_vs_random(seed: u64) -> usize {
    let mut north = OpenAiAdvancedAgent::with_seed(seed ^ 0xA17A);
    let mut south = OpenAiAdvancedAgent::with_seed(seed ^ 0xC0DE);
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

fn play_openai_vs_heuristic_pair(seed: u64) -> (bool, bool) {
    let mut a_north = OpenAiAdvancedAgent::with_seed(seed ^ 0xA01);
    let mut a_south = OpenAiAdvancedAgent::with_seed(seed ^ 0xA02);
    let mut b_east = HeuristicAgent::new();
    let mut b_west = HeuristicAgent::new();
    let players = [
        Player::Bot(&mut a_north),
        Player::Bot(&mut b_east),
        Player::Bot(&mut a_south),
        Player::Bot(&mut b_west),
    ];
    let first = Driver::with_seed(
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
        == 0;

    let mut b_north = HeuristicAgent::new();
    let mut b_south = HeuristicAgent::new();
    let mut a_east = OpenAiAdvancedAgent::with_seed(seed ^ 0xA03);
    let mut a_west = OpenAiAdvancedAgent::with_seed(seed ^ 0xA04);
    let players = [
        Player::Bot(&mut b_north),
        Player::Bot(&mut a_east),
        Player::Bot(&mut b_south),
        Player::Bot(&mut a_west),
    ];
    let second = Driver::with_seed(
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
        == 1;

    (first, second)
}

#[test]
fn openai_advanced_team_dominates_random_team() {
    let matches: usize = 200;
    let wins = (0..matches)
        .filter(|&seed| play_openai_vs_random(seed as u64) == 0)
        .count();

    assert!(
        wins * 100 >= matches * 85,
        "openai advanced won only {wins}/{matches} matches"
    );
}

#[test]
fn a_full_openai_advanced_table_completes() {
    let mut bots = [
        OpenAiAdvancedAgent::with_seed(1),
        OpenAiAdvancedAgent::with_seed(2),
        OpenAiAdvancedAgent::with_seed(3),
        OpenAiAdvancedAgent::with_seed(4),
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
        2026,
    )
    .run()
    .expect("a headless match never fails on I/O");
    assert!(outcome.scores[outcome.winner] >= GameConfig::default().target_score);
}

#[test]
fn openai_advanced_has_positive_record_against_heuristic() {
    let pairs: usize = 80;
    let wins = (0..pairs)
        .map(|seed| play_openai_vs_heuristic_pair(seed as u64))
        .map(|(first, second)| usize::from(first) + usize::from(second))
        .sum::<usize>();

    assert!(
        wins > pairs,
        "openai advanced won only {wins}/{} duplicate games",
        pairs * 2
    );
}
