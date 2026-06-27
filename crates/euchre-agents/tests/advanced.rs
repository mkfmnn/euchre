//! Integration tests for the [`AdvancedAgent`], run as whole matches through the
//! real engine.
//!
//! They pin down the two properties that justify the agent's existence: it
//! crushes random play, and it is a clear, repeatable improvement on the plain
//! [`HeuristicAgent`]. A regression that broke its bidding or play would show up
//! here as a collapsed win rate.

use euchre_agents::{AdvancedAgent, HeuristicAgent, RandomAgent};
use euchre_engine::{Driver, GameConfig, Player, Verbosity};
use euchre_interface::Agent;

/// Runs one match with the North/South team against the East/West team for a
/// given seed, returning the winning team. Each team is built fresh by the
/// supplied closures.
fn play_match(
    seed: u64,
    mut ns: impl FnMut() -> Box<dyn Agent>,
    mut ew: impl FnMut() -> Box<dyn Agent>,
) -> usize {
    let mut north = ns();
    let mut south = ns();
    let mut east = ew();
    let mut west = ew();
    let players = [
        Player::Bot(north.as_mut()),
        Player::Bot(east.as_mut()),
        Player::Bot(south.as_mut()),
        Player::Bot(west.as_mut()),
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
fn advanced_team_dominates_random_team() {
    let matches: usize = 200;
    let wins = (0..matches)
        .filter(|&seed| {
            play_match(
                seed as u64,
                || Box::new(AdvancedAgent::new()),
                || Box::new(RandomAgent::with_seed(seed as u64 ^ 0xA1CE)),
            ) == 0
        })
        .count();

    // The advanced agent should win nearly everything against noise; assert a
    // conservative bar so the test is not flaky.
    assert!(
        wins * 100 >= matches * 85,
        "advanced won only {wins}/{matches} against random"
    );
}

#[test]
fn advanced_team_beats_the_heuristic_team() {
    let matches: usize = 400;
    let wins = (0..matches)
        .filter(|&seed| {
            play_match(
                seed as u64,
                || Box::new(AdvancedAgent::new()),
                || Box::new(HeuristicAgent::new()),
            ) == 0
        })
        .count();

    // Measured around 60%+; assert a conservative majority so normal variance
    // cannot trip the test while a real regression still would.
    assert!(
        wins * 100 >= matches * 54,
        "advanced won only {wins}/{matches} against the heuristic"
    );
}

#[test]
fn a_full_advanced_table_completes() {
    // Four advanced agents must always make legal moves and finish a match
    // without tripping any of the engine's legality assertions.
    let mut bots = [
        AdvancedAgent::new(),
        AdvancedAgent::new(),
        AdvancedAgent::new(),
        AdvancedAgent::new(),
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
