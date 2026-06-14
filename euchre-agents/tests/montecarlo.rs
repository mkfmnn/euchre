//! Integration tests for the [`MonteCarloAgent`], run as whole matches through
//! the real engine.
//!
//! They pin down the property that justifies the agent's existence: its
//! search-based play beats the strong [`AdvancedAgent`] it delegates its bidding
//! to, and it crushes random play. A four-at-a-table match also fuzzes the
//! solver and determinizer against the engine's legality checks.
//!
//! The search makes each match far slower than the heuristic agents', so these
//! tests use a deliberately small determinization count and modest match counts;
//! the headline strength number comes from `cargo run --release -p euchre-eval --
//! montecarlo advanced --sprt`.

use euchre_agents::{AdvancedAgent, MonteCarloAgent, RandomAgent};
use euchre_engine::{Driver, GameConfig, Player, Team, Verbosity};
use euchre_interface::Agent;

/// Determinizations per play in the tests — small, to keep the suite quick while
/// still clearly out-playing the heuristics.
const TEST_N: usize = 8;

/// Duplicate-dealing pairs played against the advanced agent. Each pair is two
/// matches on the same deck with the sides swapped, so deal luck cancels and a
/// modest number of (slow) matches suffices to show the edge. Kept small because
/// every match runs the search; the deterministic seeds make the outcome fixed.
const ADVANCED_PAIRS: usize = 20;

/// Runs one match with the North/South team against the East/West team for a
/// given seed, returning the winning team. Each seat is built fresh by the
/// supplied closures.
fn play_match(
    seed: u64,
    mut ns: impl FnMut() -> Box<dyn Agent>,
    mut ew: impl FnMut() -> Box<dyn Agent>,
) -> Team {
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

/// Builds a Monte-Carlo agent seeded distinctly per construction, at the test
/// search width.
fn montecarlo(tag: u64) -> Box<dyn Agent> {
    Box::new(MonteCarloAgent::with_seed(tag).with_determinizations(TEST_N))
}

/// A fresh-seeded factory closure for the Monte-Carlo agent, giving each of the
/// two seats it fills a distinct sampling stream.
fn montecarlo_side(base: u64) -> impl FnMut() -> Box<dyn Agent> {
    let mut tag = base;
    move || {
        tag = tag.wrapping_add(1);
        montecarlo(tag)
    }
}

/// Plays one deck both ways — Monte-Carlo North/South, then East/West — and
/// returns how many of the two matches it won (0, 1, or 2). Holding the deck
/// fixed across the swap cancels deal luck, the same trick the eval harness uses.
fn montecarlo_wins_in_pair(seed: u64) -> u32 {
    let base = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let as_ns = play_match(seed, montecarlo_side(base), || {
        Box::new(AdvancedAgent::new())
    }) == Team::NorthSouth;
    let as_ew = play_match(
        seed,
        || Box::new(AdvancedAgent::new()),
        montecarlo_side(base ^ 0xFFFF_FFFF),
    ) == Team::EastWest;
    u32::from(as_ns) + u32::from(as_ew)
}

#[test]
fn montecarlo_team_dominates_random_team() {
    let matches: usize = 16;
    let mut counter = 0u64;
    let wins = (0..matches)
        .filter(|&seed| {
            play_match(
                seed as u64,
                || {
                    counter += 1;
                    montecarlo(seed as u64 * 1000 + counter)
                },
                || Box::new(RandomAgent::with_seed(seed as u64 ^ 0xA1CE)),
            ) == Team::NorthSouth
        })
        .count();

    eprintln!("[montecarlo vs random] {wins}/{matches}");
    assert!(
        wins * 100 >= matches * 80,
        "montecarlo won only {wins}/{matches} against random"
    );
}

#[test]
fn montecarlo_team_beats_the_advanced_team() {
    let wins: u32 = (0..ADVANCED_PAIRS)
        .map(|p| montecarlo_wins_in_pair(p as u64))
        .sum();
    let total = (ADVANCED_PAIRS * 2) as u32;

    eprintln!("[montecarlo vs advanced, paired] {wins}/{total}");
    // The measured edge at this search width is a clear majority (~59–62%); with
    // deal luck cancelled by the paired design, a conservative 53% bar leaves
    // headroom while a real regression (a broken solver, which the anchor would
    // pull back to roughly the advanced agent's own ~50%) would still trip it.
    assert!(
        wins * 100 >= total * 53,
        "montecarlo won only {wins}/{total} against the advanced agent"
    );
}

#[test]
fn a_full_montecarlo_table_completes() {
    // Four searching agents must always make legal moves and finish a match
    // without tripping any of the engine's legality assertions.
    let mut bots = [
        MonteCarloAgent::with_seed(1).with_determinizations(6),
        MonteCarloAgent::with_seed(2).with_determinizations(6),
        MonteCarloAgent::with_seed(3).with_determinizations(6),
        MonteCarloAgent::with_seed(4).with_determinizations(6),
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
