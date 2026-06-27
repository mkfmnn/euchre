//! Integration tests for the [`NeuralAgent`], run as whole matches through the
//! real engine.
//!
//! They pin down the property that justifies the agent: its *search-free* policy
//! crushes random play, carries a clear edge over the plain heuristic, and — after
//! self-play reinforcement fine-tuning — actually *beats* the
//! [`AdvancedAgent`](euchre_agents::AdvancedAgent) it was cloned from. A
//! four-at-a-table match also fuzzes the feature encoder and the legality of
//! every move the net picks against the engine's own checks.
//!
//! The shipped model is the behavioural clone fine-tuned by self-play RL
//! (`examples/train_neural.rs` then `examples/train_rl.rs`); a broken model would
//! collapse toward random play and trip these bars. The headline numbers come from
//! `cargo run --release -p euchre-eval -- neural advanced`.

use euchre_agents::{AdvancedAgent, HeuristicAgent, NeuralAgent, RandomAgent};
use euchre_engine::{Driver, GameConfig, Player, Verbosity};
use euchre_interface::Agent;

/// Runs one match with the North/South team against the East/West team for a
/// given seed, returning the winning team. Each team is built fresh by the
/// supplied closures (North and South share a factory, East and West the other).
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

fn neural() -> Box<dyn Agent> {
    Box::new(NeuralAgent::pretrained())
}

/// Plays one deck both ways — neural North/South, then East/West — against a
/// deterministic `opponent` factory, returning how many of the two matches the
/// neural agent won (0, 1, or 2). Holding the deck fixed across the swap cancels
/// deal luck, the same trick the eval harness uses.
fn neural_wins_in_pair(seed: u64, opponent: impl Fn() -> Box<dyn Agent> + Copy) -> u32 {
    let as_ns = play_match(seed, neural, opponent) == 0;
    let as_ew = play_match(seed, opponent, neural) == 1;
    u32::from(as_ns) + u32::from(as_ew)
}

#[test]
fn neural_team_dominates_random_team() {
    let matches: usize = 200;
    let wins = (0..matches)
        .filter(|&seed| {
            play_match(seed as u64, neural, || {
                Box::new(RandomAgent::with_seed(seed as u64 ^ 0xA1CE))
            }) == 0
        })
        .count();

    // A learned policy that works should win almost everything against noise.
    assert!(
        wins * 100 >= matches * 85,
        "neural won only {wins}/{matches} against random"
    );
}

#[test]
fn neural_team_beats_the_heuristic_team() {
    let pairs: usize = 150;
    let wins: u32 = (0..pairs)
        .map(|p| neural_wins_in_pair(p as u64, || Box::new(HeuristicAgent::new())))
        .sum();
    let total = (pairs * 2) as u32;

    eprintln!("[neural vs heuristic, paired] {wins}/{total}");
    // Distilled from the advanced agent, the net inherits its clear edge over the
    // plain heuristic (measured ~60%); a conservative 53% bar leaves headroom
    // while a regression toward random play would still trip it.
    assert!(
        wins * 100 >= total * 53,
        "neural won only {wins}/{total} against the heuristic"
    );
}

#[test]
fn neural_beats_its_teacher() {
    let pairs: usize = 150;
    let wins: u32 = (0..pairs)
        .map(|p| neural_wins_in_pair(p as u64, || Box::new(AdvancedAgent::new())))
        .sum();
    let total = (pairs * 2) as u32;

    eprintln!("[neural vs advanced, paired] {wins}/{total}");
    // The behavioural clone starts level with its teacher; self-play RL fine-tuning
    // pushes it clearly ahead (measured ~63% here, ~62% on the eval harness). The
    // 56% bar leaves headroom while still asserting the agent *beats* the advanced
    // agent — the whole point of the RL stage. A model that regressed to the bare
    // clone (~50%) or toward random play (~10–15%) would trip it.
    assert!(
        wins * 100 >= total * 56,
        "neural won only {wins}/{total} against the advanced agent it was cloned from"
    );
}

#[test]
fn a_full_neural_table_completes() {
    // Four neural agents must always make legal moves and finish a match without
    // tripping any of the engine's legality assertions.
    let mut bots = [neural(), neural(), neural(), neural()];
    let [a, b, c, d] = &mut bots;
    let players = [
        Player::Bot(a.as_mut()),
        Player::Bot(b.as_mut()),
        Player::Bot(c.as_mut()),
        Player::Bot(d.as_mut()),
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
