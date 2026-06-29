//! Integration tests for the [`StrongAgent`], run as whole matches through the
//! real engine.
//!
//! They pin down the property that justifies the agent: it is *search-free* like
//! the [`NeuralAgent`](euchre_agents::NeuralAgent) champion, yet it **wins their
//! head-to-head** — the whole reason it exists. A four-at-a-table match also
//! fuzzes the feature encoder and the legality of every move the net picks against
//! the engine's own checks.
//!
//! The shipped weights are a wider clone of the champion fine-tuned by self-play
//! RL that spars against, and is checkpoint-selected to beat, the champion
//! (`examples/train_strong.rs`). A broken model would collapse toward the bare
//! clone (a coin flip against the champion) or toward random play and trip these
//! bars. The headline number comes from `cargo run --release -p euchre-eval --
//! strong neural`.

use euchre_agents::{NeuralAgent, RandomAgent, StrongAgent};
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

fn strong() -> Box<dyn Agent> {
    Box::new(StrongAgent::pretrained())
}

/// Plays one deck both ways — strong North/South, then East/West — against a
/// deterministic `opponent` factory, returning how many of the two matches the
/// strong agent won (0, 1, or 2). Holding the deck fixed across the swap cancels
/// deal luck, the same trick the eval harness uses.
fn strong_wins_in_pair(seed: u64, opponent: impl Fn() -> Box<dyn Agent> + Copy) -> u32 {
    let as_ns = play_match(seed, strong, opponent) == 0;
    let as_ew = play_match(seed, opponent, strong) == 1;
    u32::from(as_ns) + u32::from(as_ew)
}

#[test]
fn strong_team_dominates_random_team() {
    let matches: usize = 200;
    let wins = (0..matches)
        .filter(|&seed| {
            play_match(seed as u64, strong, || {
                Box::new(RandomAgent::with_seed(seed as u64 ^ 0xA1CE))
            }) == 0
        })
        .count();

    // A strong learned policy should win almost everything against noise.
    assert!(
        wins * 100 >= matches * 85,
        "strong won only {wins}/{matches} against random"
    );
}

#[test]
fn strong_beats_the_neural_champion() {
    let pairs: usize = 150;
    let mut wins: u32 = 0;
    let mut strong_better = 0u32; // duplicate pairs the strong agent swept 2–0
    let mut neural_better = 0u32; // pairs the champion swept
    for p in 0..pairs {
        let w = strong_wins_in_pair(p as u64, || Box::new(NeuralAgent::pretrained()));
        wins += w;
        match w {
            2 => strong_better += 1,
            0 => neural_better += 1,
            _ => {}
        }
    }
    let total = (pairs * 2) as u32;

    eprintln!(
        "[strong vs neural, paired] {wins}/{total}  decisive pairs: strong {strong_better}, neural {neural_better}"
    );
    // The agent is the champion's own weights pushed *past* it by self-play RL.
    // Measured ~56% on the eval harness over 3000 games (and ~55% on a 250-pair
    // prefix); the 51% bar leaves ample headroom while still asserting the agent
    // beats the champion — the whole point. A model that regressed to the warm
    // start (~50%) or toward random play would trip it.
    assert!(
        wins * 100 >= total * 51,
        "strong won only {wins}/{total} against the neural champion"
    );
    // The variance-reduced verdict: the strong agent must take more decks outright
    // than the champion does.
    assert!(
        strong_better > neural_better,
        "strong swept {strong_better} decks vs the champion's {neural_better}"
    );
}

#[test]
fn a_full_strong_table_completes() {
    // Four strong agents must always make legal moves and finish a match without
    // tripping any of the engine's legality assertions.
    let mut bots = [strong(), strong(), strong(), strong()];
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
