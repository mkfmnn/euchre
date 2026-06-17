//! Trains the [`NeuralAgent`](euchre_agents::NeuralAgent)'s policy networks by
//! behavioural cloning and writes the resulting model to disk.
//!
//! A *teacher* agent plays out many full matches; at every decision we encode the
//! public view and record the action the teacher chose. Those `(features, label)`
//! pairs train the four policy heads. To broaden the states the play head sees
//! beyond the teacher's own (very good) lines, a fraction of *plays* are taken at
//! random while still being labelled with the teacher's preferred card — a cheap
//! DAgger-style exploration that makes the clone more robust off-trajectory.
//!
//! Run it in release mode (the search-based teacher is slow otherwise):
//!
//! ```text
//! cargo run --release -p euchre-agents --example train_neural -- \
//!     --teacher advanced --matches 2000 --out euchre-agents/assets/euchre-net.bin
//! ```
//!
//! Flags (all optional): `--teacher {random|heuristic|advanced|montecarlo}`,
//! `--matches N`, `--epsilon F` (play exploration rate), `--hidden N`,
//! `--epochs N`, `--seed N`, `--out PATH`, and `--eval` to play the trained agent
//! against the baselines and print win rates.

use std::sync::Arc;
use std::time::Instant;

use euchre_agents::neural::features::{
    call_class, call_features, call_legal, card_mask, card_slot, discard_features, play_features,
    upcard_class, upcard_features, upcard_legal,
};
use euchre_agents::neural::{Head, NeuralModel, Sample, TrainConfig, train};
use euchre_agents::{AdvancedAgent, HeuristicAgent, MonteCarloAgent, NeuralAgent, RandomAgent};
use euchre_engine::{Action, Decision, Driver, Game, GameConfig, Player, Team, Verbosity, deal};
use euchre_interface::{Agent, Seat};
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};

fn main() {
    let opts = Options::parse();
    println!(
        "teacher={} matches={} epsilon={} hidden={} epochs={} seed={}",
        opts.teacher, opts.matches, opts.epsilon, opts.hidden, opts.epochs, opts.seed
    );

    let t0 = Instant::now();
    let samples = collect(&opts);
    report_samples(&samples);
    println!("collected in {:.1}s", t0.elapsed().as_secs_f64());

    let config = TrainConfig {
        hidden: opts.hidden,
        epochs: opts.epochs,
        seed: opts.seed,
        ..TrainConfig::default()
    };
    let t1 = Instant::now();
    let (model, reports) = train(&samples, config);
    println!("trained in {:.1}s", t1.elapsed().as_secs_f64());
    for r in &reports {
        println!(
            "  {:<8} samples={:>7} train_loss={:.4} val_acc={:.3}",
            head_name(r.head),
            r.train_samples,
            r.train_loss,
            r.val_accuracy
        );
    }

    let bytes = model.save();
    std::fs::write(&opts.out, &bytes).expect("write model file");
    println!("wrote {} ({} bytes)", opts.out, bytes.len());

    if opts.eval {
        evaluate(Arc::new(model));
    }
}

// --- Data collection ---------------------------------------------------------

/// Plays `opts.matches` teacher-vs-teacher matches, recording a [`Sample`] at
/// every decision of every seat.
fn collect(opts: &Options) -> Vec<Sample> {
    let config = GameConfig::default();
    let mut deck_rng = SmallRng::seed_from_u64(opts.seed);
    let mut explore_rng = SmallRng::seed_from_u64(opts.seed ^ 0x5EED_E2E2);
    let mut samples = Vec::new();

    for m in 0..opts.matches {
        let mut agents = make_teachers(&opts.teacher, opts.seed.wrapping_add(m as u64 * 4));
        let mut game = Game::new(config, deal(&mut deck_rng));
        loop {
            match game.next_action() {
                Action::BidUpcard { seat, up_card } => {
                    let view = game.view(seat);
                    let feats = upcard_features(&view, up_card);
                    let action = agents[idx(seat)].bid_upcard(&view, up_card);
                    samples.push(Sample {
                        head: Head::Upcard,
                        features: feats,
                        target: upcard_class(action),
                        legal: upcard_legal(),
                    });
                    game.apply(Decision::Upcard(action)).expect("legal up-card");
                }
                Action::BidCall {
                    seat,
                    turned_down,
                    may_pass,
                } => {
                    let stuck = !may_pass;
                    let view = game.view(seat);
                    let feats = call_features(&view, turned_down, stuck);
                    let action = agents[idx(seat)].bid_call(&view, turned_down);
                    samples.push(Sample {
                        head: Head::Call,
                        features: feats,
                        target: call_class(action, turned_down),
                        legal: call_legal(stuck),
                    });
                    game.apply(Decision::Call(action)).expect("legal call");
                }
                Action::Discard { seat, .. } => {
                    let view = game.view(seat);
                    let trump = view.trump().expect("trump set at discard");
                    let feats = discard_features(&view);
                    let card = agents[idx(seat)].discard(&view);
                    samples.push(Sample {
                        head: Head::Discard,
                        features: feats,
                        target: card_slot(card, trump),
                        legal: card_mask(view.hand, trump),
                    });
                    game.apply(Decision::Discard(card)).expect("legal discard");
                }
                Action::Play { seat, legal } => {
                    let view = game.view(seat);
                    let trump = view.trump().expect("trump set at play");
                    let feats = play_features(&view);
                    let card = agents[idx(seat)].play_card(&view, &legal);
                    samples.push(Sample {
                        head: Head::Play,
                        features: feats,
                        target: card_slot(card, trump),
                        legal: card_mask(&legal, trump),
                    });
                    // Act with a little exploration to widen play-state coverage,
                    // but always learn the teacher's preferred card.
                    let played = if opts.epsilon > 0.0 && explore_rng.random::<f32>() < opts.epsilon
                    {
                        *legal.choose(&mut explore_rng).expect("legal is non-empty")
                    } else {
                        card
                    };
                    game.apply(Decision::Play(played)).expect("legal play");
                }
                Action::HandComplete { result, .. } => {
                    for seat in Seat::ALL {
                        let view = game.view(seat);
                        agents[idx(seat)].observe_hand_end(&view, &result);
                    }
                    if game.is_over() {
                        break;
                    }
                    game.start_next_hand(deal(&mut deck_rng))
                        .expect("ready for next hand");
                }
            }
        }
        if (m + 1) % 250 == 0 {
            println!("  {} / {} matches", m + 1, opts.matches);
        }
    }
    samples
}

/// Builds four fresh teacher agents of the named kind, seeded for any that are
/// stochastic.
fn make_teachers(name: &str, base_seed: u64) -> [Box<dyn Agent>; 4] {
    std::array::from_fn(|i| make_agent(name, base_seed.wrapping_add(i as u64)))
}

/// Builds one agent by name. `montecarlo` is the strong search teacher; the
/// others are useful for quick pipeline checks.
fn make_agent(name: &str, seed: u64) -> Box<dyn Agent> {
    match name {
        "random" => Box::new(RandomAgent::with_seed(seed)),
        "heuristic" => Box::new(HeuristicAgent::new()),
        "advanced" => Box::new(AdvancedAgent::new()),
        "montecarlo" => Box::new(MonteCarloAgent::with_seed(seed)),
        other => panic!("unknown teacher: {other}"),
    }
}

fn report_samples(samples: &[Sample]) {
    let mut counts = [0usize; 4];
    for s in samples {
        counts[s.head.index()] += 1;
    }
    println!(
        "samples: total={} upcard={} call={} discard={} play={}",
        samples.len(),
        counts[Head::Upcard.index()],
        counts[Head::Call.index()],
        counts[Head::Discard.index()],
        counts[Head::Play.index()],
    );
}

// --- Evaluation --------------------------------------------------------------

/// Plays the trained agent against each baseline with duplicate dealing and
/// prints match win rates (a quick read; the real number comes from `euchre-eval`).
fn evaluate(model: Arc<NeuralModel>) {
    const PAIRS: u64 = 300;
    for opp in ["random", "heuristic", "advanced"] {
        let wins = head_to_head(&model, opp, PAIRS);
        println!(
            "neural vs {:<10} {:.1}% over {} games",
            opp,
            100.0 * wins as f64 / (2 * PAIRS) as f64,
            2 * PAIRS
        );
    }
}

/// Match wins for the neural agent over `pairs` duplicate pairs against `opp`.
fn head_to_head(model: &Arc<NeuralModel>, opp: &str, pairs: u64) -> u64 {
    let mut wins = 0;
    for seed in 0..pairs {
        if play(model, opp, seed, true) == Team::NorthSouth {
            wins += 1;
        }
        if play(model, opp, seed, false) == Team::EastWest {
            wins += 1;
        }
    }
    wins
}

/// One match: the neural agent on North/South when `nn_ns`, else East/West.
fn play(model: &Arc<NeuralModel>, opp: &str, seed: u64, nn_ns: bool) -> Team {
    let nn = || Box::new(NeuralAgent::from_shared(model.clone())) as Box<dyn Agent>;
    let other = || make_agent(opp, seed ^ 0xA1CE);
    let (mut north, mut east, mut south, mut west) = if nn_ns {
        (nn(), other(), nn(), other())
    } else {
        (other(), nn(), other(), nn())
    };
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
    .expect("headless match never fails on I/O")
    .winner
}

fn idx(seat: Seat) -> usize {
    Seat::ALL
        .iter()
        .position(|&s| s == seat)
        .expect("seat in ALL")
}

fn head_name(head: Head) -> &'static str {
    match head {
        Head::Upcard => "upcard",
        Head::Call => "call",
        Head::Discard => "discard",
        Head::Play => "play",
    }
}

// --- Minimal argument parsing ------------------------------------------------

struct Options {
    teacher: String,
    matches: usize,
    epsilon: f32,
    hidden: usize,
    epochs: usize,
    seed: u64,
    out: String,
    eval: bool,
}

impl Options {
    fn parse() -> Options {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut o = Options {
            teacher: "advanced".into(),
            matches: 2000,
            epsilon: 0.1,
            hidden: 128,
            epochs: 14,
            seed: 0,
            out: "euchre-agents/assets/euchre-net.bin".into(),
            eval: false,
        };
        let mut i = 0;
        while i < args.len() {
            let key = args[i].as_str();
            let mut value = || {
                i += 1;
                args.get(i)
                    .cloned()
                    .unwrap_or_else(|| panic!("missing value for {key}"))
            };
            match key {
                "--teacher" => o.teacher = value(),
                "--matches" => o.matches = value().parse().expect("matches is a number"),
                "--epsilon" => o.epsilon = value().parse().expect("epsilon is a number"),
                "--hidden" => o.hidden = value().parse().expect("hidden is a number"),
                "--epochs" => o.epochs = value().parse().expect("epochs is a number"),
                "--seed" => o.seed = value().parse().expect("seed is a number"),
                "--out" => o.out = value(),
                "--eval" => o.eval = true,
                other => panic!("unknown flag: {other}"),
            }
            i += 1;
        }
        o
    }
}
