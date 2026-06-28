//! Fine-tunes the [`NeuralAgent`](euchre_agents::NeuralAgent)'s policy networks
//! by **self-play reinforcement learning**, starting from the behaviourally
//! cloned weights and pushing the policy past the teacher it was cloned from.
//!
//! ## What it does
//!
//! 1. Loads a warm-start [`NeuralModel`] (by default the shipped, BC-trained
//!    weights) and wraps it in a [`PolicyTrainer`].
//! 2. Repeatedly plays a batch of full matches. Most are *self-play* (the current
//!    policy against itself); the rest are sparring matches against the
//!    [`AdvancedAgent`] teacher. In every match the policy-controlled seats pick
//!    moves by **sampling** their own (masked, temperature-scaled) softmax, which
//!    supplies the exploration RL needs.
//! 3. Turns each hand's signed point swing into a reward for the seats that played
//!    it, whitens the returns per head to form an advantage baseline, and takes a
//!    REINFORCE step on each head ([`PolicyTrainer::step`]).
//! 4. Periodically evaluates the *greedy* (arg-max) policy against the advanced
//!    agent with duplicate dealing, keeping the best checkpoint, and writes that
//!    checkpoint out at the end.
//!
//! The reward is the per-hand point differential (e.g. +2 for a march, −2 for
//! being euchred), credited to every decision that side made during the hand.
//! It is dense — every played hand teaches something — and, because the running
//! match score is a feature, the policy can still learn score-aware play. The
//! checkpoint that ships is selected on the real objective: match wins against the
//! advanced agent.
//!
//! Run it in release mode after the BC stage has produced the warm start; the
//! defaults below are the settings that trained the shipped weights, so the bare
//! command reproduces the shipped agent:
//!
//! ```text
//! cargo run --release -p euchre-agents --example train_rl
//! ```
//!
//! Flags (all optional): `--warm-start PATH`, `--out PATH`, `--iters N`,
//! `--games N` (matches collected per iteration), `--batch-size N`, `--lr F`,
//! `--entropy F` (entropy-bonus coefficient), `--temperature F` (sampling
//! temperature), `--self-play-frac F` (fraction of matches that are self-play vs.
//! sparring against the advanced agent), `--eval-pairs N`, `--eval-every N`,
//! and `--seed N`.

use std::sync::Arc;
use std::time::Instant;

use euchre_agents::neural::features::{
    call_action, call_features, call_legal, card_mask, discard_features, play_features,
    slot_to_card, upcard_action, upcard_features, upcard_legal,
};
use euchre_agents::neural::{Head, NeuralModel, PolicyExample, PolicyTrainer, sample_masked};
use euchre_agents::{AdvancedAgent, NeuralAgent};
use euchre_engine::{Action, Decision, Driver, Game, GameConfig, Player, Verbosity, deal};
use euchre_interface::{Agent, HandResult, Seat};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

/// A fixed deck-seed band for evaluation, disjoint from the training decks, so
/// the win-rate against the advanced agent is measured on the *same* matches
/// every time and checkpoints can be compared apples-to-apples.
const EVAL_SEED_BASE: u64 = 1_000_000_000;

fn main() {
    let opts = Options::parse();
    println!(
        "warm_start={} iters={} games={} batch={} lr={} entropy={} temp={} self_play_frac={} eval_pairs={}",
        opts.warm_start,
        opts.iters,
        opts.games,
        opts.batch_size,
        opts.lr,
        opts.entropy,
        opts.temperature,
        opts.self_play_frac,
        opts.eval_pairs,
    );

    let warm_bytes = std::fs::read(&opts.warm_start).unwrap_or_else(|e| {
        panic!(
            "could not read warm-start model {} ({e}); run the train_neural example first",
            opts.warm_start
        )
    });
    let warm = NeuralModel::load(&warm_bytes).expect("warm-start model is valid");
    let mut trainer = PolicyTrainer::from_model(&warm);

    let mut mode_rng = SmallRng::seed_from_u64(opts.seed ^ 0x4D0D_E5EE);
    let mut deck_rng = SmallRng::seed_from_u64(opts.seed ^ 0xDECC_DECC);
    let mut act_rng = SmallRng::seed_from_u64(opts.seed ^ 0xAC70_AC70);
    let mut shuf_rng = SmallRng::seed_from_u64(opts.seed ^ 0x5417_5417);
    let mut adv = AdvancedAgent::new();

    // Baseline: never ship something weaker than the warm start.
    let mut best_model = trainer.model();
    let mut best = eval_vs_advanced(&Arc::new(best_model.clone()), opts.eval_pairs);
    println!("iter   0  vs advanced {:5.1}%  (warm start)", best * 100.0);

    let t0 = Instant::now();
    for iter in 1..=opts.iters {
        // --- Collect a batch of self-play / sparring matches. ---
        let mut samples: [Vec<RlSample>; 4] = std::array::from_fn(|_| Vec::new());
        for _ in 0..opts.games {
            let is_learner = if mode_rng.random::<f32>() < opts.self_play_frac {
                [true; 4] // self-play: every seat learns
            } else if mode_rng.random::<f32>() < 0.5 {
                [true, false, true, false] // neural North/South vs advanced
            } else {
                [false, true, false, true] // neural East/West vs advanced
            };
            play_and_collect(
                &trainer,
                is_learner,
                &mut adv,
                &mut deck_rng,
                &mut act_rng,
                opts.temperature,
                &mut samples,
            );
        }

        // --- One REINFORCE pass per head over the collected data. ---
        let mut entropy_acc = 0.0;
        let mut steps = 0;
        for head in Head::ALL {
            let recs = &samples[head.index()];
            if recs.len() < 2 {
                continue;
            }
            let (mean, std) = mean_std(recs.iter().map(|r| r.reward));
            let mut order: Vec<usize> = (0..recs.len()).collect();
            order.shuffle(&mut shuf_rng);
            for chunk in order.chunks(opts.batch_size.max(1)) {
                let batch: Vec<PolicyExample<'_>> = chunk
                    .iter()
                    .map(|&i| {
                        let r = &recs[i];
                        PolicyExample {
                            features: &r.features,
                            action: r.action,
                            legal: r.legal,
                            advantage: (r.reward - mean) / (std + 1e-6),
                        }
                    })
                    .collect();
                entropy_acc += trainer.step(head, &batch, opts.lr, opts.entropy, opts.temperature);
                steps += 1;
            }
        }
        let mean_entropy = if steps > 0 {
            entropy_acc / steps as f32
        } else {
            0.0
        };

        // --- Periodic evaluation and checkpoint selection. ---
        if iter % opts.eval_every == 0 || iter == opts.iters {
            let model = trainer.model();
            let wr = eval_vs_advanced(&Arc::new(model.clone()), opts.eval_pairs);
            let improved = wr > best;
            println!(
                "iter {iter:3}  vs advanced {:5.1}%  entropy {:.3}  samples[u/c/d/p]={}/{}/{}/{}  {:.0}s{}",
                wr * 100.0,
                mean_entropy,
                samples[Head::Upcard.index()].len(),
                samples[Head::Call.index()].len(),
                samples[Head::Discard.index()].len(),
                samples[Head::Play.index()].len(),
                t0.elapsed().as_secs_f64(),
                if improved { "  <- new best" } else { "" },
            );
            if improved {
                best = wr;
                best_model = model;
            }
        }
    }

    let bytes = best_model.save();
    std::fs::write(&opts.out, &bytes).expect("write model file");
    println!(
        "wrote {} ({} bytes); best vs advanced {:.1}%",
        opts.out,
        bytes.len(),
        best * 100.0
    );
}

// --- Self-play data collection -----------------------------------------------

/// One recorded learner decision, awaiting its reward at hand's end.
struct Pending {
    head: Head,
    features: Vec<f32>,
    action: usize,
    legal: u32,
    seat: Seat,
}

/// A finished training example: a decision plus the reward it earned.
struct RlSample {
    features: Vec<f32>,
    action: usize,
    legal: u32,
    reward: f32,
}

/// Plays one full match, sampling the policy at every learner seat and the
/// [`AdvancedAgent`] at the rest, and appends a labelled [`RlSample`] (per head)
/// for each learner decision once the hand it belongs to is scored.
#[allow(clippy::too_many_arguments)]
fn play_and_collect(
    trainer: &PolicyTrainer,
    is_learner: [bool; 4],
    adv: &mut AdvancedAgent,
    deck_rng: &mut SmallRng,
    act_rng: &mut SmallRng,
    temperature: f32,
    samples: &mut [Vec<RlSample>; 4],
) {
    let config = GameConfig::default();
    let mut game = Game::new(config, deal(deck_rng));
    let mut pending: Vec<Pending> = Vec::new();

    loop {
        match game.next_action() {
            Action::BidUpcard { seat, .. } => {
                let view = game.view(seat);
                if is_learner[game.player_at(seat)] {
                    let feats = upcard_features(&view);
                    let legal = upcard_legal();
                    let logits = trainer.net(Head::Upcard).forward(&feats);
                    let class = sample_masked(&logits, legal, temperature, act_rng);
                    pending.push(Pending {
                        head: Head::Upcard,
                        features: feats,
                        action: class,
                        legal,
                        seat,
                    });
                    game.apply(Decision::Upcard(upcard_action(class)))
                        .expect("legal up-card");
                } else {
                    let action = adv.bid_upcard(&view);
                    game.apply(Decision::Upcard(action)).expect("legal up-card");
                }
            }
            Action::BidCall {
                seat,
                turned_down,
                may_pass,
            } => {
                let stuck = !may_pass;
                let view = game.view(seat);
                if is_learner[game.player_at(seat)] {
                    let feats = call_features(&view, turned_down, stuck);
                    let legal = call_legal(stuck);
                    let logits = trainer.net(Head::Call).forward(&feats);
                    let class = sample_masked(&logits, legal, temperature, act_rng);
                    pending.push(Pending {
                        head: Head::Call,
                        features: feats,
                        action: class,
                        legal,
                        seat,
                    });
                    game.apply(Decision::Call(call_action(class, turned_down)))
                        .expect("legal call");
                } else {
                    let action = adv.bid_call(&view);
                    game.apply(Decision::Call(action)).expect("legal call");
                }
            }
            Action::Discard { seat, .. } => {
                let view = game.view(seat);
                let trump = view.trump().expect("trump set at discard");
                if is_learner[game.player_at(seat)] {
                    let feats = discard_features(&view);
                    let legal = card_mask(view.hand, trump);
                    let logits = trainer.net(Head::Discard).forward(&feats);
                    let class = sample_masked(&logits, legal, temperature, act_rng);
                    pending.push(Pending {
                        head: Head::Discard,
                        features: feats,
                        action: class,
                        legal,
                        seat,
                    });
                    game.apply(Decision::Discard(slot_to_card(class, trump)))
                        .expect("legal discard");
                } else {
                    let card = adv.discard(&view);
                    game.apply(Decision::Discard(card)).expect("legal discard");
                }
            }
            Action::Play { seat, legal } => {
                let view = game.view(seat);
                let trump = view.trump().expect("trump set at play");
                if is_learner[game.player_at(seat)] {
                    if legal.len() == 1 {
                        // Forced: no decision to learn from.
                        game.apply(Decision::Play(legal[0])).expect("legal play");
                    } else {
                        let feats = play_features(&view);
                        let mask = card_mask(&legal, trump);
                        let logits = trainer.net(Head::Play).forward(&feats);
                        let class = sample_masked(&logits, mask, temperature, act_rng);
                        pending.push(Pending {
                            head: Head::Play,
                            features: feats,
                            action: class,
                            legal: mask,
                            seat,
                        });
                        game.apply(Decision::Play(slot_to_card(class, trump)))
                            .expect("legal play");
                    }
                } else {
                    let card = adv.play_card(&view, &legal);
                    game.apply(Decision::Play(card)).expect("legal play");
                }
            }
            Action::HandComplete { .. } => {
                for p in pending.drain(..) {
                    // The reward is the hand's signed point swing for the seat that
                    // made the decision (its own team's net points).
                    let reward = match game.hand_result(p.seat) {
                        HandResult::Played(score) => score.points_awarded as f32,
                        HandResult::PassedOut => 0.0,
                    };
                    samples[p.head.index()].push(RlSample {
                        features: p.features,
                        action: p.action,
                        legal: p.legal,
                        reward,
                    });
                }
                if game.is_over() {
                    break;
                }
                game.start_next_hand(deal(deck_rng))
                    .expect("ready for next hand");
            }
        }
    }
}

/// Mean and population standard deviation of an iterator of rewards.
fn mean_std(values: impl Iterator<Item = f32> + Clone) -> (f32, f32) {
    let mut n = 0usize;
    let mut sum = 0.0f32;
    for v in values.clone() {
        sum += v;
        n += 1;
    }
    let mean = sum / n as f32;
    let var = values.map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    (mean, var.sqrt())
}

// --- Evaluation --------------------------------------------------------------

/// Match win rate of the greedy neural agent against the advanced agent over
/// `pairs` duplicate pairs on the fixed evaluation deck band.
fn eval_vs_advanced(model: &Arc<NeuralModel>, pairs: u64) -> f64 {
    let mut wins = 0u64;
    for s in 0..pairs {
        let seed = EVAL_SEED_BASE + s;
        if eval_match(model, seed, true) == 0 {
            wins += 1;
        }
        if eval_match(model, seed, false) == 1 {
            wins += 1;
        }
    }
    wins as f64 / (2 * pairs) as f64
}

/// One evaluation match: the (deterministic) neural agent on North/South when
/// `nn_ns`, else East/West, against the advanced agent. Returns the winning team
/// index (0 = North/South, 1 = East/West).
fn eval_match(model: &Arc<NeuralModel>, seed: u64, nn_ns: bool) -> usize {
    let nn = || Box::new(NeuralAgent::from_shared(model.clone())) as Box<dyn Agent>;
    let adv = || Box::new(AdvancedAgent::new()) as Box<dyn Agent>;
    let (mut north, mut east, mut south, mut west) = if nn_ns {
        (nn(), adv(), nn(), adv())
    } else {
        (adv(), nn(), adv(), nn())
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

// --- Minimal argument parsing ------------------------------------------------

struct Options {
    warm_start: String,
    out: String,
    iters: usize,
    games: usize,
    batch_size: usize,
    lr: f32,
    entropy: f32,
    temperature: f32,
    self_play_frac: f32,
    eval_pairs: u64,
    eval_every: usize,
    seed: u64,
}

impl Options {
    fn parse() -> Options {
        let args: Vec<String> = std::env::args().skip(1).collect();
        // Defaults are the settings that trained the shipped weights, so a bare
        // `cargo run` reproduces the shipped agent.
        let mut o = Options {
            warm_start: "euchre-agents/assets/euchre-net.bin".into(),
            out: "euchre-agents/assets/euchre-net.bin".into(),
            iters: 90,
            games: 256,
            batch_size: 512,
            lr: 7e-4,
            entropy: 0.03,
            temperature: 1.5,
            self_play_frac: 0.5,
            eval_pairs: 500,
            eval_every: 5,
            seed: 0,
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
                "--warm-start" => o.warm_start = value(),
                "--out" => o.out = value(),
                "--iters" => o.iters = value().parse().expect("iters is a number"),
                "--games" => o.games = value().parse().expect("games is a number"),
                "--batch-size" => o.batch_size = value().parse().expect("batch-size is a number"),
                "--lr" => o.lr = value().parse().expect("lr is a number"),
                "--entropy" => o.entropy = value().parse().expect("entropy is a number"),
                "--temperature" => {
                    o.temperature = value().parse().expect("temperature is a number")
                }
                "--self-play-frac" => {
                    o.self_play_frac = value().parse().expect("self-play-frac is a number")
                }
                "--eval-pairs" => o.eval_pairs = value().parse().expect("eval-pairs is a number"),
                "--eval-every" => o.eval_every = value().parse().expect("eval-every is a number"),
                "--seed" => o.seed = value().parse().expect("seed is a number"),
                other => panic!("unknown flag: {other}"),
            }
            i += 1;
        }
        o
    }
}
