//! Trains the [`StrongAgent`](euchre_agents::StrongAgent)'s policy networks by
//! **self-play reinforcement learning aimed squarely at beating the
//! [`NeuralAgent`](euchre_agents::NeuralAgent) champion**.
//!
//! The shipped [`NeuralAgent`] is the strongest search-free bot, and the
//! `train_rl` example tuned it by self-play while *selecting checkpoints on
//! win-rate against the [`AdvancedAgent`]*. That objective is one step removed
//! from the goal here, which is simply: **win more matches than the neural
//! champion.** This example optimises that objective directly.
//!
//! ## What it does
//!
//! 1. Loads a warm-start [`NeuralModel`] — by default the shipped champion's own
//!    weights, so training *starts at the champion's level* — and wraps it in a
//!    [`PolicyTrainer`].
//! 2. Repeatedly plays a batch of matches. A fraction are *self-play* (the
//!    current policy against itself); the rest are *sparring* matches in which the
//!    two learner seats face the **frozen neural champion** (the very opponent we
//!    must beat) or, less often, the [`AdvancedAgent`]. Learner seats sample their
//!    own masked, temperature-scaled softmax for exploration.
//! 3. Turns each hand's signed point swing into a reward, whitens returns per head
//!    into advantages, and takes a REINFORCE step per head
//!    ([`PolicyTrainer::step`]) — exactly the machinery the shipped agent was
//!    trained with.
//! 4. Periodically evaluates the *greedy* (arg-max) policy **against the frozen
//!    neural champion** with duplicate dealing on a fixed, training-disjoint deck
//!    band, keeping the checkpoint that beats the champion by the most, and writes
//!    that checkpoint out. Because the warm start *is* the champion, the kept model
//!    is never weaker than it.
//!
//! Run it in release mode. The defaults warm-start from, spar against, and select
//! checkpoints against the shipped champion, so a bare run reproduces one round of
//! training; the shipped weights are a couple of rounds, each warm-started from the
//! previous round's best to compound the edge:
//!
//! ```text
//! cargo run --release -p euchre-agents --example train_strong                                   # round 1, from the champion
//! cargo run --release -p euchre-agents --example train_strong -- --warm-start <round-1-output>  # round 2, iterating
//! ```
//!
//! Flags mirror `train_rl`, plus `--neural-spar-frac F` (share of the non-self-play
//! matches that face the neural champion rather than the advanced agent) and
//! `--champion PATH` (the frozen reference/sparring model; defaults to the shipped
//! champion). The sampling `--temperature` defaults higher than `train_rl`'s: the
//! champion policy is already sharp, so it needs the extra exploration to move.

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

/// A fixed deck-seed band for evaluation, disjoint from the training decks (which
/// start at 0) and from the eval band `train_rl` uses, so checkpoint selection is
/// measured on the same matches every time and is independent of both training and
/// the final cross-check in `euchre-eval`.
const EVAL_SEED_BASE: u64 = 2_000_000_000;

fn main() {
    let opts = Options::parse();
    println!(
        "warm_start={} champion={} iters={} games={} batch={} lr={} entropy={} temp={} self_play_frac={} neural_spar_frac={} eval_pairs={}",
        opts.warm_start,
        opts.champion,
        opts.iters,
        opts.games,
        opts.batch_size,
        opts.lr,
        opts.entropy,
        opts.temperature,
        opts.self_play_frac,
        opts.neural_spar_frac,
        opts.eval_pairs,
    );

    let warm_bytes = std::fs::read(&opts.warm_start).unwrap_or_else(|e| {
        panic!(
            "could not read warm-start model {} ({e}); run train_neural first",
            opts.warm_start
        )
    });
    let warm = NeuralModel::load(&warm_bytes).expect("warm-start model is valid");
    let mut trainer = PolicyTrainer::from_model(&warm);

    let champ_bytes = std::fs::read(&opts.champion)
        .unwrap_or_else(|e| panic!("could not read champion model {} ({e})", opts.champion));
    let champion = Arc::new(NeuralModel::load(&champ_bytes).expect("champion model is valid"));

    let mut mode_rng = SmallRng::seed_from_u64(opts.seed ^ 0x4D0D_E5EE);
    let mut deck_rng = SmallRng::seed_from_u64(opts.seed ^ 0xDECC_DECC);
    let mut act_rng = SmallRng::seed_from_u64(opts.seed ^ 0xAC70_AC70);
    let mut shuf_rng = SmallRng::seed_from_u64(opts.seed ^ 0x5417_5417);

    // Baseline: the warm start (the champion's weights) scored against the frozen
    // champion. Around 50% by construction when warm-started from the champion.
    let mut best_model = trainer.model();
    let mut best = eval_vs_champion(&Arc::new(best_model.clone()), &champion, opts.eval_pairs);
    println!("iter   0  vs champion {:6.2}%  (warm start)", best * 100.0);

    let t0 = Instant::now();
    for iter in 1..=opts.iters {
        // --- Collect a batch of self-play / sparring matches. ---
        let mut samples: [Vec<RlSample>; 4] = std::array::from_fn(|_| Vec::new());
        for _ in 0..opts.games {
            // Decide the match type and the seats the learner occupies.
            let (is_learner, opp): ([bool; 4], Opponent) =
                if mode_rng.random::<f32>() < opts.self_play_frac {
                    ([true; 4], Opponent::SelfPlay) // self-play: every seat learns
                } else {
                    let face_neural = mode_rng.random::<f32>() < opts.neural_spar_frac;
                    let opp = if face_neural {
                        Opponent::Neural
                    } else {
                        Opponent::Advanced
                    };
                    if mode_rng.random::<bool>() {
                        ([true, false, true, false], opp) // learner North/South
                    } else {
                        ([false, true, false, true], opp) // learner East/West
                    }
                };
            let mut opponent = make_opponent(opp, &champion);
            play_and_collect(
                &trainer,
                is_learner,
                &mut opponent,
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
            let wr = eval_vs_champion(&Arc::new(model.clone()), &champion, opts.eval_pairs);
            let improved = wr > best;
            println!(
                "iter {iter:3}  vs champion {:6.2}%  entropy {:.3}  samples[u/c/d/p]={}/{}/{}/{}  {:.0}s{}",
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
        "wrote {} ({} bytes); best vs champion {:.2}%",
        opts.out,
        bytes.len(),
        best * 100.0
    );
}

// --- Self-play data collection -----------------------------------------------

/// Which agent fills the non-learner seats of a collected match.
#[derive(Clone, Copy)]
enum Opponent {
    /// Every seat is a learner; no opponent agent is built.
    SelfPlay,
    /// The frozen neural champion — the opponent we are training to beat.
    Neural,
    /// The advanced heuristic, kept in the mix for varied, robust opposition.
    Advanced,
}

/// Builds the opponent agent for the non-learner seats, or `None` for self-play.
fn make_opponent(opp: Opponent, champion: &Arc<NeuralModel>) -> Option<Box<dyn Agent>> {
    match opp {
        Opponent::SelfPlay => None,
        Opponent::Neural => Some(Box::new(NeuralAgent::from_shared(champion.clone()))),
        Opponent::Advanced => Some(Box::new(AdvancedAgent::new())),
    }
}

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
/// `opponent` at the rest, and appends a labelled [`RlSample`] (per head) for each
/// learner decision once the hand it belongs to is scored. `opponent` is `None`
/// only when every seat is a learner (self-play).
#[allow(clippy::too_many_arguments)]
fn play_and_collect(
    trainer: &PolicyTrainer,
    is_learner: [bool; 4],
    opponent: &mut Option<Box<dyn Agent>>,
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
                    let action = opponent.as_mut().expect("opponent").bid_upcard(&view);
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
                    let action = opponent.as_mut().expect("opponent").bid_call(&view);
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
                    let card = opponent.as_mut().expect("opponent").discard(&view);
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
                    let card = opponent
                        .as_deref_mut()
                        .expect("opponent")
                        .play_card(&view, &legal);
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

/// Match win rate of the greedy learner `model` against the frozen neural
/// `champion` over `pairs` duplicate pairs on the fixed evaluation deck band.
fn eval_vs_champion(model: &Arc<NeuralModel>, champion: &Arc<NeuralModel>, pairs: u64) -> f64 {
    let mut wins = 0u64;
    for s in 0..pairs {
        let seed = EVAL_SEED_BASE + s;
        if eval_match(model, champion, seed, true) == 0 {
            wins += 1;
        }
        if eval_match(model, champion, seed, false) == 1 {
            wins += 1;
        }
    }
    wins as f64 / (2 * pairs) as f64
}

/// One evaluation match: the learner on North/South when `learner_ns`, else
/// East/West, against the frozen champion. Returns the winning team index
/// (0 = North/South, 1 = East/West).
fn eval_match(
    model: &Arc<NeuralModel>,
    champion: &Arc<NeuralModel>,
    seed: u64,
    learner_ns: bool,
) -> usize {
    let learner = || Box::new(NeuralAgent::from_shared(model.clone())) as Box<dyn Agent>;
    let champ = || Box::new(NeuralAgent::from_shared(champion.clone())) as Box<dyn Agent>;
    let (mut north, mut east, mut south, mut west) = if learner_ns {
        (learner(), champ(), learner(), champ())
    } else {
        (champ(), learner(), champ(), learner())
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
    champion: String,
    out: String,
    iters: usize,
    games: usize,
    batch_size: usize,
    lr: f32,
    entropy: f32,
    temperature: f32,
    self_play_frac: f32,
    neural_spar_frac: f32,
    eval_pairs: u64,
    eval_every: usize,
    seed: u64,
}

impl Options {
    fn parse() -> Options {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut o = Options {
            warm_start: "euchre-agents/assets/euchre-net.bin".into(),
            champion: "euchre-agents/assets/euchre-net.bin".into(),
            out: "euchre-agents/assets/euchre-strong.bin".into(),
            iters: 150,
            games: 256,
            batch_size: 512,
            lr: 7e-4,
            entropy: 0.06,
            temperature: 2.0,
            self_play_frac: 0.5,
            neural_spar_frac: 0.7,
            eval_pairs: 800,
            eval_every: 10,
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
                "--champion" => o.champion = value(),
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
                "--neural-spar-frac" => {
                    o.neural_spar_frac = value().parse().expect("neural-spar-frac is a number")
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
