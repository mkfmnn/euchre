//! A neural-network agent and the machinery to train it.
//!
//! ## What it is
//!
//! [`NeuralAgent`] plays Euchre with four small policy networks — one per
//! decision point (order-up, call, discard, play) — and nothing else: each move
//! is a single forward pass followed by a masked arg-max over the legal options.
//! There is **no tree search** at play time, by design.
//!
//! ## How it is trained
//!
//! Training has two stages, and the shipped weights are the product of both.
//!
//! 1. **Behavioural cloning** (policy distillation) gives a competent warm start.
//!    A strong existing agent — the [`AdvancedAgent`](crate::AdvancedAgent), or the
//!    search-based [`MonteCarloAgent`](crate::MonteCarloAgent) — plays out a large
//!    number of games, and at every decision we record the public
//!    [`GameView`](euchre_interface::GameView) (encoded by [`features`]) together
//!    with the action the teacher chose. Each head is fit as a classifier that
//!    reproduces the teacher's choices (the `train_neural` example, [`train`]). A
//!    clean clone lands *just below* its teacher.
//! 2. **Self-play reinforcement learning** then pushes the policy *past* its
//!    teacher. Starting from the cloned weights, the agent plays itself (and the
//!    teacher, as a sparring partner) while sampling from its own stochastic
//!    policy; each hand's signed point swing is the reward, and the four heads are
//!    nudged toward the actions that paid off by the REINFORCE policy gradient
//!    ([`PolicyTrainer`], the `train_rl` example). This is where the agent stops
//!    imitating the advanced agent and starts beating it — while still deciding
//!    every move with a single forward pass, no search.
//!
//! ## Why these choices
//!
//! * **Clone first, then reinforce.** Behavioural cloning reaches a strong policy
//!   reliably and cheaply with a stable supervised objective; warm-starting RL
//!   from it sidesteps the cold-start flailing that makes from-scratch self-play
//!   slow and unstable, so the policy gradient spends its samples *improving* good
//!   play rather than discovering the rules.
//! * **REINFORCE with a whitened-return baseline and an entropy bonus.** A plain
//!   Monte-Carlo policy gradient needs no extra value network (matching the
//!   project's lean, verifiable bent); per-batch advantage whitening supplies the
//!   variance-reducing baseline, and the entropy bonus keeps the policy from
//!   collapsing before it has explored. The per-hand point swing is a dense reward
//!   (every hand teaches something), and the running score is a feature, so the
//!   policy can still learn to be score-aware.
//! * **A hand-written MLP over a deep-learning framework.** The networks are
//!   tiny, so an explicit, gradient-checked implementation (see [`net`]) keeps the
//!   numerics verifiable and the inference path dependency-free and deterministic
//!   — matching the rest of this workspace, which hand-rolls its stats too. The
//!   policy gradient reuses the same `softmax − onehot` gradient the supervised
//!   loss is built on, so the RL path inherits that gradient check.
//! * **Trump-relative feature encoding** (see [`features`]) so the net learns
//!   card values once rather than once per suit — the dominant quality lever.
//!
//! ## Layout
//!
//! * [`net`] — the MLP, Adam, masked-softmax loss, the REINFORCE step, serialization.
//! * [`features`] — `GameView` → feature vectors, and action ↔ class mappings.
//! * [`mod@train`] — the model bundle, save/load, the supervised loop, and the
//!   [`PolicyTrainer`] used for RL fine-tuning.
//! * [`agent`] — the [`NeuralAgent`] inference path.
//!
//! Collecting the training data (driving real games with a teacher, or in
//! self-play) lives in the `train_neural` and `train_rl` examples, which can
//! depend on the engine; this module cannot, keeping the agent layer's
//! dependencies clean.

pub mod agent;
pub mod features;
pub mod net;
pub mod train;

pub use agent::NeuralAgent;
pub use features::Head;
pub use net::{PolicyExample, sample_masked};
pub use train::{NeuralModel, PolicyTrainer, Sample, TrainConfig, train};
