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
//! The networks are trained by **behavioural cloning** (policy distillation): a
//! strong existing agent — the [`AdvancedAgent`](crate::AdvancedAgent), or the
//! search-based [`MonteCarloAgent`](crate::MonteCarloAgent) — plays out a large
//! number of games, and at every decision we record the public
//! [`GameView`](euchre_interface::GameView) (encoded by [`features`]) together
//! with the action the teacher chose. Each head is then fit as a classifier that
//! reproduces the teacher's choices. Distilling a search-based teacher this way
//! yields a network that approaches the teacher's strength while playing at the
//! speed of a single matrix multiply — exactly what is wanted for a search-free
//! evaluation, and a clean base to later pair *with* search.
//!
//! ## Why these choices
//!
//! * **Behavioural cloning over reinforcement learning.** We already have strong
//!   teachers, so cloning gives a competitive policy reliably and cheaply, with a
//!   stable supervised objective and no self-play instability. The pieces are laid
//!   out so a policy-gradient fine-tune could be added later.
//! * **A hand-written MLP over a deep-learning framework.** The networks are
//!   tiny, so an explicit, gradient-checked implementation (see [`net`]) keeps the
//!   the inference path lightweight and deterministic.
//! * **Trump-relative feature encoding** (see [`features`]) so the net learns
//!   card values once rather than once per suit — the dominant quality lever.
//!
//! ## Layout
//!
//! * [`net`] — the MLP, Adam, masked-softmax loss, serialization.
//! * [`features`] — `GameView` → feature vectors, and action ↔ class mappings.
//! * [`mod@train`] — the model bundle, save/load, and the supervised training loop.
//! * [`agent`] — the [`NeuralAgent`] inference path.
//!
//! Collecting the training samples (driving real games with a teacher) lives in
//! the `train_neural` example, which can depend on the engine; this module
//! cannot, keeping the agent layer's dependencies clean.

pub mod agent;
pub mod features;
pub mod net;
pub mod train;

pub use agent::NeuralAgent;
pub use features::Head;
pub use train::{NeuralModel, Sample, TrainConfig, train};
