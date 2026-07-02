//! [`StrongAgent`]: a search-free policy agent trained to beat the
//! [`NeuralAgent`](crate::NeuralAgent) champion.
//!
//! ## What it is
//!
//! `StrongAgent` shares the [`NeuralAgent`]'s architecture *exactly* — the same
//! four small policy networks (order-up, call, discard, play), one masked forward
//! pass per decision, no search — and reuses its inference path verbatim, so it
//! plays at the same speed. Only the *weights* differ: they are tuned specifically
//! to **win more matches than the neural champion**, and they do.
//!
//! ## How it is trained
//!
//! The [`NeuralAgent`] is itself the product of behavioural cloning followed by
//! self-play RL, but its self-play checkpoints were selected on win-rate against
//! the [`AdvancedAgent`](crate::AdvancedAgent) — a *proxy* for the real goal.
//! `StrongAgent` optimises the objective directly:
//!
//! 1. **Warm-start from the champion itself.** Training begins from the shipped
//!    champion's own weights, so the policy starts *at* the champion's level — the
//!    kept model is never weaker than it.
//! 2. **Self-play RL aimed at the champion.** The policy plays itself and *spars
//!    against the frozen neural champion*, taking the same REINFORCE step the
//!    champion was trained with but with a hotter sampling temperature (the
//!    already-sharp champion policy needs the extra exploration to move). It keeps
//!    the checkpoint that **beats the champion** by the most on a fixed,
//!    training-disjoint deck band, and the run is iterated — each round warm-starts
//!    from the last round's best — to compound the edge (the `train_strong`
//!    example).
//!
//! Because it plays with a single forward pass through the same-sized nets, it
//! matches the champion's speed while winning their head-to-head — see the
//! `tests/strong.rs` integration test and `cargo run -p euchre-eval -- strong
//! neural`.
//!
//! Like [`NeuralAgent`], play is a deterministic masked arg-max, so two agents
//! built from the same weights play identically and matches stay reproducible.

use std::sync::{Arc, LazyLock};

use euchre_interface::{Agent, CallBid, Card, GameView, UpcardBid};

use crate::NeuralAgent;
use crate::neural::NeuralModel;

/// The trained weights shipped with the crate, produced by the `train_strong`
/// example. Parsed once on first use and shared by every [`StrongAgent::pretrained`].
const EMBEDDED_MODEL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/euchre-strong.bin"
));

static PRETRAINED: LazyLock<Arc<NeuralModel>> = LazyLock::new(|| {
    Arc::new(NeuralModel::load(EMBEDDED_MODEL).expect("the embedded strong model is valid"))
});

/// A search-free policy agent whose weights are tuned to beat the neural champion.
///
/// It reuses the [`NeuralAgent`] inference path verbatim (a single masked forward
/// pass per decision); only the weights differ. Construct one from the shipped
/// weights with [`StrongAgent::pretrained`], or wrap an in-memory model with
/// [`StrongAgent::from_model`] / [`StrongAgent::from_shared`] (used by the trainer
/// to evaluate a freshly fit model).
#[derive(Debug, Clone)]
pub struct StrongAgent {
    inner: NeuralAgent,
}

impl StrongAgent {
    /// Builds an agent around an already-loaded model.
    pub fn from_model(model: NeuralModel) -> Self {
        StrongAgent {
            inner: NeuralAgent::from_model(model),
        }
    }

    /// Builds an agent sharing an already-`Arc`-wrapped model, so many agents can
    /// be spun up cheaply without re-parsing or re-cloning the weights.
    pub fn from_shared(model: Arc<NeuralModel>) -> Self {
        StrongAgent {
            inner: NeuralAgent::from_shared(model),
        }
    }

    /// Builds an agent from the weights shipped with the crate. The model is parsed
    /// once and shared, so constructing many pretrained agents (as the evaluation
    /// harness does) is cheap.
    pub fn pretrained() -> Self {
        StrongAgent {
            inner: NeuralAgent::from_shared(PRETRAINED.clone()),
        }
    }

    /// Scores each legal up-card bid by the network's raw logit, for an assist
    /// UI (see [`NeuralAgent::score_bid_upcard`]). Shares the play path, so the
    /// top-scored bid is exactly the one [`StrongAgent`] would make.
    pub fn score_bid_upcard(&self, view: &GameView<'_>) -> Vec<(UpcardBid, f32)> {
        self.inner.score_bid_upcard(view)
    }

    /// Scores each legal second-round call by its raw logit (see
    /// [`NeuralAgent::score_bid_call`]).
    pub fn score_bid_call(&self, view: &GameView<'_>) -> Vec<(CallBid, f32)> {
        self.inner.score_bid_call(view)
    }

    /// Scores each card the dealer could bury by its raw logit (see
    /// [`NeuralAgent::score_discard`]).
    pub fn score_discard(&self, view: &GameView<'_>) -> Vec<(Card, f32)> {
        self.inner.score_discard(view)
    }

    /// Scores each legal card by its raw logit (see [`NeuralAgent::score_play`]).
    pub fn score_play(&self, view: &GameView<'_>, legal: &[Card]) -> Vec<(Card, f32)> {
        self.inner.score_play(view, legal)
    }
}

impl Default for StrongAgent {
    fn default() -> Self {
        StrongAgent::pretrained()
    }
}

impl Agent for StrongAgent {
    fn bid_upcard(&mut self, view: &GameView<'_>) -> UpcardBid {
        self.inner.bid_upcard(view)
    }

    fn bid_call(&mut self, view: &GameView<'_>) -> CallBid {
        self.inner.bid_call(view)
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        self.inner.discard(view)
    }

    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card {
        self.inner.play_card(view, legal)
    }
}
