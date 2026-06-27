//! [`NeuralAgent`]: a search-free bot driven entirely by the trained policy
//! networks.
//!
//! Every decision is a single forward pass through the relevant head, followed
//! by a masked arg-max over the legal options — no tree search, no sampling, no
//! per-hand state. Because the choice is a deterministic arg-max, two agents
//! built from the same model play identically, which keeps matches reproducible.

use std::sync::{Arc, LazyLock};

use euchre_interface::{Agent, CallBid, Card, GameView, Seat, UpcardBid};

use super::features::{self, Head};
use super::train::NeuralModel;

/// The trained weights shipped with the crate, distilled from the
/// [`AdvancedAgent`](crate::AdvancedAgent) by the `train_neural` example. Parsed
/// once on first use and shared by every [`NeuralAgent::pretrained`].
const EMBEDDED_MODEL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/euchre-net.bin"
));

static PRETRAINED: LazyLock<Arc<NeuralModel>> = LazyLock::new(|| {
    Arc::new(NeuralModel::load(EMBEDDED_MODEL).expect("the embedded model is valid"))
});

/// A bot that plays by evaluating a trained [`NeuralModel`].
///
/// Construct one from an in-memory model with [`NeuralAgent::from_model`] (used by
/// the trainer to evaluate a freshly fit model) or from the weights shipped with
/// the crate with [`NeuralAgent::pretrained`].
#[derive(Debug, Clone)]
pub struct NeuralAgent {
    model: Arc<NeuralModel>,
}

impl NeuralAgent {
    /// Builds an agent around an already-loaded model.
    pub fn from_model(model: NeuralModel) -> Self {
        NeuralAgent {
            model: Arc::new(model),
        }
    }

    /// Builds an agent sharing an already-`Arc`-wrapped model, so many agents can
    /// be spun up cheaply without re-parsing or re-cloning the weights.
    pub fn from_shared(model: Arc<NeuralModel>) -> Self {
        NeuralAgent { model }
    }

    /// Builds an agent from the weights shipped with the crate. The model is
    /// parsed once and shared, so constructing many pretrained agents (as the
    /// evaluation harness does) is cheap.
    pub fn pretrained() -> Self {
        NeuralAgent {
            model: PRETRAINED.clone(),
        }
    }
}

impl Default for NeuralAgent {
    fn default() -> Self {
        NeuralAgent::pretrained()
    }
}

impl Agent for NeuralAgent {
    fn bid_upcard(&mut self, view: &GameView<'_>) -> UpcardBid {
        let feats = features::upcard_features(view);
        let logits = self.model.net(Head::Upcard).forward(&feats);
        let class = argmax_legal(&logits, features::upcard_legal());
        features::upcard_action(class)
    }

    fn bid_call(&mut self, view: &GameView<'_>) -> CallBid {
        let turned_down = view.up_card.suit;
        let stuck = view.rules.stick_the_dealer && view.seat == Seat::Dealer;
        let feats = features::call_features(view, turned_down, stuck);
        let logits = self.model.net(Head::Call).forward(&feats);
        let class = argmax_legal(&logits, features::call_legal(stuck));
        features::call_action(class, turned_down)
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        let trump = view.trump().expect("trump is set before the discard");
        let feats = features::discard_features(view);
        let logits = self.model.net(Head::Discard).forward(&feats);
        features::best_card(&logits, view.hand, trump)
    }

    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card {
        if legal.len() == 1 {
            return legal[0];
        }
        let trump = view.trump().expect("trump is set during play");
        let feats = features::play_features(view);
        let logits = self.model.net(Head::Play).forward(&feats);
        features::best_card(&logits, legal, trump)
    }
}

/// The legal class with the highest logit.
fn argmax_legal(logits: &[f32], legal: u32) -> usize {
    (0..logits.len())
        .filter(|&k| legal & (1 << k) != 0)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .expect("at least one class is legal")
}
