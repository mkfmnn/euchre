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

impl NeuralAgent {
    /// Scores each legal up-card bid (pass, order up, order up alone) by the raw
    /// logit the [`Head::Upcard`] net assigns it. This is the same forward pass
    /// [`Agent::bid_upcard`] makes, exposed so an assist UI can show *why* the
    /// agent prefers a move; the highest-scored bid is exactly the one
    /// [`Agent::bid_upcard`] would play. Higher is better; the values are not
    /// probabilities.
    pub fn score_bid_upcard(&self, view: &GameView<'_>) -> Vec<(UpcardBid, f32)> {
        let feats = features::upcard_features(view);
        let logits = self.model.net(Head::Upcard).forward(&feats);
        legal_scores(&logits, features::upcard_legal())
            .map(|(class, score)| (features::upcard_action(class), score))
            .collect()
    }

    /// Scores each legal second-round call by the raw logit the [`Head::Call`]
    /// net assigns it (see [`NeuralAgent::score_bid_upcard`]).
    pub fn score_bid_call(&self, view: &GameView<'_>) -> Vec<(CallBid, f32)> {
        let turned_down = view.up_card.suit;
        let stuck = view.rules.stick_the_dealer && view.seat == Seat::Dealer;
        let feats = features::call_features(view, turned_down, stuck);
        let logits = self.model.net(Head::Call).forward(&feats);
        legal_scores(&logits, features::call_legal(stuck))
            .map(|(class, score)| (features::call_action(class, turned_down), score))
            .collect()
    }

    /// Scores each card the dealer could bury by the raw logit the
    /// [`Head::Discard`] net assigns its slot (see
    /// [`NeuralAgent::score_bid_upcard`]).
    pub fn score_discard(&self, view: &GameView<'_>) -> Vec<(Card, f32)> {
        let trump = view.trump().expect("trump is set before the discard");
        let feats = features::discard_features(view);
        let logits = self.model.net(Head::Discard).forward(&feats);
        view.hand
            .iter()
            .map(|&c| (c, logits[features::card_slot(c, trump)]))
            .collect()
    }

    /// Scores each legal card by the raw logit the [`Head::Play`] net assigns
    /// its slot (see [`NeuralAgent::score_bid_upcard`]).
    pub fn score_play(&self, view: &GameView<'_>, legal: &[Card]) -> Vec<(Card, f32)> {
        let trump = view.trump().expect("trump is set during play");
        let feats = features::play_features(view);
        let logits = self.model.net(Head::Play).forward(&feats);
        legal
            .iter()
            .map(|&c| (c, logits[features::card_slot(c, trump)]))
            .collect()
    }
}

/// Iterates `(class, logit)` over exactly the classes set in the `legal` mask.
fn legal_scores(logits: &[f32], legal: u32) -> impl Iterator<Item = (usize, f32)> + '_ {
    (0..logits.len())
        .filter(move |&k| legal & (1 << k) != 0)
        .map(|k| (k, logits[k]))
}

/// The legal class with the highest logit.
fn argmax_legal(logits: &[f32], legal: u32) -> usize {
    (0..logits.len())
        .filter(|&k| legal & (1 << k) != 0)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .expect("at least one class is legal")
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::{Card, Contract, GameRules, Rank, Scores, Suit, Trick};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    /// The top-scored option from each `score_*` method must be exactly the move
    /// the agent actually makes — the contract the assist UI relies on to outline
    /// the recommended control.
    #[test]
    fn top_score_matches_the_agents_decision() {
        let agent = NeuralAgent::pretrained();

        // Bidding: hand and up-card with no contract yet.
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Clubs),
        ];
        let empty = Trick::new();
        let bidding = GameView {
            seat: Seat::First,
            up_card: card(Rank::Queen, Suit::Spades),
            hand: &hand,
            contract: None,
            discarded: None,
            current_trick: &empty,
            completed_tricks: &[],
            scores: Scores::default(),
            rules: GameRules::default(),
        };
        assert_eq!(
            top(&agent.score_bid_upcard(&bidding)),
            NeuralAgent::pretrained().bid_upcard(&bidding),
            "upcard recommendation disagrees with bid_upcard",
        );
        assert_eq!(
            top(&agent.score_bid_call(&bidding)),
            NeuralAgent::pretrained().bid_call(&bidding),
            "call recommendation disagrees with bid_call",
        );

        // Play: a spades contract, leading the first trick with five cards.
        let contract = Contract {
            trump: Suit::Spades,
            maker: Seat::First,
            alone: false,
        };
        let playing = GameView {
            seat: Seat::First,
            up_card: card(Rank::Queen, Suit::Spades),
            hand: &hand,
            contract: Some(contract),
            discarded: None,
            current_trick: &empty,
            completed_tricks: &[],
            scores: Scores::default(),
            rules: GameRules::default(),
        };
        let legal = hand.to_vec();
        assert_eq!(
            top(&agent.score_play(&playing, &legal)),
            NeuralAgent::pretrained().play_card(&playing, &legal),
            "play recommendation disagrees with play_card",
        );
    }

    /// The option with the highest score (the assist UI's recommended move).
    fn top<T: Copy>(scored: &[(T, f32)]) -> T {
        scored
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty")
            .0
    }
}
