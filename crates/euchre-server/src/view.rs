//! Translation between the engine's types and the wire protocol.
//!
//! This is the single place that converts a client's [`ClientMsg`] into an
//! engine [`Decision`], derives the public [`PublicAction`] from what was
//! actually applied, builds the per-seat [`PlayerView`] snapshot, and forms the
//! [`TurnHint`] for a pending [`Action`]. Keeping it in one module means the
//! hidden-information rule (a discard reveals no card) has exactly one home.

use euchre_agents::NeuralAgent;
use euchre_engine::{Action, Decision, Game};
use euchre_interface::{CallBid, GameView, Suit, UpcardBid};

use crate::protocol::{
    ClientMsg, Player, PlayerView, PublicAction, ScoredAction, SuggestedAction, TurnHint,
};

/// Builds the owned [`PlayerView`] snapshot for the fixed `player` from the live
/// game.
pub fn snapshot(game: &Game, player: Player) -> PlayerView {
    let seat = game.seat_of(player as usize);
    let v = game.view(seat);
    PlayerView {
        seat: player,
        dealer: game.dealer() as Player,
        hand: v.hand.to_vec(),
        contract: v.contract,
        current_trick: v.current_trick.clone(),
        completed_tricks: v.completed_tricks.to_vec(),
        scores: v.scores,
        rules: v.rules,
    }
}

/// The hint describing what `action` asks of the active seat.
pub fn hint_for(action: &Action, game: &Game) -> TurnHint {
    match action {
        Action::BidUpcard { .. } => TurnHint::Bid {
            up: true,
            may_pass: true,
        },
        Action::BidCall { may_pass, .. } => TurnHint::Bid {
            up: false,
            may_pass: *may_pass,
        },
        Action::Discard { .. } => TurnHint::Discard,
        Action::Play { .. } => {
            let lead = game
                .contract()
                .and_then(|c| game.current_trick().led_suit(c.trump));
            TurnHint::Play { lead }
        }
        Action::HandComplete { .. } => TurnHint::Discard, // never asked of a client
    }
}

/// Maps a client's message to the engine [`Decision`] the pending `action`
/// expects, or returns a human-readable reason it does not fit.
///
/// The engine still validates the result (suit legality, follow-suit, card
/// ownership); this only checks that the *kind* of message matches the phase.
pub fn decision_from(msg: &ClientMsg, action: &Action) -> Result<Decision, String> {
    match action {
        Action::BidUpcard { .. } => match msg {
            ClientMsg::Pass => Ok(Decision::Upcard(UpcardBid::Pass)),
            ClientMsg::Bid { alone, .. } => {
                Ok(Decision::Upcard(UpcardBid::OrderUp { alone: *alone }))
            }
            _ => Err("expected BID or PASS".into()),
        },
        Action::BidCall { .. } => match msg {
            ClientMsg::Pass => Ok(Decision::Call(CallBid::Pass)),
            ClientMsg::Bid { suit, alone } => Ok(Decision::Call(CallBid::Call {
                suit: *suit,
                alone: *alone,
            })),
            _ => Err("expected BID or PASS".into()),
        },
        Action::Discard { .. } => match msg {
            ClientMsg::Discard { card } => Ok(Decision::Discard(*card)),
            _ => Err("expected DISCARD".into()),
        },
        Action::Play { .. } => match msg {
            ClientMsg::Play { card } => Ok(Decision::Play(*card)),
            _ => Err("expected PLAY".into()),
        },
        Action::HandComplete { .. } => Err("no decision is expected right now".into()),
    }
}

/// Derives the public broadcast form of a decision that was just applied.
///
/// `action` is the pending action it answered, needed to recover the up-card's
/// suit for an order-up (which the [`Decision`] itself does not carry).
pub fn public_action(action: &Action, decision: &Decision) -> PublicAction {
    match decision {
        Decision::Upcard(UpcardBid::Pass) | Decision::Call(CallBid::Pass) => PublicAction::Pass,
        Decision::Upcard(UpcardBid::OrderUp { alone }) => {
            let suit = match action {
                Action::BidUpcard { up_card, .. } => up_card.suit,
                _ => unreachable!("order-up answers a BidUpcard action"),
            };
            PublicAction::Bid {
                suit,
                alone: *alone,
            }
        }
        Decision::Call(CallBid::Call { suit, alone }) => PublicAction::Bid {
            suit: *suit,
            alone: *alone,
        },
        Decision::Discard(_) => PublicAction::Discard,
        Decision::Play(card) => PublicAction::Play { card: *card },
    }
}

/// Builds the assist suggestion for the pending `action` from `agent`'s view of
/// the active seat: the recommended move plus, for every option, its raw score
/// and the probability it is the best move.
///
/// The recommended move is the highest-scoring option, so it always matches the
/// move the [`NeuralAgent`] would actually make. Probabilities are a softmax of
/// the raw scores over the legal options (the net trains its logits as exactly
/// this masked softmax), so they sum to 1 and read as a confidence per option.
/// Returns `None` for [`Action::HandComplete`], which asks no decision.
pub fn suggestion(
    agent: &NeuralAgent,
    action: &Action,
    view: &GameView<'_>,
) -> Option<(SuggestedAction, Vec<ScoredAction>)> {
    let raw: Vec<(SuggestedAction, f32)> = match action {
        Action::BidUpcard { .. } => {
            let up_suit = view.up_card.suit;
            agent
                .score_bid_upcard(view)
                .into_iter()
                .map(|(bid, score)| (upcard_suggested(bid, up_suit), score))
                .collect()
        }
        Action::BidCall { .. } => agent
            .score_bid_call(view)
            .into_iter()
            .map(|(bid, score)| (call_suggested(bid), score))
            .collect(),
        Action::Discard { .. } => agent
            .score_discard(view)
            .into_iter()
            .map(|(card, score)| (SuggestedAction::Discard { card }, score))
            .collect(),
        Action::Play { legal, .. } => agent
            .score_play(view, legal)
            .into_iter()
            .map(|(card, score)| (SuggestedAction::Play { card }, score))
            .collect(),
        Action::HandComplete { .. } => return None,
    };
    if raw.is_empty() {
        return None;
    }
    let probabilities = softmax(raw.iter().map(|(_, score)| *score));
    let scores: Vec<ScoredAction> = raw
        .into_iter()
        .zip(probabilities)
        .map(|((action, score), probability)| ScoredAction {
            action,
            score,
            probability,
        })
        .collect();
    let recommended = scores
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))?
        .action
        .clone();
    Some((recommended, scores))
}

/// A numerically-stable softmax turning raw option scores into probabilities
/// that sum to 1. This matches the net's own masked softmax, since the scores
/// are exactly the logits of the legal options.
fn softmax(scores: impl Iterator<Item = f32>) -> Vec<f32> {
    let scores: Vec<f32> = scores.collect();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Translates a first-round up-card bid into the wire suggestion vocabulary,
/// naming the up-card's suit on an order-up so the UI can match the call.
fn upcard_suggested(bid: UpcardBid, up_suit: Suit) -> SuggestedAction {
    match bid {
        UpcardBid::Pass => SuggestedAction::Pass,
        UpcardBid::OrderUp { alone } => SuggestedAction::Bid {
            suit: up_suit,
            alone,
        },
    }
}

/// Translates a second-round call into the wire suggestion vocabulary.
fn call_suggested(bid: CallBid) -> SuggestedAction {
    match bid {
        CallBid::Pass => SuggestedAction::Pass,
        CallBid::Call { suit, alone } => SuggestedAction::Bid { suit, alone },
    }
}
