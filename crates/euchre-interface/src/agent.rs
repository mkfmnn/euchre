//! The [`Agent`] trait: the decisions an AI bot must make to play Euchre.
//!
//! A hand of Euchre proceeds through a bidding phase followed by a play phase.
//! The engine drives the game and calls into the agent at each point where that
//! agent must choose. Every decision is handed a [`GameView`] describing
//! exactly what the agent is allowed to know.
//!
//! ## Bidding
//!
//! After the deal, the top card of the remaining pack is turned face up. Going
//! clockwise from the dealer's left, each seat may either *order up* that suit
//! as trump or *pass* ([`Agent::bid_upcard`]). If everyone passes, a second
//! round begins in which each seat may *name* any other suit as trump or pass
//! ([`Agent::bid_call`]).
//!
//! If a seat orders up the turned card, the dealer takes it into hand and
//! [discards](Agent::discard) one card. The seat that fixes trump (in either
//! round) may choose to play *alone* (its `alone` flag), sitting its partner out
//! for a chance at a larger bonus.
//!
//! ## Play
//!
//! Five tricks are played. On its turn, an agent chooses a card with
//! [`Agent::play_card`], subject to the usual obligation to follow the led suit
//! when able.

use crate::card::{Card, Suit};
use crate::game::GameView;

/// An agent's choice in the first round of bidding, when the up-card's suit is
/// the candidate trump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UpcardBid {
    /// Decline to make the up-card's suit trump and let the auction continue.
    Pass,
    /// Order up the up-card's suit as trump.
    OrderUp { alone: bool },
}

/// An agent's choice in the second round of bidding, when it may name a suit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CallBid {
    /// Decline to name a trump suit.
    Pass,
    /// Name `suit` as trump.
    ///
    /// The suit of the (now turned-down) up-card may not be chosen; the engine
    /// rejects an attempt to call it.
    Call { suit: Suit, alone: bool },
}

/// The strategy an AI bot implements to play Euchre.
///
/// The engine owns turn order, legality checking, and scoring; an agent only
/// supplies decisions. Implementations should be deterministic with respect to
/// their inputs where possible, but this is not required — an agent may consult
/// a random source or run search internally.
///
/// All methods take `&mut self` so an agent may maintain state across the
/// course of a hand or match (for example, remembered cards or opponent
/// models). The engine guarantees that calls for a single agent are made
/// sequentially, never concurrently.
pub trait Agent {
    /// First bidding round: decide whether to order up the turned `view.up_card`,
    /// making its suit trump.
    ///
    /// `view.contract` is `None` at this point. Returning
    /// [`UpcardBid::OrderUp`] fixes trump as `view.up_card.suit`.
    fn bid_upcard(&mut self, view: &GameView<'_>) -> UpcardBid;

    /// Second bidding round: decide whether to name a trump suit.
    ///
    /// Reached only if every seat passed in the first round.
    /// `view.up_card.suit` is not a legal choice.
    ///
    /// In some house rules the dealer is forced to call rather than pass on
    /// this round ("stick the dealer"); agents should inspect `view` to
    /// determine if it applies to them.
    fn bid_call(&mut self, view: &GameView<'_>) -> CallBid;

    /// Choose which card to discard after, as dealer, picking up the ordered-up
    /// card.
    ///
    /// At the moment of this call `view.hand` contains six cards (the original
    /// five plus the up-card just taken in). The returned card must be one of
    /// them; the engine rejects a card not held.
    fn discard(&mut self, view: &GameView<'_>) -> Card;

    /// Choose a card to play to the current trick.
    ///
    /// The agent must follow the led suit if it holds a card of that
    /// (effective) suit; otherwise it may play anything. `legal` lists exactly
    /// the cards that satisfy this obligation, as a convenience and as the
    /// authoritative set the engine will accept. The returned card must appear
    /// in `legal`.
    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card;

    /// Called once when a hand ends, after the final trick, so stateful agents
    /// can learn from the outcome.
    ///
    /// `result` summarizes how the hand was scored. The default implementation
    /// does nothing.
    fn observe_hand_end(&mut self, _view: &GameView<'_>, _result: &HandResult) {}
}

/// How a completed hand turned out, delivered to [`Agent::observe_hand_end`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandResult {
    /// Every seat passed in both bidding rounds, so no trump was named and the
    /// hand was thrown in without being played. No points are awarded.
    PassedOut,
    /// Trump was named and the hand was played out. See [`HandScore`] for the
    /// scoring details.
    Played(HandScore),
}

/// A summary of how a played-out hand was scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HandScore {
    /// Tricks won by the makers (0..=5).
    pub maker_tricks: u8,
    /// Net points awarded to the agent's team;
    /// negative if the other team earned points.
    pub points_awarded: i8,
}

impl HandScore {
    /// Whether the makers were *euchred* — they failed to win at least three of
    /// the five tricks, so the defenders scored.
    pub const fn euchred(self) -> bool {
        self.maker_tricks < 3
    }

    /// Whether the makers swept all five tricks (a *march*).
    pub const fn march(self) -> bool {
        self.maker_tricks == 5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Card, Rank, Suit};
    use crate::game::{GameRules, GameView, Scores, Seat, Trick};

    /// A trivial agent used to prove the trait is object-safe and usable.
    struct FirstLegalAgent;

    impl Agent for FirstLegalAgent {
        fn bid_upcard(&mut self, _view: &GameView<'_>) -> UpcardBid {
            UpcardBid::Pass
        }

        fn bid_call(&mut self, view: &GameView<'_>) -> CallBid {
            // Name the first suit that is not the turned-down up-card's suit.
            let turned_down = view.up_card.suit;
            let suit = Suit::ALL.into_iter().find(|&s| s != turned_down).unwrap();
            CallBid::Call { suit, alone: false }
        }

        fn discard(&mut self, view: &GameView<'_>) -> Card {
            view.hand[0]
        }

        fn play_card(&mut self, _view: &GameView<'_>, legal: &[Card]) -> Card {
            legal[0]
        }
    }

    fn view_with_upcard<'a>(hand: &'a [Card], trick: &'a Trick, up_card: Card) -> GameView<'a> {
        GameView {
            seat: Seat::First,
            up_card,
            hand,
            contract: None,
            discarded: None,
            current_trick: trick,
            completed_tricks: &[],
            scores: Scores::default(),
            rules: GameRules::default(),
        }
    }

    #[test]
    fn agent_is_object_safe() {
        let mut agent: Box<dyn Agent> = Box::new(FirstLegalAgent);
        let trick = Trick::new();
        let hand = [Card::new(Rank::Ace, Suit::Hearts)];
        let up = Card::new(Rank::Nine, Suit::Spades);
        let view = view_with_upcard(&hand, &trick, up);
        assert_eq!(agent.bid_upcard(&view), UpcardBid::Pass);
        assert_eq!(agent.play_card(&view, &hand), hand[0]);
    }

    #[test]
    fn call_bid_avoids_turned_down_suit() {
        let mut agent = FirstLegalAgent;
        let trick = Trick::new();
        let hand = [Card::new(Rank::Ace, Suit::Hearts)];
        // Clubs is the up-card suit, so it must not be the named suit.
        let view = view_with_upcard(&hand, &trick, Card::new(Rank::Nine, Suit::Clubs));
        match agent.bid_call(&view) {
            CallBid::Call { suit, .. } => assert_ne!(suit, Suit::Clubs),
            CallBid::Pass => panic!("expected a call"),
        }
    }
}
