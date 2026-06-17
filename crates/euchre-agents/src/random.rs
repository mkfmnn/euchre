//! [`RandomAgent`]: a bot that picks uniformly at random among its legal
//! choices.
//!
//! This agent applies no strategy whatsoever. At each decision it enumerates the
//! options the engine would accept and chooses one with equal probability. It is
//! useful as a baseline opponent, as a fuzz source for the engine, and as a
//! sanity check that a smarter agent actually beats noise.

use euchre_interface::{Agent, Bid, CallBid, Card, GameView, Suit, UpcardBid};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;

/// An agent that selects a legal option uniformly at random at every decision.
#[derive(Debug, Clone)]
pub struct RandomAgent {
    rng: SmallRng,
}

impl RandomAgent {
    /// Creates a random agent seeded from system entropy.
    pub fn new() -> Self {
        RandomAgent {
            rng: SmallRng::from_rng(&mut rand::rng()),
        }
    }

    /// Creates a random agent with a fixed seed, for reproducible play.
    pub fn with_seed(seed: u64) -> Self {
        RandomAgent {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Whether this seat is the dealer forced to name a suit under the
    /// "stick the dealer" rule, and so may not pass the second round.
    fn is_stuck(view: &GameView<'_>) -> bool {
        view.rules.stick_the_dealer && view.seat == view.dealer
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        RandomAgent::new()
    }
}

impl Agent for RandomAgent {
    fn bid_upcard(&mut self, _view: &GameView<'_>, _up_card: Card) -> UpcardBid {
        // The three legal answers: pass, order up with the partner, or alone.
        let options = [
            UpcardBid::Pass,
            UpcardBid::OrderUp(Bid::WithPartner),
            UpcardBid::OrderUp(Bid::Alone),
        ];
        *options.choose(&mut self.rng).expect("options is non-empty")
    }

    fn bid_call(&mut self, view: &GameView<'_>, turned_down: Suit) -> CallBid {
        // Every nameable suit, each with both bid styles, plus a pass when one
        // is allowed.
        let mut options = Vec::with_capacity(7);
        if !Self::is_stuck(view) {
            options.push(CallBid::Pass);
        }
        for suit in Suit::ALL {
            if suit == turned_down {
                continue;
            }
            options.push(CallBid::Call {
                suit,
                bid: Bid::WithPartner,
            });
            options.push(CallBid::Call {
                suit,
                bid: Bid::Alone,
            });
        }
        *options.choose(&mut self.rng).expect("options is non-empty")
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        *view.hand.choose(&mut self.rng).expect("hand is non-empty")
    }

    fn play_card(&mut self, _view: &GameView<'_>, legal: &[Card]) -> Card {
        *legal.choose(&mut self.rng).expect("legal is non-empty")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::{GameRules, Rank, Scores, Seat, Trick};

    fn view_for<'a>(
        hand: &'a [Card],
        trick: &'a Trick,
        seat: Seat,
        dealer: Seat,
        rules: GameRules,
    ) -> GameView<'a> {
        GameView {
            seat,
            dealer,
            hand,
            contract: None,
            current_trick: trick,
            completed_tricks: &[],
            scores: Scores::default(),
            rules,
        }
    }

    #[test]
    fn discard_and_play_stay_in_the_legal_set() {
        let mut agent = RandomAgent::with_seed(42);
        let hand = [
            Card::new(Rank::Nine, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::King, Suit::Spades),
        ];
        let trick = Trick::new();
        let view = view_for(
            &hand,
            &trick,
            Seat::North,
            Seat::North,
            GameRules::default(),
        );
        for _ in 0..50 {
            assert!(hand.contains(&agent.discard(&view)));
            assert!(hand.contains(&agent.play_card(&view, &hand)));
        }
    }

    #[test]
    fn never_names_the_turned_down_suit() {
        let mut agent = RandomAgent::with_seed(99);
        let hand = [Card::new(Rank::Nine, Suit::Clubs)];
        let trick = Trick::new();
        let view = view_for(&hand, &trick, Seat::East, Seat::North, GameRules::default());
        for _ in 0..200 {
            if let CallBid::Call { suit, .. } = agent.bid_call(&view, Suit::Diamonds) {
                assert_ne!(suit, Suit::Diamonds);
            }
        }
    }

    #[test]
    fn a_stuck_dealer_never_passes() {
        let mut agent = RandomAgent::with_seed(7);
        let hand = [Card::new(Rank::Nine, Suit::Clubs)];
        let trick = Trick::new();
        let rules = GameRules {
            stick_the_dealer: true,
        };
        // The dealer, on the second round, is stuck and must name a suit.
        let view = view_for(&hand, &trick, Seat::North, Seat::North, rules);
        for _ in 0..200 {
            assert!(matches!(
                agent.bid_call(&view, Suit::Spades),
                CallBid::Call { .. }
            ));
        }
    }
}
