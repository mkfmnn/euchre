//! A bot that makes uniformly random legal choices.

use euchre_interface::{Agent, Bid, CallBid, Card, GameView, Suit, UpcardBid};

use crate::rng::Rng;

/// An agent that bids and plays at random, always within the rules.
///
/// It orders up or names trump a fixed fraction of the time, never goes alone,
/// discards a random card, and plays a uniformly random legal card. It plays
/// badly on purpose: it exists as a baseline opponent and as a cheap source of
/// legal games for testing and benchmarking stronger agents.
#[derive(Debug, Clone)]
pub struct RandomAgent {
    rng: Rng,
    /// Probability (out of 100) of making trump when given the chance.
    make_chance: u32,
}

impl RandomAgent {
    /// Creates a random agent with a default 25% inclination to make trump.
    pub fn new(rng: Rng) -> Self {
        RandomAgent {
            rng,
            make_chance: 25,
        }
    }

    /// Sets how eager the agent is to name trump, as a percentage `0..=100`.
    pub fn with_make_chance(mut self, percent: u32) -> Self {
        self.make_chance = percent.min(100);
        self
    }

    fn wants_to_make(&mut self) -> bool {
        self.rng.below(100) < self.make_chance
    }

    fn pick(&mut self, cards: &[Card]) -> Card {
        cards[self.rng.below(cards.len() as u32) as usize]
    }
}

impl Agent for RandomAgent {
    fn bid_upcard(&mut self, _view: &GameView<'_>, _up_card: Card) -> UpcardBid {
        if self.wants_to_make() {
            UpcardBid::OrderUp(Bid::WithPartner)
        } else {
            UpcardBid::Pass
        }
    }

    fn bid_call(&mut self, _view: &GameView<'_>, turned_down: Suit) -> CallBid {
        if self.wants_to_make() {
            // Choose a random legal suit (anything but the turned-down one).
            let choices: Vec<Suit> = Suit::ALL
                .into_iter()
                .filter(|&s| s != turned_down)
                .collect();
            let suit = choices[self.rng.below(choices.len() as u32) as usize];
            CallBid::Call {
                suit,
                bid: Bid::WithPartner,
            }
        } else {
            CallBid::Pass
        }
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        self.pick(view.hand)
    }

    fn play_card(&mut self, _view: &GameView<'_>, legal: &[Card]) -> Card {
        self.pick(legal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    #[test]
    fn random_agents_finish_a_match() {
        let agents: [Box<dyn Agent>; 4] = [
            Box::new(RandomAgent::new(Rng::seed_from_u64(1))),
            Box::new(RandomAgent::new(Rng::seed_from_u64(2))),
            Box::new(RandomAgent::new(Rng::seed_from_u64(3))),
            Box::new(RandomAgent::new(Rng::seed_from_u64(4))),
        ];
        let config = EngineConfig {
            seed: Some(123),
            ..Default::default()
        };
        let mut engine = Engine::new(agents, config);
        let outcome = engine.play_match();
        assert!(outcome.winner_score() >= 10);
        assert!(outcome.hands_played > 0);
    }

    #[test]
    fn matches_are_reproducible() {
        let run = |seed| {
            let agents: [Box<dyn Agent>; 4] = [
                Box::new(RandomAgent::new(Rng::seed_from_u64(10))),
                Box::new(RandomAgent::new(Rng::seed_from_u64(20))),
                Box::new(RandomAgent::new(Rng::seed_from_u64(30))),
                Box::new(RandomAgent::new(Rng::seed_from_u64(40))),
            ];
            let mut engine = Engine::new(
                agents,
                EngineConfig {
                    seed: Some(seed),
                    ..Default::default()
                },
            );
            engine.play_match()
        };
        assert_eq!(run(555), run(555));
    }
}
