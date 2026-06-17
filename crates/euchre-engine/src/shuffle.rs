//! Dealing a shuffled deck.
//!
//! The [`Game`](crate::Game) core is deterministic and never shuffles — it deals
//! whatever 24-card deck it is handed. Producing that shuffled deck is the
//! caller's job; this module supplies the one helper both the terminal
//! [driver](crate::driver) and a server use, so there is a single place that
//! turns randomness into a deal.

use euchre_interface::Card;
use rand::Rng;
use rand::seq::SliceRandom;

/// A freshly shuffled full 24-card Euchre deck, drawn from `rng`.
///
/// Pass the result to [`Game::new`](crate::Game::new) or
/// [`Game::start_next_hand`](crate::Game::start_next_hand). The shuffle quality
/// is only as good as `rng`; for reproducible deals seed the generator.
pub fn deal<R: Rng + ?Sized>(rng: &mut R) -> [Card; 24] {
    let mut cards = Card::deck();
    cards.shuffle(rng);
    cards.try_into().expect("Card::deck yields 24 cards")
}
