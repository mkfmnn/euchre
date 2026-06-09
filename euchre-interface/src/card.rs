//! Core card types for Euchre.
//!
//! Euchre is played with a 24-card deck: the ranks Nine through Ace in each of
//! the four suits. Trump handling is unusual because the Jack of the trump suit
//! (the *right bower*) and the Jack of the same color (the *left bower*) become
//! the two highest cards and the left bower is treated as belonging to the
//! trump suit for the duration of the hand.

use std::fmt;

/// The color of a suit. Used to determine the left bower, which is the Jack of
/// the suit that shares the trump suit's color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Red,
    Black,
}

/// One of the four suits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

impl Suit {
    /// All four suits, in a stable order.
    pub const ALL: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    /// The color of this suit.
    pub const fn color(self) -> Color {
        match self {
            Suit::Diamonds | Suit::Hearts => Color::Red,
            Suit::Clubs | Suit::Spades => Color::Black,
        }
    }

    /// The other suit of the same color. The Jack of this suit is the left
    /// bower when `self` is trump.
    pub const fn same_color(self) -> Suit {
        match self {
            Suit::Clubs => Suit::Spades,
            Suit::Spades => Suit::Clubs,
            Suit::Diamonds => Suit::Hearts,
            Suit::Hearts => Suit::Diamonds,
        }
    }

    /// The single-character symbol for this suit.
    pub const fn symbol(self) -> char {
        match self {
            Suit::Clubs => '♣',
            Suit::Diamonds => '♦',
            Suit::Hearts => '♥',
            Suit::Spades => '♠',
        }
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Suit::Clubs => "Clubs",
            Suit::Diamonds => "Diamonds",
            Suit::Hearts => "Hearts",
            Suit::Spades => "Spades",
        })
    }
}

/// A card rank. Euchre only uses Nine through Ace.
///
/// The ordering of the variants reflects the natural (non-trump) rank order,
/// with `Nine` lowest and `Ace` highest. Note that this ordering does **not**
/// account for bowers; use [`Card::trump_strength`] for trump-aware comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rank {
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    /// All ranks used in Euchre, from lowest to highest.
    pub const ALL: [Rank; 6] = [
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    /// The short label for this rank (e.g. `"J"` for Jack, `"10"` for Ten).
    pub const fn label(self) -> &'static str {
        match self {
            Rank::Nine => "9",
            Rank::Ten => "10",
            Rank::Jack => "J",
            Rank::Queen => "Q",
            Rank::King => "K",
            Rank::Ace => "A",
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A single playing card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    /// Constructs a card.
    pub const fn new(rank: Rank, suit: Suit) -> Self {
        Card { rank, suit }
    }

    /// The full 24-card Euchre deck, in a stable order.
    pub fn deck() -> Vec<Card> {
        let mut cards = Vec::with_capacity(24);
        for suit in Suit::ALL {
            for rank in Rank::ALL {
                cards.push(Card::new(rank, suit));
            }
        }
        cards
    }

    /// Whether this card is the right bower (the Jack of the trump suit) given
    /// `trump`.
    pub fn is_right_bower(self, trump: Suit) -> bool {
        self.rank == Rank::Jack && self.suit == trump
    }

    /// Whether this card is the left bower (the Jack of the suit matching
    /// `trump`'s color) given `trump`.
    pub fn is_left_bower(self, trump: Suit) -> bool {
        self.rank == Rank::Jack && self.suit == trump.same_color()
    }

    /// Whether this card is either bower given `trump`.
    pub fn is_bower(self, trump: Suit) -> bool {
        self.is_right_bower(trump) || self.is_left_bower(trump)
    }

    /// Whether this card counts as a trump card given `trump`.
    ///
    /// This is true for any card of the trump suit and additionally for the
    /// left bower, which plays as a trump even though its printed suit differs.
    pub fn is_trump(self, trump: Suit) -> bool {
        self.suit == trump || self.is_left_bower(trump)
    }

    /// The *effective* suit of this card for the purpose of following suit.
    ///
    /// This is the printed suit for every card except the left bower, which is
    /// considered part of the trump suit.
    pub fn effective_suit(self, trump: Suit) -> Suit {
        if self.is_left_bower(trump) {
            trump
        } else {
            self.suit
        }
    }

    /// A trump-aware strength score for ranking cards within a trick.
    ///
    /// Higher scores beat lower scores. The scale is only meaningful for
    /// comparing two cards under the same `trump` and `led` suit; do not read
    /// absolute meaning into the numbers. Cards that are neither trump nor of
    /// the led suit score lowest because they cannot win a trick.
    pub fn trump_strength(self, trump: Suit, led: Suit) -> u32 {
        // Bowers and trump form the top of the order.
        if self.is_right_bower(trump) {
            return 1000;
        }
        if self.is_left_bower(trump) {
            return 999;
        }
        if self.suit == trump {
            return 100 + self.rank as u32;
        }
        // Non-trump cards can only win if they followed the led suit.
        if self.effective_suit(trump) == led {
            return 10 + self.rank as u32;
        }
        // Off-suit discard: cannot win.
        self.rank as u32
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank.label(), self.suit.symbol())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_has_24_unique_cards() {
        let deck = Card::deck();
        assert_eq!(deck.len(), 24);
        let mut sorted = deck.clone();
        sorted.sort_by_key(|c| (c.suit, c.rank));
        sorted.dedup();
        assert_eq!(sorted.len(), 24);
    }

    #[test]
    fn bowers_are_identified() {
        let right = Card::new(Rank::Jack, Suit::Spades);
        let left = Card::new(Rank::Jack, Suit::Clubs);
        assert!(right.is_right_bower(Suit::Spades));
        assert!(!right.is_left_bower(Suit::Spades));
        assert!(left.is_left_bower(Suit::Spades));
        assert!(!left.is_right_bower(Suit::Spades));
        assert!(left.is_trump(Suit::Spades));
        assert_eq!(left.effective_suit(Suit::Spades), Suit::Spades);
    }

    #[test]
    fn left_bower_follows_trump_not_printed_suit() {
        let left = Card::new(Rank::Jack, Suit::Hearts);
        // Trump is diamonds (same color as hearts).
        assert!(left.is_left_bower(Suit::Diamonds));
        assert_eq!(left.effective_suit(Suit::Diamonds), Suit::Diamonds);
        // When trump is a black suit, the heart Jack is just a heart.
        assert!(!left.is_trump(Suit::Spades));
        assert_eq!(left.effective_suit(Suit::Spades), Suit::Hearts);
    }

    #[test]
    fn trump_ordering_is_correct() {
        let trump = Suit::Hearts;
        let led = Suit::Hearts;
        let right = Card::new(Rank::Jack, Suit::Hearts);
        let left = Card::new(Rank::Jack, Suit::Diamonds);
        let ace_trump = Card::new(Rank::Ace, Suit::Hearts);
        let nine_trump = Card::new(Rank::Nine, Suit::Hearts);
        let off_ace = Card::new(Rank::Ace, Suit::Spades);

        assert!(right.trump_strength(trump, led) > left.trump_strength(trump, led));
        assert!(left.trump_strength(trump, led) > ace_trump.trump_strength(trump, led));
        assert!(ace_trump.trump_strength(trump, led) > nine_trump.trump_strength(trump, led));
        assert!(nine_trump.trump_strength(trump, led) > off_ace.trump_strength(trump, led));
    }

    #[test]
    fn led_suit_beats_off_suit() {
        let trump = Suit::Hearts;
        let led = Suit::Clubs;
        let led_nine = Card::new(Rank::Nine, Suit::Clubs);
        let off_ace = Card::new(Rank::Ace, Suit::Spades);
        assert!(led_nine.trump_strength(trump, led) > off_ace.trump_strength(trump, led));
    }
}
