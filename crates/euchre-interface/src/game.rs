//! Seats and the observable game state passed to an [`Agent`].
//!
//! [`Agent`]: crate::agent::Agent

use crate::card::{Card, Suit};

/// One of the four seats at the table, named by its position *relative to the
/// dealer*.
///
/// `Dealer` dealt the hand; `First` (the seat to the dealer's immediate left) is
/// first to bid and first to lead, then `Second` and `Third` follow clockwise.
/// Because the deal rotates each hand, these labels are relative — the same
/// physical player occupies a different `Seat` from one hand to the next. The
/// engine and other fixed-identity consumers track players by a separate index;
/// an [`Agent`] only ever reasons in these dealer-relative terms.
///
/// Partners sit across the table, so `First`/`Third` are one team and
/// `Second`/`Dealer` the other.
///
/// [`Agent`]: crate::agent::Agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Seat {
    First,
    Second,
    Third,
    Dealer,
}

impl Seat {
    /// All four seats in bidding order, from the dealer's left around to the
    /// dealer.
    pub const ALL: [Seat; 4] = [Seat::First, Seat::Second, Seat::Third, Seat::Dealer];

    /// The seat to the immediate left (the next seat clockwise), which is the
    /// next to act or play.
    pub const fn next(self) -> Seat {
        match self {
            Seat::First => Seat::Second,
            Seat::Second => Seat::Third,
            Seat::Third => Seat::Dealer,
            Seat::Dealer => Seat::First,
        }
    }

    /// This seat's partner, sitting directly across the table.
    pub const fn partner(self) -> Seat {
        match self {
            Seat::First => Seat::Third,
            Seat::Third => Seat::First,
            Seat::Second => Seat::Dealer,
            Seat::Dealer => Seat::Second,
        }
    }

    /// Whether `other` is on this seat's team — either this very seat or its
    /// partner across the table.
    pub const fn same_team(self, other: Seat) -> bool {
        matches!(
            (self, other),
            (Seat::First | Seat::Third, Seat::First | Seat::Third)
                | (Seat::Second | Seat::Dealer, Seat::Second | Seat::Dealer)
        )
    }
}

/// A card played into the current trick, tagged with the seat that played it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Play {
    pub seat: Seat,
    pub card: Card,
}

/// The cards played so far in the trick currently in progress.
///
/// Plays are stored in the order they were made; the first play establishes the
/// led suit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Trick {
    plays: Vec<Play>,
}

impl Trick {
    /// An empty trick with no cards played yet.
    pub fn new() -> Self {
        Trick { plays: Vec::new() }
    }

    /// The plays made so far, in order.
    pub fn plays(&self) -> &[Play] {
        &self.plays
    }

    /// Whether no card has been played to this trick yet.
    pub fn is_empty(&self) -> bool {
        self.plays.is_empty()
    }

    /// The number of cards played to this trick.
    pub fn len(&self) -> usize {
        self.plays.len()
    }

    /// Records a play. Intended for use by the game engine, not agents.
    pub fn push(&mut self, play: Play) {
        self.plays.push(play);
    }

    /// The suit that was led, or `None` if the trick is empty.
    ///
    /// This is the [effective suit](Card::effective_suit) of the first card,
    /// so a led left bower correctly reports as the trump suit.
    pub fn led_suit(&self, trump: Suit) -> Option<Suit> {
        self.plays.first().map(|p| p.card.effective_suit(trump))
    }

    /// The seat currently winning the trick, or `None` if it is empty.
    pub fn winner(&self, trump: Suit) -> Option<Seat> {
        let led = self.led_suit(trump)?;
        self.plays
            .iter()
            .max_by_key(|p| p.card.trump_strength(trump, led))
            .map(|p| p.seat)
    }
}

/// How a hand's trump was decided, including whether the maker went alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Contract {
    /// The chosen trump suit.
    pub trump: Suit,
    /// The seat that named trump (the "maker").
    pub maker: Seat,
    /// Whether the maker is playing alone, with their partner sitting out.
    pub alone: bool,
}

impl Contract {
    /// The seat sitting out this hand, if the maker chose to go alone.
    pub fn sitting_out(&self) -> Option<Seat> {
        if self.alone {
            Some(self.maker.partner())
        } else {
            None
        }
    }
}

/// The house rules in effect for a match.
///
/// Euchre is played with a number of optional rule variations; this collects
/// the ones the engine supports so agents can adjust their strategy to them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GameRules {
    /// Whether "stick the dealer" is in effect: if every seat passes the first
    /// round and every non-dealer seat passes the second, the dealer is forced
    /// to name a trump suit rather than being allowed to pass it out.
    pub stick_the_dealer: bool,
}

/// A read-only view of everything an agent legitimately knows when it is asked
/// to make a decision.
///
/// The engine constructs this and hands it to the [`Agent`]; agents should
/// treat it as the complete, authoritative picture of the public game state
/// plus their own private hand. It deliberately excludes hidden information
/// such as opponents' hands or the undealt cards in the kitty.
///
/// [`Agent`]: crate::agent::Agent
#[derive(Debug, Clone)]
pub struct GameView<'a> {
    /// The seat this agent occupies.
    pub seat: Seat,
    /// The card turned up for bidding by the dealer at the end of the deal.
    pub up_card: Card,
    /// The cards currently in the agent's hand.
    pub hand: &'a [Card],
    /// The agreed contract for this hand, once trump has been named.
    ///
    /// This is `None` during bidding, before a trump suit has been chosen.
    pub contract: Option<Contract>,
    /// The card discarded by the dealer. This is only populated for the
    /// dealer's view, and only if the dealer did discard.
    pub discarded: Option<Card>,
    /// The trick currently in progress.
    pub current_trick: &'a Trick,
    /// Completed tricks this hand, oldest first, each paired with the seat that
    /// won it.
    pub completed_tricks: &'a [(Trick, Seat)],
    /// Cumulative match score, from this seat's point of view.
    pub scores: Scores,
    /// The house rules in effect for this match.
    pub rules: GameRules,
}

impl GameView<'_> {
    /// The trump suit, if a contract has been established.
    pub fn trump(&self) -> Option<Suit> {
        self.contract.map(|c| c.trump)
    }
}

/// Cumulative match scores, told from one seat's point of view: `us` is the
/// viewing seat's own team, `them` the opponents'.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Scores {
    /// The viewing seat's own team's score.
    pub us: u8,
    /// The opposing team's score.
    pub them: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Card, Rank};

    #[test]
    fn seat_relationships() {
        assert_eq!(Seat::First.next(), Seat::Second);
        assert_eq!(Seat::Dealer.next(), Seat::First);
        assert_eq!(Seat::First.partner(), Seat::Third);
        assert!(Seat::First.same_team(Seat::Third));
        assert!(Seat::Second.same_team(Seat::Dealer));
        assert!(!Seat::First.same_team(Seat::Second));
        assert!(!Seat::First.same_team(Seat::Dealer));
    }

    #[test]
    fn trick_winner_accounts_for_trump() {
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // First leads the ace of hearts.
        trick.push(Play {
            seat: Seat::First,
            card: Card::new(Rank::Ace, Suit::Hearts),
        });
        // Second follows with a heart.
        trick.push(Play {
            seat: Seat::Second,
            card: Card::new(Rank::King, Suit::Hearts),
        });
        // Third trumps in with the nine of spades.
        trick.push(Play {
            seat: Seat::Third,
            card: Card::new(Rank::Nine, Suit::Spades),
        });
        assert_eq!(trick.led_suit(trump), Some(Suit::Hearts));
        assert_eq!(trick.winner(trump), Some(Seat::Third));
    }

    #[test]
    fn led_left_bower_reports_as_trump() {
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        trick.push(Play {
            seat: Seat::First,
            card: Card::new(Rank::Jack, Suit::Clubs), // left bower
        });
        assert_eq!(trick.led_suit(trump), Some(Suit::Spades));
    }

    #[test]
    fn going_alone_seats_out_partner() {
        let contract = Contract {
            trump: Suit::Hearts,
            maker: Seat::First,
            alone: true,
        };
        assert_eq!(contract.sitting_out(), Some(Seat::Third));
    }
}
