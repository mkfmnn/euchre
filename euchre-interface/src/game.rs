//! Seats, teams, and the observable game state passed to an [`Agent`].
//!
//! [`Agent`]: crate::agent::Agent

use crate::card::{Card, Suit};

/// One of the four seats at the table, arranged clockwise.
///
/// Seats `North`/`South` form one team and `East`/`West` form the other, so
/// partners sit across from each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Seat {
    North,
    East,
    South,
    West,
}

impl Seat {
    /// All four seats in clockwise order starting from `North`.
    pub const ALL: [Seat; 4] = [Seat::North, Seat::East, Seat::South, Seat::West];

    /// The seat to the immediate left (the next seat clockwise), which is the
    /// next to act or play.
    pub const fn next(self) -> Seat {
        match self {
            Seat::North => Seat::East,
            Seat::East => Seat::South,
            Seat::South => Seat::West,
            Seat::West => Seat::North,
        }
    }

    /// This seat's partner, sitting directly across the table.
    pub const fn partner(self) -> Seat {
        match self {
            Seat::North => Seat::South,
            Seat::South => Seat::North,
            Seat::East => Seat::West,
            Seat::West => Seat::East,
        }
    }

    /// The team this seat belongs to.
    pub const fn team(self) -> Team {
        match self {
            Seat::North | Seat::South => Team::NorthSouth,
            Seat::East | Seat::West => Team::EastWest,
        }
    }
}

/// One of the two partnerships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Team {
    NorthSouth,
    EastWest,
}

impl Team {
    /// The opposing team.
    pub const fn opponent(self) -> Team {
        match self {
            Team::NorthSouth => Team::EastWest,
            Team::EastWest => Team::NorthSouth,
        }
    }
}

/// A card played into the current trick, tagged with the seat that played it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Play {
    pub seat: Seat,
    pub card: Card,
}

/// The cards played so far in the trick currently in progress.
///
/// Plays are stored in the order they were made; the first play establishes the
/// led suit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
    /// The seat that dealt this hand.
    pub dealer: Seat,
    /// The cards currently in the agent's hand.
    pub hand: &'a [Card],
    /// The agreed contract for this hand, once trump has been named.
    ///
    /// This is `None` during bidding, before a trump suit has been chosen.
    pub contract: Option<Contract>,
    /// The trick currently in progress.
    pub current_trick: &'a Trick,
    /// Completed tricks this hand, oldest first, each paired with the seat that
    /// won it.
    pub completed_tricks: &'a [(Trick, Seat)],
    /// Cumulative match score for each team.
    pub scores: Scores,
}

impl GameView<'_> {
    /// The trump suit, if a contract has been established.
    pub fn trump(&self) -> Option<Suit> {
        self.contract.map(|c| c.trump)
    }
}

/// Cumulative match scores for both teams.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scores {
    pub north_south: u8,
    pub east_west: u8,
}

impl Scores {
    /// The score for a given team.
    pub const fn for_team(&self, team: Team) -> u8 {
        match team {
            Team::NorthSouth => self.north_south,
            Team::EastWest => self.east_west,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Card, Rank};

    #[test]
    fn seat_relationships() {
        assert_eq!(Seat::North.next(), Seat::East);
        assert_eq!(Seat::West.next(), Seat::North);
        assert_eq!(Seat::North.partner(), Seat::South);
        assert_eq!(Seat::East.team(), Team::EastWest);
        assert_eq!(Seat::North.team().opponent(), Team::EastWest);
    }

    #[test]
    fn trick_winner_accounts_for_trump() {
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // North leads the ace of hearts.
        trick.push(Play {
            seat: Seat::North,
            card: Card::new(Rank::Ace, Suit::Hearts),
        });
        // East follows with a heart.
        trick.push(Play {
            seat: Seat::East,
            card: Card::new(Rank::King, Suit::Hearts),
        });
        // South trumps in with the nine of spades.
        trick.push(Play {
            seat: Seat::South,
            card: Card::new(Rank::Nine, Suit::Spades),
        });
        assert_eq!(trick.led_suit(trump), Some(Suit::Hearts));
        assert_eq!(trick.winner(trump), Some(Seat::South));
    }

    #[test]
    fn led_left_bower_reports_as_trump() {
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        trick.push(Play {
            seat: Seat::North,
            card: Card::new(Rank::Jack, Suit::Clubs), // left bower
        });
        assert_eq!(trick.led_suit(trump), Some(Suit::Spades));
    }

    #[test]
    fn going_alone_seats_out_partner() {
        let contract = Contract {
            trump: Suit::Hearts,
            maker: Seat::North,
            alone: true,
        };
        assert_eq!(contract.sitting_out(), Some(Seat::South));
    }
}
