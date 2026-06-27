//! A self-contained perfect-information ("double-dummy") solver for the play of
//! a Euchre hand.
//!
//! [`MonteCarloAgent`](crate::MonteCarloAgent) calls this to evaluate a single
//! *determinization* — a guessed full assignment of the hidden cards. Given every
//! seat's remaining cards laid face up, [`solve`] returns the number of tricks
//! the making team takes under optimal play by both sides, found by alpha-beta
//! search. A Euchre ending is tiny — at most five cards per seat — so the exact
//! search is fast and needs no heuristics of its own.
//!
//! The state carries no hidden information and no randomness, which keeps it pure
//! and trivially testable. All card comparison routes through
//! [`Card::trump_strength`] and [`Card::effective_suit`], so the bower rules stay
//! consistent with the engine's.

use euchre_interface::{Card, GameView, Rank, Seat, Suit};

/// A placeholder for the unused slots of the in-progress trick; never read beyond
/// `trick_len`.
const FILLER: Card = Card::new(Rank::Nine, Suit::Clubs);

/// A perfect-information snapshot of the remaining play, as an omniscient solver
/// sees it. Cheap to copy so the search can fork at every node without
/// allocating.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DdState {
    /// Each seat's remaining cards as a 24-bit set keyed by [`card_index`].
    hands: [u32; 4],
    trump: Suit,
    /// The seat sitting out a loner, if any; it never plays.
    sitting_out: Option<Seat>,
    /// Whose turn it is to play.
    turn: Seat,
    /// Cards played to the in-progress trick, in order; `trick[0]` is the lead.
    /// Only the first `trick_len` entries are meaningful.
    trick: [(Seat, Card); 4],
    trick_len: usize,
    /// The (relative) team that named trump, whose taken tricks the search drives
    /// up. `0` = First/Third, `1` = Second/Dealer; see [`team_index`].
    maker_team: usize,
    /// Tricks the making team has taken so far.
    maker_tricks: u8,
    /// Total cards still to be played across all hands; zero ends the hand.
    cards_left: u8,
}

impl DdState {
    /// Builds the starting search state from a determinized `world` (each seat's
    /// remaining cards, indexed by [`seat_index`]) and the public `view`.
    ///
    /// The in-progress trick and the count of tricks already taken are seeded from
    /// the view, and it is `me`'s turn to play. The cards already played to the
    /// current trick belong to no hand in `world` (they are gone from the deck),
    /// so the partial trick is reconstructed purely from the view.
    pub(crate) fn from_world(
        world: &[Vec<Card>; 4],
        view: &GameView<'_>,
        me: Seat,
        trump: Suit,
    ) -> DdState {
        let contract = view.contract.expect("a hand in play has a contract");
        let maker_team = team_index(contract.maker);

        let mut hands = [0u32; 4];
        let mut cards_left = 0u8;
        for s in Seat::ALL {
            let si = seat_index(s);
            for &c in &world[si] {
                hands[si] |= 1u32 << card_index(c);
                cards_left += 1;
            }
        }

        let mut maker_tricks = 0u8;
        for (_trick, winner) in view.completed_tricks {
            if team_index(*winner) == maker_team {
                maker_tricks += 1;
            }
        }

        let mut trick = [(Seat::First, FILLER); 4];
        let mut trick_len = 0;
        for play in view.current_trick.plays() {
            trick[trick_len] = (play.seat, play.card);
            trick_len += 1;
        }

        DdState {
            hands,
            trump,
            sitting_out: contract.sitting_out(),
            turn: me,
            trick,
            trick_len,
            maker_team,
            maker_tricks,
            cards_left,
        }
    }

    /// Builds the starting state for a freshly dealt hand about to be played out:
    /// every seat's five cards, an empty trick, and `leader` on turn. Used to
    /// evaluate a candidate contract during bidding, where there is no partial
    /// trick or history to seed.
    pub(crate) fn new_play(
        hands: &[Vec<Card>; 4],
        trump: Suit,
        sitting_out: Option<Seat>,
        maker_team: usize,
        leader: Seat,
    ) -> DdState {
        let mut masks = [0u32; 4];
        let mut cards_left = 0u8;
        for s in Seat::ALL {
            let si = seat_index(s);
            for &c in &hands[si] {
                masks[si] |= 1u32 << card_index(c);
                cards_left += 1;
            }
        }
        DdState {
            hands: masks,
            trump,
            sitting_out,
            turn: leader,
            trick: [(Seat::First, FILLER); 4],
            trick_len: 0,
            maker_team,
            maker_tricks: 0,
            cards_left,
        }
    }

    /// The number of seats actually playing (three under a loner, else four).
    fn active_count(&self) -> usize {
        if self.sitting_out.is_some() { 3 } else { 4 }
    }

    /// The next seat to act clockwise, skipping a seat sitting out a loner.
    fn next_active(&self, seat: Seat) -> Seat {
        let candidate = seat.next();
        if Some(candidate) == self.sitting_out {
            candidate.next()
        } else {
            candidate
        }
    }

    /// Plays `card` for the seat on turn, returning the resulting state.
    ///
    /// When the card completes a trick the winner is tallied and leads the next;
    /// otherwise the turn passes to the next active seat.
    pub(crate) fn play(&self, card: Card) -> DdState {
        let mut s = *self;
        s.hands[seat_index(self.turn)] &= !(1u32 << card_index(card));
        s.trick[s.trick_len] = (self.turn, card);
        s.trick_len += 1;
        s.cards_left -= 1;

        if s.trick_len == self.active_count() {
            let led = s.trick[0].1.effective_suit(s.trump);
            let mut winner = s.trick[0].0;
            let mut best = s.trick[0].1.trump_strength(s.trump, led);
            for i in 1..s.trick_len {
                let strength = s.trick[i].1.trump_strength(s.trump, led);
                if strength > best {
                    best = strength;
                    winner = s.trick[i].0;
                }
            }
            if team_index(winner) == s.maker_team {
                s.maker_tricks += 1;
            }
            s.trick_len = 0;
            s.turn = winner;
        } else {
            s.turn = self.next_active(self.turn);
        }
        s
    }

    /// Writes the legal plays for the seat on turn into `buf` and returns the
    /// count: cards following the led effective suit if any are held, else the
    /// whole hand.
    fn legal_moves(&self, buf: &mut [Card; 6]) -> usize {
        let mut all = [FILLER; 6];
        let mut n = 0;
        let mut bits = self.hands[seat_index(self.turn)];
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            all[n] = card_from_index(idx);
            n += 1;
        }

        if self.trick_len == 0 {
            buf[..n].copy_from_slice(&all[..n]);
            return n;
        }

        let led = self.trick[0].1.effective_suit(self.trump);
        let mut m = 0;
        for &c in &all[..n] {
            if c.effective_suit(self.trump) == led {
                buf[m] = c;
                m += 1;
            }
        }
        if m == 0 {
            buf[..n].copy_from_slice(&all[..n]);
            n
        } else {
            m
        }
    }
}

/// The making team's trick count (0..=5) under optimal play from `state`.
pub(crate) fn solve(state: &DdState) -> u8 {
    solve_ab(state, 0, 5)
}

/// Alpha-beta search on the bounded integer "tricks taken by the making team".
///
/// It is a two-team zero-sum game: every maker-team seat drives the count up
/// (a maximizer), every defender drives it down (a minimizer). The tight `[0, 5]`
/// bounds make cutoffs frequent, so no transposition table is needed at this
/// depth.
fn solve_ab(state: &DdState, mut alpha: u8, mut beta: u8) -> u8 {
    if state.cards_left == 0 {
        return state.maker_tricks;
    }

    let mut buf = [FILLER; 6];
    let n = state.legal_moves(&mut buf);
    debug_assert!(
        n > 0,
        "a seat on turn with cards left always has a legal play"
    );
    if n == 0 {
        return state.maker_tricks;
    }

    if team_index(state.turn) == state.maker_team {
        let mut value = 0;
        for &card in &buf[..n] {
            value = value.max(solve_ab(&state.play(card), alpha, beta));
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        value
    } else {
        let mut value = 5;
        for &card in &buf[..n] {
            value = value.min(solve_ab(&state.play(card), alpha, beta));
            beta = beta.min(value);
            if beta <= alpha {
                break;
            }
        }
        value
    }
}

/// Maps a card to a stable `0..24` index (suit-major, rank-minor), matching the
/// engine's convention.
pub(crate) fn card_index(card: Card) -> usize {
    suit_index(card.suit) * 6 + card.rank as usize
}

/// The inverse of [`card_index`].
fn card_from_index(idx: usize) -> Card {
    Card::new(Rank::ALL[idx % 6], Suit::ALL[idx / 6])
}

/// Maps a seat to a stable `0..4` index matching [`Seat::ALL`].
pub(crate) fn seat_index(seat: Seat) -> usize {
    match seat {
        Seat::First => 0,
        Seat::Second => 1,
        Seat::Third => 2,
        Seat::Dealer => 3,
    }
}

/// The relative team a seat belongs to (`0` = First/Third, `1` = Second/Dealer).
pub(crate) fn team_index(seat: Seat) -> usize {
    seat_index(seat) % 2
}

/// Maps a suit to a stable `0..4` index matching [`Suit::ALL`].
pub(crate) fn suit_index(suit: Suit) -> usize {
    match suit {
        Suit::Clubs => 0,
        Suit::Diamonds => 1,
        Suit::Hearts => 2,
        Suit::Spades => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    /// Builds a fresh-trick state from explicit per-seat hands, with `turn`
    /// leading. A thin wrapper over the production [`DdState::new_play`].
    fn state(
        hands: [Vec<Card>; 4],
        trump: Suit,
        sitting_out: Option<Seat>,
        turn: Seat,
        maker_team: usize,
    ) -> DdState {
        DdState::new_play(&hands, trump, sitting_out, maker_team, turn)
    }

    #[test]
    fn card_index_round_trips() {
        for s in Suit::ALL {
            for r in Rank::ALL {
                let c = card(r, s);
                assert_eq!(card_from_index(card_index(c)), c);
            }
        }
    }

    #[test]
    fn lone_trump_wins_the_single_trick() {
        // North (maker) holds the right bower against three off-suit cards.
        let hands = [
            vec![card(Rank::Jack, Suit::Spades)],
            vec![card(Rank::Ace, Suit::Hearts)],
            vec![card(Rank::Ace, Suit::Diamonds)],
            vec![card(Rank::Ace, Suit::Clubs)],
        ];
        let st = state(hands, Suit::Spades, None, Seat::First, 0);
        assert_eq!(solve(&st), 1);
    }

    #[test]
    fn left_bower_is_trump_and_takes_the_trick() {
        // Trump hearts. North leads the ace of diamonds; East holds the left bower
        // (jack of diamonds), which is trump, so East is void in diamonds and wins
        // by ruffing. Maker is East/West, so the optimal count is one maker trick.
        let hands = [
            vec![card(Rank::Ace, Suit::Diamonds)],
            vec![card(Rank::Jack, Suit::Diamonds)], // left bower under hearts
            vec![card(Rank::Nine, Suit::Diamonds)],
            vec![card(Rank::Nine, Suit::Clubs)],
        ];
        let st = state(hands, Suit::Hearts, None, Seat::First, 1);
        assert_eq!(solve(&st), 1);
    }

    #[test]
    fn a_hand_of_top_trump_marches() {
        // North holds the five highest trumps and leads; nothing can stop a sweep.
        let north = vec![
            card(Rank::Jack, Suit::Spades), // right bower
            card(Rank::Jack, Suit::Clubs),  // left bower
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Queen, Suit::Spades),
        ];
        let east = vec![
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
            card(Rank::Queen, Suit::Hearts),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::King, Suit::Diamonds),
        ];
        let south = vec![
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
            card(Rank::Queen, Suit::Diamonds),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Diamonds),
        ];
        let west = vec![
            card(Rank::Ace, Suit::Clubs),
            card(Rank::King, Suit::Clubs),
            card(Rank::Queen, Suit::Clubs),
            card(Rank::Ten, Suit::Clubs),
            card(Rank::Nine, Suit::Clubs),
        ];
        let st = state(
            [north, east, south, west],
            Suit::Spades,
            None,
            Seat::First,
            0,
        );
        assert_eq!(solve(&st), 5);
    }

    #[test]
    fn loner_skips_the_sitting_seat_and_marches() {
        // North goes alone with the top five trumps; South sits out. Only three
        // seats play, tricks complete at three cards, and North sweeps.
        let north = vec![
            card(Rank::Jack, Suit::Spades),
            card(Rank::Jack, Suit::Clubs),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Queen, Suit::Spades),
        ];
        let east = vec![
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
            card(Rank::Queen, Suit::Hearts),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::King, Suit::Diamonds),
        ];
        let west = vec![
            card(Rank::Ace, Suit::Clubs),
            card(Rank::King, Suit::Clubs),
            card(Rank::Queen, Suit::Clubs),
            card(Rank::Ten, Suit::Clubs),
            card(Rank::Nine, Suit::Clubs),
        ];
        let st = state(
            [north, east, vec![], west],
            Suit::Spades,
            Some(Seat::Third),
            Seat::First,
            0,
        );
        assert_eq!(solve(&st), 5);
    }

    #[test]
    fn follow_suit_is_obligatory_even_over_a_bower() {
        // East holds a heart and the right bower; with a heart led it may only
        // play the heart, not ruff.
        let mut st = state(
            [
                vec![card(Rank::Ace, Suit::Hearts)],
                vec![
                    card(Rank::King, Suit::Hearts),
                    card(Rank::Jack, Suit::Spades),
                ],
                vec![],
                vec![],
            ],
            Suit::Spades,
            None,
            Seat::First,
            0,
        );
        // Seed a heart lead by North and put East on turn.
        st.trick[0] = (Seat::First, card(Rank::Ace, Suit::Hearts));
        st.trick_len = 1;
        st.turn = Seat::Second;
        let mut buf = [FILLER; 6];
        let n = st.legal_moves(&mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf[0], card(Rank::King, Suit::Hearts));
    }

    #[test]
    fn solve_is_deterministic() {
        let hands = [
            vec![
                card(Rank::Jack, Suit::Spades),
                card(Rank::Nine, Suit::Diamonds),
            ],
            vec![
                card(Rank::Ace, Suit::Hearts),
                card(Rank::King, Suit::Hearts),
            ],
            vec![card(Rank::Ace, Suit::Clubs), card(Rank::King, Suit::Clubs)],
            vec![
                card(Rank::Queen, Suit::Hearts),
                card(Rank::Queen, Suit::Clubs),
            ],
        ];
        let st = state(hands, Suit::Spades, None, Seat::First, 0);
        assert_eq!(solve(&st), solve(&st));
    }

    #[test]
    fn new_play_leader_decides_an_off_suit_trick() {
        // One card each, no trump in play: the led suit decides which ace wins, so
        // the leader determines the outcome. This pins down `new_play`'s `leader`.
        let hands = [
            vec![card(Rank::Ace, Suit::Hearts)],    // North
            vec![card(Rank::Ace, Suit::Diamonds)],  // East
            vec![card(Rank::Nine, Suit::Hearts)],   // South
            vec![card(Rank::Nine, Suit::Diamonds)], // West
        ];
        // North leads hearts → the heart ace wins for the makers.
        let north_leads = DdState::new_play(&hands, Suit::Spades, None, 0, Seat::First);
        assert_eq!(solve(&north_leads), 1);
        // East leads diamonds → the diamond ace wins for the defenders.
        let east_leads = DdState::new_play(&hands, Suit::Spades, None, 0, Seat::Second);
        assert_eq!(solve(&east_leads), 0);
    }
}
