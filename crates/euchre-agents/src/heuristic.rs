//! [`HeuristicAgent`]: a bot that plays with a handful of simple, human-style
//! rules of thumb.
//!
//! The agent is intentionally lightweight — no search, no opponent modeling —
//! but it plays a recognizably sensible game and comfortably beats a
//! [random][crate::RandomAgent] opponent. Its decisions break down as:
//!
//! * **Bidding** is driven by a [hand-strength score](hand_strength): bowers,
//!   trump, and off-aces are worth points, short suits add a little for their
//!   ruffing potential, and the up-card is credited or debited depending on who
//!   would pick it up. The agent makes trump when the score clears a threshold
//!   and goes alone when it is overwhelming.
//! * **Discarding** (as the dealer who took the up-card) throws the lowest
//!   off-trump card, preferring to void a suit so it can ruff later.
//! * **Playing** leads off-aces and high trump, wins tricks as cheaply as it
//!   can, ducks when its partner is already winning, and sloughs its weakest
//!   card when it cannot win.

use euchre_interface::{Agent, Bid, CallBid, Card, GameView, Rank, Suit, UpcardBid};

/// Hand-strength score at or above which the agent makes trump.
const MAKE_THRESHOLD: i32 = 9;
/// Hand-strength score at or above which the agent makes trump *alone*.
const ALONE_THRESHOLD: i32 = 15;

/// An agent that plays with basic heuristic strategy.
///
/// It is stateless between decisions; everything it needs comes from the
/// [`GameView`] it is handed. Construct one with [`HeuristicAgent::new`].
#[derive(Debug, Clone, Default)]
pub struct HeuristicAgent;

impl HeuristicAgent {
    /// Creates a heuristic agent.
    pub fn new() -> Self {
        HeuristicAgent
    }

    /// Whether this seat is the dealer forced to name a suit under the
    /// "stick the dealer" rule, and so may not pass the second round.
    fn is_stuck(view: &GameView<'_>) -> bool {
        view.rules.stick_the_dealer && view.seat == view.dealer
    }

    /// Chooses a card to lead (the agent is first to play this trick).
    fn lead(view: &GameView<'_>, legal: &[Card], trump: Suit) -> Card {
        // An off-suit ace is a likely winner and frees up a future void.
        if let Some(&ace) = legal
            .iter()
            .find(|c| c.rank == Rank::Ace && !c.is_trump(trump))
        {
            return ace;
        }

        // As the maker, draw the opponents' trump by leading our highest.
        let we_made = view
            .contract
            .map(|c| c.maker.team() == view.seat.team())
            .unwrap_or(false);
        let trump_held = legal.iter().filter(|c| c.is_trump(trump)).count();
        if we_made && trump_held >= 2 {
            return *highest_trump(legal, trump).expect("trump_held >= 2");
        }

        // Otherwise lead our lowest non-trump card, hoarding trump for later;
        // if we are all trump, lead the lowest trump.
        lowest_non_trump(legal, trump)
            .copied()
            .unwrap_or_else(|| *lowest_trump(legal, trump).expect("hand is non-empty"))
    }

    /// Chooses a card when following into a trick already in progress.
    fn follow(view: &GameView<'_>, legal: &[Card], trump: Suit) -> Card {
        let trick = view.current_trick;
        let led = trick.led_suit(trump).expect("trick is non-empty");
        let winning = trick
            .plays()
            .iter()
            .max_by_key(|p| p.card.trump_strength(trump, led))
            .expect("trick is non-empty");

        // If our partner is already winning, don't waste a high card on it.
        if winning.seat == view.seat.partner() {
            return weakest(legal, trump, led);
        }

        // Otherwise win as cheaply as possible if we can; else throw our worst.
        let winning_strength = winning.card.trump_strength(trump, led);
        legal
            .iter()
            .filter(|c| c.trump_strength(trump, led) > winning_strength)
            .min_by_key(|c| c.trump_strength(trump, led))
            .copied()
            .unwrap_or_else(|| weakest(legal, trump, led))
    }
}

impl Agent for HeuristicAgent {
    fn bid_upcard(&mut self, view: &GameView<'_>, up_card: Card) -> UpcardBid {
        let trump = up_card.suit;
        let me = view.seat;
        let dealer = view.dealer;

        let strength = if dealer == me {
            // We are the dealer: we would take the up-card and bury our worst.
            // Score the six-card hand; the discard step trims the weakest card.
            let mut cards = view.hand.to_vec();
            cards.push(up_card);
            hand_strength(&cards, trump) - DISCARD_CORRECTION
        } else {
            let base = hand_strength(view.hand, trump);
            // The dealer pockets the up-card as trump: a gift to our side if the
            // dealer is us or our partner, a handicap if it is an opponent.
            let gift = trump_card_value(up_card, trump);
            if dealer == me.partner() {
                base + gift
            } else {
                base - gift
            }
        };

        if strength >= ALONE_THRESHOLD {
            UpcardBid::OrderUp(Bid::Alone)
        } else if strength >= MAKE_THRESHOLD {
            UpcardBid::OrderUp(Bid::WithPartner)
        } else {
            UpcardBid::Pass
        }
    }

    fn bid_call(&mut self, view: &GameView<'_>, turned_down: Suit) -> CallBid {
        // Score every nameable suit and keep the best.
        let best = Suit::ALL
            .into_iter()
            .filter(|&s| s != turned_down)
            .map(|s| (s, hand_strength(view.hand, s)))
            .max_by_key(|&(_, score)| score)
            .expect("three suits remain after the turned-down one");
        let (suit, strength) = best;

        if strength >= ALONE_THRESHOLD {
            CallBid::Call {
                suit,
                bid: Bid::Alone,
            }
        } else if strength >= MAKE_THRESHOLD || Self::is_stuck(view) {
            // Clear the bar, or take the best suit anyway when forced to call.
            CallBid::Call {
                suit,
                bid: Bid::WithPartner,
            }
        } else {
            CallBid::Pass
        }
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        let trump = view.trump().expect("trump is set before the discard");
        let hand = view.hand;

        let non_trump: Vec<Card> = hand
            .iter()
            .copied()
            .filter(|c| !c.is_trump(trump))
            .collect();
        if non_trump.is_empty() {
            // A hand of all trump: part with the weakest trump.
            return *lowest_trump(hand, trump).expect("hand is non-empty");
        }

        let count_in = |suit: Suit| non_trump.iter().filter(|c| c.suit == suit).count();

        // Prefer to void a suit by dropping a low singleton, opening up ruffs.
        if let Some(&singleton) = non_trump
            .iter()
            .filter(|c| c.rank != Rank::Ace && count_in(c.suit) == 1)
            .min_by_key(|c| c.rank)
        {
            return singleton;
        }

        // Otherwise drop the lowest non-trump card, sparing aces when we can.
        let keep_ace = non_trump.iter().any(|c| c.rank != Rank::Ace);
        *non_trump
            .iter()
            .filter(|c| !keep_ace || c.rank != Rank::Ace)
            .min_by_key(|c| c.rank)
            .expect("non_trump is non-empty")
    }

    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card {
        let trump = view.trump().expect("trump is set during play");
        if view.current_trick.is_empty() {
            Self::lead(view, legal, trump)
        } else {
            Self::follow(view, legal, trump)
        }
    }
}

// --- Hand evaluation ---------------------------------------------------------

/// Correction subtracted from a six-card score to approximate the value of the
/// five-card hand the dealer keeps after discarding the weakest card.
const DISCARD_CORRECTION: i32 = 1;

/// A rough strength score for `cards` if `trump` were trump.
///
/// The scale is arbitrary but monotonic: bowers and high trump dominate,
/// off-aces help, and being short in a side suit adds a little because a void
/// lets the hand ruff. It is the common currency behind every bidding decision.
fn hand_strength(cards: &[Card], trump: Suit) -> i32 {
    let mut score = 0;
    let mut trump_count = 0;
    let mut has_suit = [false; 4];

    for &card in cards {
        if card.is_trump(trump) {
            trump_count += 1;
        }
        score += if card.is_trump(trump) {
            trump_card_value(card, trump)
        } else {
            match card.rank {
                Rank::Ace => 2,
                Rank::King => 1,
                _ => 0,
            }
        };
        has_suit[suit_index(card.effective_suit(trump))] = true;
    }

    // Voids only help if we have trump to ruff with; weight by how much trump.
    if trump_count >= 1 {
        let voids = Suit::ALL
            .into_iter()
            .filter(|&s| s != trump && !has_suit[suit_index(s)])
            .count() as i32;
        score += voids * trump_count.min(2);
    }

    score
}

/// The standalone value of a single trump card, used both inside
/// [`hand_strength`] and to price the up-card a dealer would pick up.
fn trump_card_value(card: Card, trump: Suit) -> i32 {
    if card.is_right_bower(trump) {
        6
    } else if card.is_left_bower(trump) {
        5
    } else {
        debug_assert!(card.is_trump(trump));
        match card.rank {
            Rank::Ace => 4,
            Rank::King => 3,
            Rank::Queen => 2,
            _ => 1,
        }
    }
}

// --- Card selection helpers --------------------------------------------------

/// The weakest card to part with: lowest trump strength, which favors dumping a
/// low off-suit card and keeping trump and the led suit.
fn weakest(cards: &[Card], trump: Suit, led: Suit) -> Card {
    *cards
        .iter()
        .min_by_key(|c| c.trump_strength(trump, led))
        .expect("cards is non-empty")
}

fn highest_trump(cards: &[Card], trump: Suit) -> Option<&Card> {
    cards
        .iter()
        .filter(|c| c.is_trump(trump))
        .max_by_key(|c| c.trump_strength(trump, trump))
}

fn lowest_trump(cards: &[Card], trump: Suit) -> Option<&Card> {
    cards
        .iter()
        .filter(|c| c.is_trump(trump))
        .min_by_key(|c| c.trump_strength(trump, trump))
}

fn lowest_non_trump(cards: &[Card], trump: Suit) -> Option<&Card> {
    cards
        .iter()
        .filter(|c| !c.is_trump(trump))
        .min_by_key(|c| c.rank)
}

/// Maps a suit to a stable array index, matching [`Suit::ALL`].
fn suit_index(suit: Suit) -> usize {
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
    use euchre_interface::{Contract, GameRules, Play, Scores, Seat, Trick};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    fn play_view<'a>(
        hand: &'a [Card],
        trick: &'a Trick,
        seat: Seat,
        contract: Contract,
    ) -> GameView<'a> {
        GameView {
            seat,
            dealer: Seat::North,
            hand,
            contract: Some(contract),
            current_trick: trick,
            completed_tricks: &[],
            scores: Scores::default(),
            rules: GameRules::default(),
        }
    }

    fn bidding_view<'a>(hand: &'a [Card], seat: Seat, dealer: Seat) -> GameView<'a> {
        // A throwaway empty trick to borrow for the view during bidding.
        static EMPTY: std::sync::OnceLock<Trick> = std::sync::OnceLock::new();
        let trick = EMPTY.get_or_init(Trick::new);
        GameView {
            seat,
            dealer,
            hand,
            contract: None,
            current_trick: trick,
            completed_tricks: &[],
            scores: Scores::default(),
            rules: GameRules::default(),
        }
    }

    #[test]
    fn bowers_and_trump_outweigh_junk() {
        let strong = [
            card(Rank::Jack, Suit::Spades), // right bower
            card(Rank::Jack, Suit::Clubs),  // left bower
            card(Rank::Ace, Suit::Spades),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
        ];
        let weak = [
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        assert!(hand_strength(&strong, Suit::Spades) > hand_strength(&weak, Suit::Spades));
    }

    #[test]
    fn orders_up_a_powerful_hand_and_passes_a_bare_one() {
        let mut agent = HeuristicAgent::new();
        let strong = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Jack, Suit::Clubs),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
        ];
        // West is dealer (an opponent), so ordering up gives them the up-card.
        let view = bidding_view(&strong, Seat::North, Seat::West);
        assert!(matches!(
            agent.bid_upcard(&view, card(Rank::Nine, Suit::Spades)),
            UpcardBid::OrderUp(_)
        ));

        let junk = [
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Queen, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        let view = bidding_view(&junk, Seat::North, Seat::West);
        assert_eq!(
            agent.bid_upcard(&view, card(Rank::Nine, Suit::Spades)),
            UpcardBid::Pass
        );
    }

    #[test]
    fn discard_voids_a_low_singleton() {
        let mut agent = HeuristicAgent::new();
        // Trump is spades. A lone low heart is the natural void candidate.
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Diamonds),
            card(Rank::Queen, Suit::Diamonds),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Spades),
        ];
        let contract = Contract {
            trump: Suit::Spades,
            maker: Seat::North,
            alone: false,
        };
        let trick = Trick::new();
        let view = play_view(&hand, &trick, Seat::North, contract);
        assert_eq!(agent.discard(&view), card(Rank::Nine, Suit::Hearts));
    }

    #[test]
    fn discard_keeps_an_ace_over_a_low_card() {
        let mut agent = HeuristicAgent::new();
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Nine, Suit::Spades),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::King, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
            card(Rank::Ten, Suit::Clubs),
        ];
        let contract = Contract {
            trump: Suit::Spades,
            maker: Seat::North,
            alone: false,
        };
        let trick = Trick::new();
        let view = play_view(&hand, &trick, Seat::North, contract);
        let discard = agent.discard(&view);
        assert_ne!(discard.rank, Rank::Ace);
        assert!(!discard.is_trump(Suit::Spades));
    }

    #[test]
    fn leads_an_off_ace_when_holding_one() {
        let mut agent = HeuristicAgent::new();
        let hand = [
            card(Rank::Nine, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ten, Suit::Diamonds),
        ];
        let contract = Contract {
            trump: Suit::Spades,
            maker: Seat::South,
            alone: false,
        };
        let trick = Trick::new();
        let view = play_view(&hand, &trick, Seat::North, contract);
        assert_eq!(agent.play_card(&view, &hand), card(Rank::Ace, Suit::Hearts));
    }

    #[test]
    fn wins_a_trick_as_cheaply_as_possible() {
        let mut agent = HeuristicAgent::new();
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // East leads the king of hearts; we (South) hold two winning hearts.
        trick.push(Play {
            seat: Seat::East,
            card: card(Rank::King, Suit::Hearts),
        });
        let hand = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Nine, Suit::Spades),
        ];
        // Following hearts: only the ace can win (the spade is a separate option,
        // but we must follow the led heart suit, so legal is just the ace here).
        let legal = [card(Rank::Ace, Suit::Hearts)];
        let contract = Contract {
            trump,
            maker: Seat::East,
            alone: false,
        };
        let view = play_view(&hand, &trick, Seat::South, contract);
        assert_eq!(
            agent.play_card(&view, &legal),
            card(Rank::Ace, Suit::Hearts)
        );
    }

    #[test]
    fn ducks_when_partner_is_winning() {
        let mut agent = HeuristicAgent::new();
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // Our partner (North) leads the right bower and is winning; we are South.
        trick.push(Play {
            seat: Seat::North,
            card: card(Rank::Jack, Suit::Spades),
        });
        trick.push(Play {
            seat: Seat::East,
            card: card(Rank::Nine, Suit::Hearts),
        });
        // We are void in trump's led suit; legal is our whole hand.
        let hand = [
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::Nine, Suit::Diamonds),
        ];
        let contract = Contract {
            trump,
            maker: Seat::North,
            alone: false,
        };
        let view = play_view(&hand, &trick, Seat::South, contract);
        // Partner has it won, so we slough the nine, not the ace.
        assert_eq!(
            agent.play_card(&view, &hand),
            card(Rank::Nine, Suit::Diamonds)
        );
    }
}
