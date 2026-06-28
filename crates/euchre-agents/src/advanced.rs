//! [`AdvancedAgent`]: a stronger heuristic bot that plays close to expert lines
//! without search or learning.
//!
//! Where [`HeuristicAgent`](crate::HeuristicAgent) plays sensible rules of thumb,
//! this agent layers on the knowledge that separates a good Euchre player from a
//! beginner. Everything is still hand-written heuristics — no tree search, no
//! Monte-Carlo, no machine learning — but the heuristics encode real strategy:
//!
//! * **Trick-counting hand evaluation.** Instead of an abstract point score,
//!   bidding is driven by an estimate of how many of the five tricks the hand is
//!   worth ([`expected_tricks`]). Trump length, the two bowers, off-aces, and
//!   ruffing voids each contribute, and the thresholds are expressed directly in
//!   tricks (you need three to avoid being euchred).
//! * **Position-aware bidding.** The seat relative to the dealer changes
//!   everything. Ordering up from the dealer's team hands *your* side the extra
//!   trump; from the defending seats it arms the dealer. The agent credits or
//!   debits the up-card accordingly, and the dealer evaluates the hand it would
//!   hold *after* taking the card and discarding.
//! * **The "next" / "green" calling convention.** When the up-card is turned
//!   down, the defending team is statistically favoured to call the *same colour*
//!   as the rejected suit ("next"), while the dealer's team does better calling
//!   the opposite colour ("green"). The agent applies this well-known edge.
//! * **Card counting in the play.** The agent reconstructs every card already
//!   played and every suit a seat has shown out of, purely from the public
//!   [`GameView`]. From that it knows which of its cards are *masters* (the
//!   highest of their suit still live) and which leads can be safely cashed, when
//!   to draw trump as the maker, when a partner's trick is already secure, and
//!   which dead card to throw away.
//! * **Score-aware aggression.** Near the end of a match the agent presses when
//!   the opponents are on the hill and plays safer when it is itself one hand
//!   from winning.

use euchre_interface::{Agent, CallBid, Card, GameView, Rank, Seat, Suit, UpcardBid};

// --- Bidding thresholds (in expected tricks) ---------------------------------

/// Estimated tricks (from this hand alone, trusting the partner for the rest) at
/// or above which the agent makes trump with its partner.
const MAKE_TRICKS: f64 = 2.1;
/// Estimated solo tricks at or above which the agent will consider going alone,
/// provided it also has top-trump control.
const ALONE_TRICKS: f64 = 3.7;

/// Bonus tricks credited to a candidate trump suit that follows the "next"
/// convention for the defending team (calling the colour of the turned-down
/// suit).
const NEXT_BONUS: f64 = 0.5;
/// Penalty applied when the dealer's team is tempted to call "next"; that team
/// is better off going "green" (the opposite colour).
const NEXT_PENALTY: f64 = 0.3;
/// Bonus for the dealer's team calling a "green" (cross-colour) suit.
const GREEN_BONUS: f64 = 0.25;

/// An advanced heuristic agent.
///
/// It is stateless between calls: every decision is reconstructed from the
/// [`GameView`] it is handed, so there is nothing to reset between hands and the
/// agent is trivially reusable and reproducible. Construct one with
/// [`AdvancedAgent::new`].
#[derive(Debug, Clone, Default)]
pub struct AdvancedAgent;

impl AdvancedAgent {
    /// Creates an advanced agent.
    pub fn new() -> Self {
        AdvancedAgent
    }
}

impl Agent for AdvancedAgent {
    fn bid_upcard(&mut self, view: &GameView<'_>) -> UpcardBid {
        let up_card = view.up_card;
        let trump = up_card.suit;
        let pos = bidding_position(view.seat);

        // `solo` evaluates the hand as it would be played alone (the partner sits
        // out and never receives the up-card); `team` folds in the up-card going
        // to whichever seat picks it up plus a credit for the partner's help.
        let (solo, team) = if pos == BiddingPosition::Dealer {
            // We are the dealer: we take the up-card and bury our worst card, so
            // evaluate the best five of the resulting six.
            let mut six = view.hand.to_vec();
            six.push(up_card);
            let picked = best_five_tricks(&six, trump);
            (picked, picked)
        } else {
            let solo = expected_tricks(view.hand, trump);
            let up_value = trump_unit(up_card, trump);
            let team = match pos {
                // Our partner (the dealer) pockets the up-card: a real gift.
                BiddingPosition::DealerPartner => solo + 0.4 + 0.4 * up_value,
                // An opposing dealer pockets it: it arms them and denies us.
                _ => solo - (0.35 + 0.4 * up_value),
            };
            (solo, team)
        };

        let adjust = score_adjust(view);
        decide(view.hand, trump, solo + adjust, team + adjust)
    }

    fn bid_call(&mut self, view: &GameView<'_>) -> CallBid {
        let turned_down = view.up_card.suit;
        let on_dealer_team = view.seat.same_team(Seat::Dealer);
        let next = turned_down.same_color();

        // Score every nameable suit, nudged by the next/green convention, and
        // keep the strongest.
        let best = Suit::ALL
            .into_iter()
            .filter(|&s| s != turned_down)
            .map(|s| {
                let convention = if s == next {
                    if on_dealer_team {
                        -NEXT_PENALTY
                    } else {
                        NEXT_BONUS
                    }
                } else if on_dealer_team {
                    GREEN_BONUS
                } else {
                    0.0
                };
                (s, expected_tricks(view.hand, s) + convention)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("three suits remain after the turned-down one");
        let (suit, score) = best;

        let adjust = score_adjust(view);
        // The same hand drives both the alone and with-partner evaluation here;
        // there is no up-card to pick up in the second round.
        let solo = expected_tricks(view.hand, suit) + adjust;
        let team = score + adjust;

        match decide(view.hand, suit, solo, team) {
            UpcardBid::OrderUp { alone } => CallBid::Call { suit, alone },
            UpcardBid::Pass if is_stuck(view) => {
                // Forced to call: take the best suit even though it falls short.
                CallBid::Call { suit, alone: false }
            }
            UpcardBid::Pass => CallBid::Pass,
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
            // All trump: part with the weakest one.
            return *lowest_trump(hand, trump).expect("hand is non-empty");
        }

        let count_in = |suit: Suit| non_trump.iter().filter(|c| c.suit == suit).count();

        // Prefer to void a side suit by dropping a low singleton, freeing a ruff.
        if let Some(&singleton) = non_trump
            .iter()
            .filter(|c| c.rank != Rank::Ace && count_in(c.suit) == 1)
            .min_by_key(|c| c.rank)
        {
            return singleton;
        }

        // Otherwise drop the lowest non-trump card, sparing aces, and biasing
        // toward emptying the shortest suit so a void comes sooner.
        let keep_ace = non_trump.iter().any(|c| c.rank != Rank::Ace);
        *non_trump
            .iter()
            .filter(|c| !keep_ace || c.rank != Rank::Ace)
            .min_by_key(|c| (count_in(c.suit), c.rank as u32))
            .expect("non_trump is non-empty")
    }

    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card {
        let trump = view.trump().expect("trump is set during play");
        let knowledge = Knowledge::from_view(view, trump);
        if view.current_trick.is_empty() {
            lead(view, legal, trump, &knowledge)
        } else {
            follow(view, legal, trump, &knowledge)
        }
    }
}

/// Turns a pair of evaluations into a bid: go alone on an overwhelming hand with
/// top-trump control, otherwise make with a partner when the estimate clears the
/// bar, otherwise pass.
fn decide(hand: &[Card], trump: Suit, solo: f64, team: f64) -> UpcardBid {
    if solo >= ALONE_TRICKS && has_top_control(hand, trump) {
        UpcardBid::OrderUp { alone: true }
    } else if team >= MAKE_TRICKS {
        UpcardBid::OrderUp { alone: false }
    } else {
        UpcardBid::Pass
    }
}

// --- Bidding position --------------------------------------------------------

/// A seat's place in the bidding order, relative to the dealer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BiddingPosition {
    /// Eldest hand, immediately left of the dealer: first to bid and first to
    /// lead in the play.
    Eldest,
    /// The dealer's partner (second to bid). Ordering up gifts the dealer the
    /// up-card.
    DealerPartner,
    /// Third seat, to the dealer's right. Ordering up arms the opposing dealer.
    Third,
    /// The dealer: takes the up-card for free when it is ordered up.
    Dealer,
}

/// Works out where `seat` sits in the bidding order — directly, since seats are
/// already named relative to the dealer.
fn bidding_position(seat: Seat) -> BiddingPosition {
    match seat {
        Seat::First => BiddingPosition::Eldest,
        Seat::Second => BiddingPosition::DealerPartner,
        Seat::Third => BiddingPosition::Third,
        Seat::Dealer => BiddingPosition::Dealer,
    }
}

/// Whether this seat is the dealer forced to name a suit under "stick the
/// dealer", and so may not pass the second round.
fn is_stuck(view: &GameView<'_>) -> bool {
    view.rules.stick_the_dealer && view.seat == Seat::Dealer
}

/// A small swing to the bidding estimate based on the match score: press when
/// the opponents are a hand from winning, relax when we are.
fn score_adjust(view: &GameView<'_>) -> f64 {
    let us = view.scores.us;
    let them = view.scores.them;
    let target = 10; // The view does not carry the target; 10 is conventional.
    let mut adjust = 0.0;
    if them + 1 >= target {
        // Opponents on the hill: contest more so they cannot coast to the win.
        adjust += 0.25;
    }
    if us + 1 >= target {
        // We are on the hill: only take the makes we are confident in.
        adjust -= 0.15;
    }
    adjust
}

// --- Hand evaluation ---------------------------------------------------------

/// Estimated number of tricks `hand` is worth with `trump` as trump, on the
/// `0.0..=5.0` scale.
///
/// The estimate is a sum of contributions: each trump's honour value, a length
/// synergy that rewards extra trumps when the hand also has top control, off-suit
/// aces and guarded kings, and ruffing value for side-suit voids backed by trump.
/// The scale is deliberately calibrated in tricks so the thresholds read as "how
/// many of the five do I expect to take".
fn expected_tricks(hand: &[Card], trump: Suit) -> f64 {
    let mut trumps: Vec<Card> = hand.iter().copied().filter(|c| c.is_trump(trump)).collect();
    trumps.sort_by(|a, b| {
        b.trump_strength(trump, trump)
            .cmp(&a.trump_strength(trump, trump))
    });
    let n = trumps.len();
    let has_right = trumps.first().is_some_and(|c| c.is_right_bower(trump));
    let has_left = trumps.iter().any(|c| c.is_left_bower(trump));

    let mut tricks: f64 = trumps.iter().map(|c| trump_unit(*c, trump)).sum();

    // Length synergy: once you can draw trump, the small ones become winners.
    // It is worth far more when you also hold the top of the suit.
    if n >= 3 {
        let control = if has_right {
            1.0
        } else if has_left {
            0.6
        } else {
            0.3
        };
        tricks += control * 0.35 * (n as f64 - 2.0);
    }
    if has_right && has_left {
        tricks += 0.2;
    }

    // Off-suit honours and voids.
    let mut voids = 0;
    for suit in Suit::ALL {
        if suit == trump {
            continue;
        }
        let in_suit: Vec<Card> = hand
            .iter()
            .copied()
            .filter(|c| !c.is_trump(trump) && c.effective_suit(trump) == suit)
            .collect();
        if in_suit.is_empty() {
            voids += 1;
            continue;
        }
        let len = in_suit.len();
        let has_ace = in_suit.iter().any(|c| c.rank == Rank::Ace);
        let has_king = in_suit.iter().any(|c| c.rank == Rank::King);
        if has_ace {
            tricks += 0.6;
            if len >= 2 && has_king {
                tricks += 0.12;
            }
        } else if has_king && len >= 2 {
            tricks += 0.16;
        }
    }

    // Ruffing only helps if there is trump to ruff with, capped by trump length.
    if n >= 1 && voids > 0 {
        tricks += 0.3 * (voids as f64).min(n as f64);
    }

    tricks.min(5.0)
}

/// The best five-card evaluation among the six cards a dealer holds after taking
/// the up-card, found by trying each possible discard.
fn best_five_tricks(six: &[Card], trump: Suit) -> f64 {
    (0..six.len())
        .map(|drop| {
            let kept: Vec<Card> = six
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != drop)
                .map(|(_, &c)| c)
                .collect();
            expected_tricks(&kept, trump)
        })
        .fold(0.0_f64, f64::max)
}

/// The honour value of a single trump card, in tricks.
fn trump_unit(card: Card, trump: Suit) -> f64 {
    if card.is_right_bower(trump) {
        1.0
    } else if card.is_left_bower(trump) {
        0.85
    } else {
        match card.rank {
            Rank::Ace => 0.65,
            Rank::King => 0.45,
            Rank::Queen => 0.33,
            Rank::Ten => 0.22,
            Rank::Nine => 0.18,
            Rank::Jack => 0.0, // Trump jacks are bowers, handled above.
        }
    }
}

/// Whether the hand holds enough top trump to gamble on going alone: the right
/// bower, or the left bower backed by the ace of trump.
fn has_top_control(hand: &[Card], trump: Suit) -> bool {
    let has_right = hand.iter().any(|c| c.is_right_bower(trump));
    let has_left = hand.iter().any(|c| c.is_left_bower(trump));
    let has_ace = hand
        .iter()
        .any(|c| c.is_trump(trump) && c.rank == Rank::Ace && !c.is_bower(trump));
    has_right || (has_left && has_ace)
}

// --- Card counting -----------------------------------------------------------

/// Everything the agent can infer about the unseen cards from the public record
/// of the hand so far: which cards have been played, and which suits each seat
/// has shown out of.
///
/// It is rebuilt from scratch on every play decision, so it never goes stale.
struct Knowledge {
    trump: Suit,
    /// Indexed by [`card_index`]: whether that card has already been played.
    seen: [bool; 24],
    /// `void[seat][suit]`: whether that seat is known to hold no card of that
    /// effective suit (revealed by failing to follow).
    void: [[bool; 4]; 4],
}

impl Knowledge {
    /// Reconstructs the table knowledge from the agent's [`GameView`].
    fn from_view(view: &GameView<'_>, trump: Suit) -> Self {
        let mut k = Knowledge {
            trump,
            seen: [false; 24],
            void: [[false; 4]; 4],
        };
        for (trick, _winner) in view.completed_tricks {
            k.record_trick(trick);
        }
        k.record_trick(view.current_trick);
        k
    }

    /// Folds one trick's plays into the seen-cards set and void inferences.
    fn record_trick(&mut self, trick: &euchre_interface::Trick) {
        let led = trick.led_suit(self.trump);
        for play in trick.plays() {
            self.seen[card_index(play.card)] = true;
            if let Some(led) = led {
                let effective = play.card.effective_suit(self.trump);
                if effective != led {
                    // Could not follow the led suit, so is void in it.
                    self.void[seat_index(play.seat)][suit_index(led)] = true;
                }
            }
        }
    }

    /// Whether `card` has been played already this hand.
    fn played(&self, card: Card) -> bool {
        self.seen[card_index(card)]
    }

    /// Whether `seat` is known to be void in `suit`.
    fn is_void(&self, seat: Seat, suit: Suit) -> bool {
        self.void[seat_index(seat)][suit_index(suit)]
    }

    /// The highest still-live card of an effective `suit`, excluding the agent's
    /// own `hand`. Cards in the kitty count as live, which only ever makes the
    /// estimate more cautious.
    fn highest_outstanding(&self, suit: Suit, hand: &[Card]) -> Option<Card> {
        let mut best: Option<Card> = None;
        for s in Suit::ALL {
            for r in Rank::ALL {
                let c = Card::new(r, s);
                if c.effective_suit(self.trump) != suit {
                    continue;
                }
                if self.played(c) || hand.contains(&c) {
                    continue;
                }
                let stronger = best.is_none_or(|b| {
                    c.trump_strength(self.trump, suit) > b.trump_strength(self.trump, suit)
                });
                if stronger {
                    best = Some(c);
                }
            }
        }
        best
    }

    /// Whether `card` is the master of its suit: no live card outside the agent's
    /// hand outranks it within that suit. (A side-suit master can still be
    /// *ruffed*; see [`Self::is_unbeatable`].)
    fn is_master(&self, card: Card, hand: &[Card]) -> bool {
        let suit = card.effective_suit(self.trump);
        match self.highest_outstanding(suit, hand) {
            None => true,
            Some(top) => {
                card.trump_strength(self.trump, suit) > top.trump_strength(self.trump, suit)
            }
        }
    }

    /// Whether `card`, played into a trick led with `led`, cannot be beaten by
    /// any live card — accounting for trumps, so a side-suit card is only
    /// unbeatable once no trump can ruff it.
    fn is_unbeatable(&self, card: Card, led: Suit, hand: &[Card]) -> bool {
        let strength = card.trump_strength(self.trump, led);
        for s in Suit::ALL {
            for r in Rank::ALL {
                let c = Card::new(r, s);
                if self.played(c) || hand.contains(&c) {
                    continue;
                }
                if c.trump_strength(self.trump, led) > strength {
                    return false;
                }
            }
        }
        true
    }

    /// How many trumps are still live outside the agent's own `hand`.
    fn trump_outstanding(&self, hand: &[Card]) -> usize {
        let mut count = 0;
        for s in Suit::ALL {
            for r in Rank::ALL {
                let c = Card::new(r, s);
                if c.is_trump(self.trump) && !self.played(c) && !hand.contains(&c) {
                    count += 1;
                }
            }
        }
        count
    }
}

// --- Play: leading -----------------------------------------------------------

/// Chooses a card to lead a fresh trick.
fn lead(view: &GameView<'_>, legal: &[Card], trump: Suit, k: &Knowledge) -> Card {
    let hand = view.hand;
    let we_made = view.contract.is_some_and(|c| view.seat.same_team(c.maker));
    let trumps: Vec<Card> = legal
        .iter()
        .copied()
        .filter(|c| c.is_trump(trump))
        .collect();
    let non_trumps: Vec<Card> = legal
        .iter()
        .copied()
        .filter(|c| !c.is_trump(trump))
        .collect();

    // As the maker with trump length, draw the defenders' trump while they still
    // have some, leading the top so it cannot be beaten.
    if we_made && trumps.len() >= 2 && k.trump_outstanding(hand) > 0 {
        return *highest_trump(&trumps, trump).expect("trumps is non-empty");
    }

    // Cash a guaranteed winner: a side-suit card nothing live can beat (no
    // higher card and no trump left to ruff it). Take the most valuable such.
    if let Some(&winner) = non_trumps
        .iter()
        .filter(|c| k.is_unbeatable(**c, c.effective_suit(trump), hand))
        .max_by_key(|c| c.trump_strength(trump, c.effective_suit(trump)))
    {
        return winner;
    }

    // On defence, cash an off-ace early — before it can be trumped — avoiding a
    // suit an opponent has already shown out of.
    if !we_made
        && let Some(&ace) = non_trumps
            .iter()
            .filter(|c| c.rank == Rank::Ace)
            .find(|c| !opponent_void_in(view, k, c.effective_suit(trump)))
    {
        return ace;
    }

    // Otherwise lead a low non-trump, hoarding trump. Prefer a suit no opponent
    // is void in (so it is not instantly ruffed), then the weakest card.
    if !non_trumps.is_empty() {
        return *non_trumps
            .iter()
            .min_by_key(|c| {
                let suit = c.effective_suit(trump);
                (
                    opponent_void_in(view, k, suit) as u8,
                    c.trump_strength(trump, suit),
                )
            })
            .expect("non_trumps is non-empty");
    }

    // All trump: the maker keeps drawing with the highest; a defender bleeds the
    // lowest.
    if we_made {
        *highest_trump(&trumps, trump).expect("legal is non-empty")
    } else {
        *lowest_trump(&trumps, trump).expect("legal is non-empty")
    }
}

// --- Play: following ---------------------------------------------------------

/// Chooses a card when following into a trick already in progress.
fn follow(view: &GameView<'_>, legal: &[Card], trump: Suit, k: &Knowledge) -> Card {
    let trick = view.current_trick;
    let led = trick.led_suit(trump).expect("trick is non-empty");
    let winning = trick
        .plays()
        .iter()
        .max_by_key(|p| p.card.trump_strength(trump, led))
        .expect("trick is non-empty");
    let win_strength = winning.card.trump_strength(trump, led);
    let hand = view.hand;

    if winning.seat == view.seat.partner() {
        // Partner is winning. If the trick is already theirs — no opponent left
        // to play, or the card cannot be beaten — throw a junk card.
        if opponents_after_me(view) == 0 || k.is_unbeatable(winning.card, led, hand) {
            return slough(legal, trump, led, k, hand);
        }
        // Partner is ahead only on a side card an opponent might overtake or
        // ruff. Lock the trick if we hold a card nothing can beat; otherwise
        // trust the partner rather than waste a high card or overtrump them.
        if !winning.card.is_trump(trump)
            && let Some(&lock) = legal
                .iter()
                .filter(|c| {
                    c.trump_strength(trump, led) > win_strength && k.is_unbeatable(**c, led, hand)
                })
                .min_by_key(|c| c.trump_strength(trump, led))
        {
            return lock;
        }
        return slough(legal, trump, led, k, hand);
    }

    // An opponent is winning: take the trick as cheaply as we can.
    if let Some(&win) = legal
        .iter()
        .filter(|c| c.trump_strength(trump, led) > win_strength)
        .min_by_key(|c| c.trump_strength(trump, led))
    {
        return win;
    }

    // Cannot win: throw the least useful card.
    slough(legal, trump, led, k, hand)
}

/// Picks the card to throw away when not contesting the trick: prefer a low,
/// non-trump, non-master card, and lean toward emptying a short suit (to set up a
/// ruff) when we still hold trump.
fn slough(legal: &[Card], trump: Suit, led: Suit, k: &Knowledge, hand: &[Card]) -> Card {
    let mut pool: Vec<Card> = legal
        .iter()
        .copied()
        .filter(|c| !c.is_trump(trump) && !k.is_master(*c, hand))
        .collect();
    if pool.is_empty() {
        pool = legal
            .iter()
            .copied()
            .filter(|c| !c.is_trump(trump))
            .collect();
    }
    if pool.is_empty() {
        pool = legal.to_vec();
    }

    let have_trump = hand.iter().any(|c| c.is_trump(trump));
    let count_in = |suit: Suit| {
        hand.iter()
            .filter(|c| !c.is_trump(trump) && c.effective_suit(trump) == suit)
            .count()
    };
    *pool
        .iter()
        .min_by_key(|c| {
            let void_bias = if have_trump {
                count_in(c.effective_suit(trump))
            } else {
                0
            };
            (void_bias, c.trump_strength(trump, led))
        })
        .expect("pool is non-empty")
}

/// How many opponents are still to play after this agent in the current trick.
fn opponents_after_me(view: &GameView<'_>) -> usize {
    let sitting = view.contract.and_then(|c| c.sitting_out());
    let total = if view.contract.is_some_and(|c| c.alone) {
        3
    } else {
        4
    };
    let after = total - view.current_trick.len() - 1;
    let mut seat = view.seat;
    let mut found = 0;
    let mut opponents = 0;
    while found < after {
        seat = seat.next();
        if Some(seat) == sitting {
            continue;
        }
        found += 1;
        if !view.seat.same_team(seat) {
            opponents += 1;
        }
    }
    opponents
}

/// Whether either opponent is known to be void in `suit`.
fn opponent_void_in(view: &GameView<'_>, k: &Knowledge, suit: Suit) -> bool {
    Seat::ALL
        .into_iter()
        .filter(|&s| !view.seat.same_team(s))
        .any(|s| k.is_void(s, suit))
}

// --- Small card helpers ------------------------------------------------------

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

/// Maps a card to a stable `0..24` index (suit-major, rank-minor).
fn card_index(card: Card) -> usize {
    suit_index(card.suit) * 6 + card.rank as usize
}

/// Maps a seat to a stable `0..4` index matching [`Seat::ALL`].
fn seat_index(seat: Seat) -> usize {
    match seat {
        Seat::First => 0,
        Seat::Second => 1,
        Seat::Third => 2,
        Seat::Dealer => 3,
    }
}

/// Maps a suit to a stable `0..4` index matching [`Suit::ALL`].
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
    use euchre_interface::{Contract, GameRules, Play, Scores, Trick};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    fn contract(trump: Suit, maker: Seat, alone: bool) -> Contract {
        Contract {
            trump,
            maker,
            alone,
        }
    }

    /// Builds a [`GameView`] from explicit parts; the borrows let each test own
    /// its hand, trick, and history. The dealer is always `Seat::Dealer`.
    fn make_view<'a>(
        seat: Seat,
        up_card: Card,
        hand: &'a [Card],
        contract: Option<Contract>,
        trick: &'a Trick,
        completed: &'a [(Trick, Seat)],
    ) -> GameView<'a> {
        GameView {
            seat,
            up_card,
            hand,
            contract,
            discarded: None,
            current_trick: trick,
            completed_tricks: completed,
            scores: Scores::default(),
            rules: GameRules::default(),
        }
    }

    // --- Hand evaluation -----------------------------------------------------

    #[test]
    fn expected_tricks_ranks_a_monster_over_junk() {
        let monster = [
            card(Rank::Jack, Suit::Spades), // right bower
            card(Rank::Jack, Suit::Clubs),  // left bower
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Diamonds),
        ];
        let junk = [
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        assert!(expected_tricks(&monster, Suit::Spades) > 4.0);
        assert!(expected_tricks(&junk, Suit::Spades) < 1.0);
    }

    #[test]
    fn top_control_needs_a_high_bower() {
        let with_right = [card(Rank::Jack, Suit::Spades)];
        let with_left_and_ace = [card(Rank::Jack, Suit::Clubs), card(Rank::Ace, Suit::Spades)];
        let lone_left = [card(Rank::Jack, Suit::Clubs)];
        assert!(has_top_control(&with_right, Suit::Spades));
        assert!(has_top_control(&with_left_and_ace, Suit::Spades));
        // The left bower without the ace behind it is not enough to go alone.
        assert!(!has_top_control(&lone_left, Suit::Spades));
    }

    // --- Bidding -------------------------------------------------------------

    #[test]
    fn orders_up_a_strong_hand_and_passes_a_bare_one() {
        let mut agent = AdvancedAgent::new();
        let empty = Trick::new();
        let strong = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Jack, Suit::Clubs),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
        ];
        // We sit eldest (First), so the dealer is an opponent and ordering up
        // arms them — yet the hand is strong enough to do it anyway.
        let view = make_view(
            Seat::First,
            card(Rank::Nine, Suit::Spades),
            &strong,
            None,
            &empty,
            &[],
        );
        assert!(matches!(agent.bid_upcard(&view), UpcardBid::OrderUp { .. }));

        let junk = [
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Queen, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        let view = make_view(
            Seat::First,
            card(Rank::Nine, Suit::Spades),
            &junk,
            None,
            &empty,
            &[],
        );
        assert_eq!(agent.bid_upcard(&view), UpcardBid::Pass);
    }

    #[test]
    fn goes_alone_on_a_loner() {
        let mut agent = AdvancedAgent::new();
        let empty = Trick::new();
        // Right, left, ace, king of trump plus an off-ace: a near-certain five.
        let loner = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Jack, Suit::Clubs),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
        ];
        // We are the dealer, so we take the up-card for free.
        let view = make_view(
            Seat::Dealer,
            card(Rank::Nine, Suit::Spades),
            &loner,
            None,
            &empty,
            &[],
        );
        assert_eq!(agent.bid_upcard(&view), UpcardBid::OrderUp { alone: true });
    }

    #[test]
    fn defending_team_calls_next_over_green() {
        let mut agent = AdvancedAgent::new();
        let empty = Trick::new();
        // Balanced in hearts (red) and spades (black). Diamonds is turned down, so
        // hearts is "next" (same colour). We sit eldest (First), on the defending
        // team, and should favour next.
        let hand = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Nine, Suit::Clubs),
        ];
        let view = make_view(
            Seat::First,
            card(Rank::Nine, Suit::Diamonds),
            &hand,
            None,
            &empty,
            &[],
        );
        match agent.bid_call(&view) {
            CallBid::Call { suit, .. } => assert_eq!(suit, Suit::Hearts),
            CallBid::Pass => panic!("expected a call on this hand"),
        }
    }

    #[test]
    fn dealer_team_calls_green_over_next() {
        let mut agent = AdvancedAgent::new();
        let empty = Trick::new();
        // Same balanced hand, but now we are the dealer's partner (Second) and
        // should cross to a black (green) suit rather than call next.
        let hand = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Nine, Suit::Clubs),
        ];
        let view = make_view(
            Seat::Second,
            card(Rank::Nine, Suit::Diamonds),
            &hand,
            None,
            &empty,
            &[],
        );
        match agent.bid_call(&view) {
            CallBid::Call { suit, .. } => assert_eq!(suit.color(), euchre_interface::Color::Black),
            CallBid::Pass => panic!("expected a call on this hand"),
        }
    }

    // --- Discarding ----------------------------------------------------------

    #[test]
    fn discard_voids_a_low_singleton() {
        let mut agent = AdvancedAgent::new();
        let empty = Trick::new();
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Diamonds),
            card(Rank::Queen, Suit::Diamonds),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Spades),
        ];
        let view = make_view(
            Seat::Dealer,
            card(Rank::Nine, Suit::Spades),
            &hand,
            Some(contract(Suit::Spades, Seat::Dealer, false)),
            &empty,
            &[],
        );
        assert_eq!(agent.discard(&view), card(Rank::Nine, Suit::Hearts));
    }

    #[test]
    fn discard_spares_an_ace() {
        let mut agent = AdvancedAgent::new();
        let empty = Trick::new();
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Nine, Suit::Spades),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::King, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
            card(Rank::Ten, Suit::Clubs),
        ];
        let view = make_view(
            Seat::Dealer,
            card(Rank::Nine, Suit::Spades),
            &hand,
            Some(contract(Suit::Spades, Seat::Dealer, false)),
            &empty,
            &[],
        );
        let discard = agent.discard(&view);
        assert_ne!(discard.rank, Rank::Ace);
        assert!(!discard.is_trump(Suit::Spades));
    }

    // --- Play ----------------------------------------------------------------

    #[test]
    fn maker_leads_high_trump_to_draw() {
        let mut agent = AdvancedAgent::new();
        let empty = Trick::new();
        let hand = [
            card(Rank::Jack, Suit::Spades), // right bower
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        let view = make_view(
            Seat::Second,
            card(Rank::Nine, Suit::Spades),
            &hand,
            Some(contract(Suit::Spades, Seat::Second, false)),
            &empty,
            &[],
        );
        assert_eq!(
            agent.play_card(&view, &hand),
            card(Rank::Jack, Suit::Spades)
        );
    }

    #[test]
    fn ducks_when_partner_has_the_trick_locked() {
        let mut agent = AdvancedAgent::new();
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // Partner (First) leads the right bower — unbeatable. Second follows.
        trick.push(Play {
            seat: Seat::First,
            card: card(Rank::Jack, Suit::Spades),
        });
        trick.push(Play {
            seat: Seat::Second,
            card: card(Rank::Nine, Suit::Hearts),
        });
        // We (Third) are void in spades; throw the junk diamond, keep the ace.
        let hand = [
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::Nine, Suit::Diamonds),
        ];
        let view = make_view(
            Seat::Third,
            card(Rank::Nine, Suit::Spades),
            &hand,
            Some(contract(trump, Seat::First, false)),
            &trick,
            &[],
        );
        assert_eq!(
            agent.play_card(&view, &hand),
            card(Rank::Nine, Suit::Diamonds)
        );
    }

    #[test]
    fn third_hand_locks_a_loose_trick_with_a_boss() {
        let mut agent = AdvancedAgent::new();
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // Partner (First) leads the ace of hearts — winning, but a side card a
        // defender behind us could ruff. Second follows low.
        trick.push(Play {
            seat: Seat::First,
            card: card(Rank::Ace, Suit::Hearts),
        });
        trick.push(Play {
            seat: Seat::Second,
            card: card(Rank::Nine, Suit::Hearts),
        });
        // We (Third) are void in hearts and hold the right bower; the Dealer is
        // still to act, so we lock the trick rather than trust a ruffable ace.
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Nine, Suit::Clubs),
        ];
        let view = make_view(
            Seat::Third,
            card(Rank::Nine, Suit::Spades),
            &hand,
            Some(contract(trump, Seat::Second, false)),
            &trick,
            &[],
        );
        assert_eq!(
            agent.play_card(&view, &hand),
            card(Rank::Jack, Suit::Spades)
        );
    }

    #[test]
    fn wins_as_cheaply_as_possible() {
        let mut agent = AdvancedAgent::new();
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // Second leads the king of hearts; we (Third) must follow and hold two
        // winning hearts — take it with the cheaper one is impossible here (only
        // the ace beats the king), so the ace it is, not a trump.
        trick.push(Play {
            seat: Seat::Second,
            card: card(Rank::King, Suit::Hearts),
        });
        let hand = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Nine, Suit::Spades),
        ];
        let legal = [card(Rank::Ace, Suit::Hearts)];
        let view = make_view(
            Seat::Third,
            card(Rank::Nine, Suit::Spades),
            &hand,
            Some(contract(trump, Seat::Second, false)),
            &trick,
            &[],
        );
        assert_eq!(
            agent.play_card(&view, &legal),
            card(Rank::Ace, Suit::Hearts)
        );
    }

    #[test]
    fn sloughs_a_dead_card_and_keeps_a_master() {
        let mut agent = AdvancedAgent::new();
        let trump = Suit::Hearts;
        let mut trick = Trick::new();
        // Second leads a trump we cannot beat; we are void in trump and must throw.
        trick.push(Play {
            seat: Seat::Second,
            card: card(Rank::Ace, Suit::Hearts),
        });
        // The ace of spades is the master of its suit; the nine of diamonds is
        // dead weight. Throw the nine.
        let hand = [
            card(Rank::Ace, Suit::Spades),
            card(Rank::Nine, Suit::Diamonds),
        ];
        let view = make_view(
            Seat::Third,
            card(Rank::Nine, Suit::Spades),
            &hand,
            Some(contract(trump, Seat::Second, false)),
            &trick,
            &[],
        );
        assert_eq!(
            agent.play_card(&view, &hand),
            card(Rank::Nine, Suit::Diamonds)
        );
    }

    // --- Card counting -------------------------------------------------------

    #[test]
    fn knowledge_infers_voids_and_masters() {
        let trump = Suit::Spades;
        // A completed trick: the dealer led a club, Second could not follow and
        // trumped, revealing Second is void in clubs.
        let mut done = Trick::new();
        done.push(Play {
            seat: Seat::Dealer,
            card: card(Rank::King, Suit::Clubs),
        });
        done.push(Play {
            seat: Seat::First,
            card: card(Rank::Ace, Suit::Clubs),
        });
        done.push(Play {
            seat: Seat::Second,
            card: card(Rank::Nine, Suit::Spades),
        });
        done.push(Play {
            seat: Seat::Third,
            card: card(Rank::Ten, Suit::Clubs),
        });
        let completed = [(done, Seat::Second)];
        let empty = Trick::new();
        let hand = [card(Rank::King, Suit::Spades)];
        let view = make_view(
            Seat::Third,
            card(Rank::Nine, Suit::Spades),
            &hand,
            Some(contract(trump, Seat::Dealer, false)),
            &empty,
            &completed,
        );
        let k = Knowledge::from_view(&view, trump);

        assert!(k.is_void(Seat::Second, Suit::Clubs));
        assert!(!k.is_void(Seat::First, Suit::Clubs));
        assert!(k.played(card(Rank::Ace, Suit::Clubs)));
        // The ace of clubs is gone, so our king of clubs would now be the master
        // of that suit (were we still holding clubs).
        assert!(k.is_master(card(Rank::King, Suit::Clubs), &hand));
    }
}
