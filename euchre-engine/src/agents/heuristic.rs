//! A competent rule-based agent.
//!
//! [`HeuristicAgent`] plays a sensible game without any search: it scores its
//! hand against a candidate trump to decide whether to make (and whether to go
//! alone), discards to create a void when it picks up the up-card, and follows
//! a small set of trick-play rules — lead your winners, win cheaply, save your
//! trump, and never trump your partner. It is no expert, but it punishes loose
//! play and gives a human a genuine game.

use euchre_interface::{Agent, Bid, CallBid, Card, GameView, Rank, Seat, Suit, Trick, UpcardBid};

/// A rule-based Euchre agent. See the [module docs](self) for its strategy.
///
/// The agent is stateless between hands and fully deterministic: the same view
/// always yields the same decision, which makes its behavior easy to test and
/// to reason about.
#[derive(Debug, Clone, Default)]
pub struct HeuristicAgent;

impl HeuristicAgent {
    /// Creates a heuristic agent.
    pub fn new() -> Self {
        HeuristicAgent
    }
}

// ---- Hand evaluation ---------------------------------------------------------

/// A rough strength value for a single card if `trump` is trump. The scale is
/// arbitrary but internally consistent; bowers and trump dominate, off-aces are
/// worth holding, and low off-cards are near worthless.
fn card_strength(card: Card, trump: Suit) -> u32 {
    if card.is_right_bower(trump) {
        return 100;
    }
    if card.is_left_bower(trump) {
        return 90;
    }
    if card.is_trump(trump) {
        return match card.rank {
            Rank::Ace => 70,
            Rank::King => 58,
            Rank::Queen => 48,
            Rank::Ten => 38,
            Rank::Nine => 33,
            Rank::Jack => unreachable!("trump jacks are bowers"),
        };
    }
    match card.rank {
        Rank::Ace => 30,
        Rank::King => 12,
        Rank::Queen => 8,
        Rank::Jack => 5,
        Rank::Ten => 3,
        Rank::Nine => 2,
    }
}

/// Scores a five-card hand for a candidate `trump`, rewarding raw card power
/// plus the trumping potential of being void in side suits.
fn evaluate_hand(hand: &[Card], trump: Suit) -> u32 {
    let raw: u32 = hand.iter().map(|&c| card_strength(c, trump)).sum();
    let trump_count = hand.iter().filter(|c| c.is_trump(trump)).count();

    // A void side suit lets you trump in — but only worth something if you have
    // trump left to do it with.
    let mut void_bonus = 0;
    if trump_count >= 2 {
        for suit in Suit::ALL {
            if suit == trump {
                continue;
            }
            let has_suit = hand.iter().any(|c| c.effective_suit(trump) == suit);
            if !has_suit {
                void_bonus += 14;
            }
        }
    }
    raw + void_bonus
}

/// Returns the best five cards (by strength) once `up_card` is added to a hand,
/// and that hand's evaluation. Used by the dealer to judge a round-one order-up,
/// since the dealer would pick the up-card up.
fn evaluate_with_pickup(hand: &[Card], up_card: Card, trump: Suit) -> u32 {
    let mut six: Vec<Card> = hand.to_vec();
    six.push(up_card);
    // Drop the single weakest card, mirroring the discard the dealer would make.
    if let Some((idx, _)) = six
        .iter()
        .enumerate()
        .min_by_key(|&(_, c)| card_strength(*c, trump))
    {
        six.remove(idx);
    }
    evaluate_hand(&six, trump)
}

// Thresholds, tuned so the agent makes on roughly average-or-better hands and
// reserves going alone for genuinely commanding ones.
const MAKE_THRESHOLD: u32 = 175;
const ALONE_THRESHOLD: u32 = 290;

impl HeuristicAgent {
    /// Decides whether the evaluated strength justifies going alone.
    fn alone_bid(score: u32) -> Bid {
        if score >= ALONE_THRESHOLD {
            Bid::Alone
        } else {
            Bid::WithPartner
        }
    }
}

impl Agent for HeuristicAgent {
    fn bid_upcard(&mut self, view: &GameView<'_>, up_card: Card) -> UpcardBid {
        let trump = up_card.suit;
        let dealer_is_partner_or_self = view.dealer.team() == view.seat.team();

        // Strength of our hand if this suit were trump. The dealer evaluates as
        // if the up-card were already picked up.
        let score = if view.seat == view.dealer {
            evaluate_with_pickup(view.hand, up_card, trump)
        } else {
            evaluate_hand(view.hand, trump)
        };

        // The up-card is going to the dealer. If that dealer is an opponent,
        // we are handing them a known trump, so demand a stronger hand; if the
        // up-card is itself powerful (a bower), demand more still.
        let mut threshold = MAKE_THRESHOLD;
        if !dealer_is_partner_or_self {
            threshold += 30;
            if up_card.rank == Rank::Jack {
                threshold += 25;
            }
        }

        if score >= threshold {
            UpcardBid::OrderUp(Self::alone_bid(score))
        } else {
            UpcardBid::Pass
        }
    }

    fn bid_call(&mut self, view: &GameView<'_>, turned_down: Suit) -> CallBid {
        // Find the strongest suit we could name (never the turned-down one).
        let best = Suit::ALL
            .into_iter()
            .filter(|&s| s != turned_down)
            .map(|s| (s, evaluate_hand(view.hand, s)))
            .max_by_key(|&(_, score)| score);

        let Some((suit, score)) = best else {
            return CallBid::Pass;
        };

        // As dealer we are last to speak; passing would throw the hand in (or
        // be overridden under "stick the dealer"), so always name our best.
        let forced = view.seat == view.dealer;

        if forced || score >= MAKE_THRESHOLD + 10 {
            CallBid::Call {
                suit,
                bid: Self::alone_bid(score),
            }
        } else {
            CallBid::Pass
        }
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        let trump = view.trump().expect("trump is set before discarding");
        // Prefer to void a side suit: a lone low card in an otherwise-unheld
        // suit is the ideal discard because it frees us to trump later.
        let singleton_void = view
            .hand
            .iter()
            .copied()
            .filter(|c| !c.is_trump(trump))
            .filter(|&c| {
                let suit = c.effective_suit(trump);
                view.hand
                    .iter()
                    .filter(|o| o.effective_suit(trump) == suit)
                    .count()
                    == 1
            })
            .min_by_key(|&c| card_strength(c, trump));
        if let Some(card) = singleton_void {
            return card;
        }
        // Otherwise just throw the weakest card overall.
        view.hand
            .iter()
            .copied()
            .min_by_key(|&c| card_strength(c, trump))
            .expect("hand is non-empty")
    }

    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card {
        let trump = view.trump().expect("trump is set during play");
        let trick = view.current_trick;

        if trick.is_empty() {
            choose_lead(view, legal, trump)
        } else {
            choose_follow(view, legal, trump, trick)
        }
    }
}

// ---- Trick play --------------------------------------------------------------

/// Picks a card to lead with.
fn choose_lead(view: &GameView<'_>, legal: &[Card], trump: Suit) -> Card {
    let we_made = view
        .contract
        .map(|c| c.maker.team() == view.seat.team())
        .unwrap_or(false);
    let trump_count = legal.iter().filter(|c| c.is_trump(trump)).count();

    // Lead an off-suit ace if we have one: it usually wins the trick outright.
    if let Some(ace) = legal
        .iter()
        .copied()
        .find(|c| !c.is_trump(trump) && c.rank == Rank::Ace)
    {
        return ace;
    }

    // As the making team, lead a high trump early to strip the opponents'
    // trump while we still hold the top of the suit.
    if we_made
        && trump_count >= 2
        && let Some(top) = legal
            .iter()
            .copied()
            .filter(|c| c.is_trump(trump))
            .max_by_key(|&c| c.trump_strength(trump, trump))
    {
        return top;
    }

    // Otherwise lead our lowest non-trump to conserve trump; if all we hold is
    // trump, lead the lowest of it.
    lowest_preferring_nontrump(legal, trump)
}

/// Picks a card to follow with, given the trick in progress.
fn choose_follow(view: &GameView<'_>, legal: &[Card], trump: Suit, trick: &Trick) -> Card {
    let led = trick
        .led_suit(trump)
        .expect("a non-empty trick has a led suit");
    let me = view.seat;

    let (winning_seat, winning_card) = current_winner(trick, trump);
    let partner_winning = winning_seat.team() == me.team();

    // Cards we hold that would take the trick as it currently stands.
    let mut winners: Vec<Card> = legal
        .iter()
        .copied()
        .filter(|c| c.trump_strength(trump, led) > winning_card.trump_strength(trump, led))
        .collect();

    let can_follow = legal.iter().all(|c| c.effective_suit(trump) == led);

    if partner_winning {
        // Partner has it; don't overtake. Throw our least useful card —
        // following suit low if we must follow, else shedding a side card.
        return if can_follow {
            lowest(legal, trump, led)
        } else {
            lowest_preferring_nontrump(legal, trump)
        };
    }

    // An opponent is winning. Take it if we can, as cheaply as possible.
    if !winners.is_empty() {
        winners.sort_by_key(|c| c.trump_strength(trump, led));
        // When following suit, win cheaply. When trumping in (we're void),
        // also use the smallest trump that does the job.
        return winners[0];
    }

    // We cannot beat the current winner: follow suit low, or shed our lowest
    // off-suit card, keeping trump for later.
    if can_follow {
        lowest(legal, trump, led)
    } else {
        lowest_preferring_nontrump(legal, trump)
    }
}

/// The seat and card currently winning the in-progress trick.
fn current_winner(trick: &Trick, trump: Suit) -> (Seat, Card) {
    let led = trick
        .led_suit(trump)
        .expect("a non-empty trick has a led suit");
    let best = trick
        .plays()
        .iter()
        .max_by_key(|p| p.card.trump_strength(trump, led))
        .expect("a non-empty trick has plays");
    (best.seat, best.card)
}

/// The lowest card by trick strength under the given led suit.
fn lowest(cards: &[Card], trump: Suit, led: Suit) -> Card {
    cards
        .iter()
        .copied()
        .min_by_key(|&c| c.trump_strength(trump, led))
        .expect("non-empty")
}

/// The lowest card, preferring to shed a non-trump before spending trump.
///
/// Trump cards are pushed to the back of the ordering so they are only chosen
/// when nothing else is available.
fn lowest_preferring_nontrump(cards: &[Card], trump: Suit) -> Card {
    cards
        .iter()
        .copied()
        .min_by_key(|&c| {
            let base = card_strength(c, trump);
            // Bias: spend non-trump first by giving trump a large offset.
            if c.is_trump(trump) { base + 1000 } else { base }
        })
        .expect("non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::rng::Rng;
    use euchre_interface::Play;

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn strong_hand_orders_up() {
        let mut agent = HeuristicAgent::new();
        // Right bower, left bower, ace of trump, plus two off cards.
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Jack, Suit::Clubs),
            card(Rank::Ace, Suit::Spades),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Diamonds),
        ];
        let trick = Trick::new();
        let view = GameView {
            seat: Seat::South,
            dealer: Seat::North, // partner deals; pickup helps us
            hand: &hand,
            contract: None,
            current_trick: &trick,
            completed_tricks: &[],
            scores: Default::default(),
        };
        let up = card(Rank::King, Suit::Spades);
        assert!(matches!(agent.bid_upcard(&view, up), UpcardBid::OrderUp(_)));
    }

    #[test]
    fn weak_hand_passes() {
        let mut agent = HeuristicAgent::new();
        let hand = [
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Queen, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        let trick = Trick::new();
        let view = GameView {
            seat: Seat::East,
            dealer: Seat::North,
            hand: &hand,
            contract: None,
            current_trick: &trick,
            completed_tricks: &[],
            scores: Default::default(),
        };
        // Up-card spades would not be our trump; weak everywhere.
        let up = card(Rank::Nine, Suit::Spades);
        assert_eq!(agent.bid_upcard(&view, up), UpcardBid::Pass);
    }

    #[test]
    fn loaded_hand_goes_alone() {
        let mut agent = HeuristicAgent::new();
        // Both bowers, ace, king, queen of trump: a guaranteed march.
        let hand = [
            card(Rank::Jack, Suit::Hearts),
            card(Rank::Jack, Suit::Diamonds),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
            card(Rank::Queen, Suit::Hearts),
        ];
        let trick = Trick::new();
        let view = GameView {
            seat: Seat::North,
            dealer: Seat::North,
            hand: &hand,
            contract: None,
            current_trick: &trick,
            completed_tricks: &[],
            scores: Default::default(),
        };
        let up = card(Rank::Ten, Suit::Hearts);
        assert_eq!(agent.bid_upcard(&view, up), UpcardBid::OrderUp(Bid::Alone));
    }

    #[test]
    fn discards_to_make_a_void() {
        let mut agent = HeuristicAgent::new();
        // Trump = spades. A singleton nine of diamonds should be shed to void
        // diamonds, even though the nine of hearts is also low.
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Hearts),
            card(Rank::Queen, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Spades),
        ];
        let trick = Trick::new();
        let contract = euchre_interface::Contract {
            trump: Suit::Spades,
            maker: Seat::North,
            alone: false,
        };
        let view = GameView {
            seat: Seat::North,
            dealer: Seat::North,
            hand: &hand,
            contract: Some(contract),
            current_trick: &trick,
            completed_tricks: &[],
            scores: Default::default(),
        };
        assert_eq!(agent.discard(&view), card(Rank::Nine, Suit::Diamonds));
    }

    #[test]
    fn wins_cheaply_when_opponent_leads() {
        let mut agent = HeuristicAgent::new();
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // West (opponent) leads the king of hearts.
        trick.push(Play {
            seat: Seat::West,
            card: card(Rank::King, Suit::Hearts),
        });
        let hand = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Queen, Suit::Hearts),
            card(Rank::Nine, Suit::Hearts),
        ];
        let contract = euchre_interface::Contract {
            trump,
            maker: Seat::North,
            alone: false,
        };
        let view = GameView {
            seat: Seat::North,
            dealer: Seat::South,
            hand: &hand,
            contract: Some(contract),
            current_trick: &trick,
            completed_tricks: &[],
            scores: Default::default(),
        };
        // Must follow hearts; the ace is the only winner, so it should be played.
        let legal = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Queen, Suit::Hearts),
            card(Rank::Nine, Suit::Hearts),
        ];
        assert_eq!(
            agent.play_card(&view, &legal),
            card(Rank::Ace, Suit::Hearts)
        );
    }

    #[test]
    fn does_not_overtake_partner() {
        let mut agent = HeuristicAgent::new();
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        // Partner (South) leads the ace of hearts — already winning.
        trick.push(Play {
            seat: Seat::South,
            card: card(Rank::Ace, Suit::Hearts),
        });
        // Opponent West plays a low heart.
        trick.push(Play {
            seat: Seat::West,
            card: card(Rank::Nine, Suit::Hearts),
        });
        let hand = [
            card(Rank::King, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
        ];
        let contract = euchre_interface::Contract {
            trump,
            maker: Seat::South,
            alone: false,
        };
        let view = GameView {
            seat: Seat::North,
            dealer: Seat::East,
            hand: &hand,
            contract: Some(contract),
            current_trick: &trick,
            completed_tricks: &[],
            scores: Default::default(),
        };
        let legal = [
            card(Rank::King, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
        ];
        // Partner is winning, so we duck with the ten rather than burning the king.
        assert_eq!(
            agent.play_card(&view, &legal),
            card(Rank::Ten, Suit::Hearts)
        );
    }

    #[test]
    fn heuristic_beats_random_handily() {
        use crate::agents::RandomAgent;
        // North/South are heuristic; East/West are random. Heuristic should win
        // the clear majority of matches.
        let mut ns_wins = 0;
        let total = 80;
        for seed in 0..total {
            let agents: [Box<dyn Agent>; 4] = [
                Box::new(HeuristicAgent::new()),
                Box::new(RandomAgent::new(Rng::seed_from_u64(seed * 4 + 1))),
                Box::new(HeuristicAgent::new()),
                Box::new(RandomAgent::new(Rng::seed_from_u64(seed * 4 + 3))),
            ];
            let config = EngineConfig {
                seed: Some(seed),
                ..Default::default()
            };
            let mut engine = Engine::new(agents, config);
            if engine.play_match().winner == euchre_interface::Team::NorthSouth {
                ns_wins += 1;
            }
        }
        // A strong margin; rule-based play should dominate random play.
        assert!(
            ns_wins as f64 / total as f64 > 0.85,
            "heuristic won only {ns_wins}/{total}"
        );
    }
}
