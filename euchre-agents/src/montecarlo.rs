//! [`MonteCarloAgent`]: a search-based bot that plays by determinized
//! perfect-information sampling (PIMC).
//!
//! Where [`AdvancedAgent`](crate::AdvancedAgent) reasons about a single imagined
//! line, this agent *searches*. At each card it must play it:
//!
//! 1. **Determinizes** the hidden cards — it samples a full, plausible deal of the
//!    unseen cards to the other seats, consistent with everything it has
//!    legitimately observed (its own hand, every card already played, and the
//!    suits each seat has shown void in).
//! 2. **Solves** that sampled world exactly. With all hands face up the remaining
//!    play is a small perfect-information game, so a [`double-dummy`](crate::solver)
//!    alpha-beta search finds the trick count under optimal play by both sides.
//! 3. **Averages** over many such samples and plays the card with the best mean
//!    outcome, scored in real match points (a euchre, a march, and a loner are
//!    worth more than a bare make, exactly as the engine scores them).
//!
//! Averaging over the hidden-card possibilities is the part a heuristic cannot
//! do, and it is what lifts this agent above [`AdvancedAgent`]. The standard
//! caveat of perfect-information sampling applies — within a single determinized
//! world every seat "knows" the layout, so the agent does not model information
//! hiding or value deception — but for out-playing the heuristic agents it is a
//! clear step up.
//!
//! **Bidding and discarding are delegated** to an embedded [`AdvancedAgent`]: its
//! position- and score-aware auction is already strong, so the improvement here
//! is deliberately confined to the play. Folding the search into bidding is a
//! natural future extension.

use euchre_interface::{Agent, CallBid, Card, GameView, Seat, Suit, Team, UpcardBid};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::{IndexedRandom, SliceRandom};

use crate::AdvancedAgent;
use crate::solver::{self, DdState, card_index, seat_index, suit_index};

/// Worlds sampled per play decision in the default configuration. Enough to
/// clearly out-play the heuristic agents while keeping a match fast.
const DEFAULT_DETERMINIZATIONS: usize = 32;

/// How many times to re-roll a void-respecting deal before falling back to an
/// unconstrained one. Generous because a contradiction is essentially impossible
/// with real game data.
const MAX_DEAL_RETRIES: usize = 32;

/// Coefficient of the confidence margin the search must clear before deviating
/// from the [`AdvancedAgent`] fallback's card. The margin scales as
/// `coeff * sqrt(determinizations)`: the sampling noise in the score gap between
/// two cards grows like `sqrt(N)`, so this holds the false-override rate roughly
/// constant — strict enough to ignore a small count's noise, yet loosening in
/// per-world terms as the count (and the search's accuracy) grows.
///
/// The advanced agent's play is already strong, so it is a sound default the
/// search only overrides when *confident*. This keeps the agent robustly at least
/// as strong as its fallback at any search width while preserving the search's
/// real edge.
const OVERRIDE_MARGIN_COEFF: f64 = 0.7;

/// A Perfect-Information Monte Carlo (PIMC) agent.
///
/// It samples [`determinizations`](Self::with_determinizations) full deals at each
/// play and solves each to a double-dummy optimum, playing the card that scores
/// best on average. Construct one with [`MonteCarloAgent::new`] or
/// [`MonteCarloAgent::with_seed`]; tune the search width with
/// [`MonteCarloAgent::with_determinizations`].
#[derive(Debug)]
pub struct MonteCarloAgent {
    /// Delegate for bidding and discarding.
    advanced: AdvancedAgent,
    /// Source of randomness for determinization sampling.
    rng: SmallRng,
    /// Worlds sampled per play decision.
    determinizations: usize,
}

impl MonteCarloAgent {
    /// Creates an agent seeded from system entropy with the default search width.
    pub fn new() -> Self {
        MonteCarloAgent {
            advanced: AdvancedAgent::new(),
            rng: SmallRng::from_rng(&mut rand::rng()),
            determinizations: DEFAULT_DETERMINIZATIONS,
        }
    }

    /// Creates an agent with a fixed seed, for reproducible play.
    pub fn with_seed(seed: u64) -> Self {
        MonteCarloAgent {
            advanced: AdvancedAgent::new(),
            rng: SmallRng::seed_from_u64(seed),
            determinizations: DEFAULT_DETERMINIZATIONS,
        }
    }

    /// Sets the number of determinizations sampled per play (clamped to at least
    /// one). Fewer is faster but noisier; more is stronger but slower. Tests use a
    /// small value to keep the suite quick.
    pub fn with_determinizations(mut self, n: usize) -> Self {
        self.determinizations = n.max(1);
        self
    }

    /// Reconstructs, from the public view, which cards have been played, how many
    /// each seat has played, and which suits each seat has shown void in.
    fn reconstruct(view: &GameView<'_>, trump: Suit) -> ([bool; 24], [usize; 4], [[bool; 4]; 4]) {
        let mut seen = [false; 24];
        let mut played = [0usize; 4];
        let mut void = [[false; 4]; 4];
        let tricks = view
            .completed_tricks
            .iter()
            .map(|(t, _)| t)
            .chain(std::iter::once(view.current_trick));
        for trick in tricks {
            let led = trick.led_suit(trump);
            for play in trick.plays() {
                seen[card_index(play.card)] = true;
                played[seat_index(play.seat)] += 1;
                if let Some(led) = led
                    && play.card.effective_suit(trump) != led
                {
                    void[seat_index(play.seat)][suit_index(led)] = true;
                }
            }
        }
        (seen, played, void)
    }

    /// Samples one full assignment of the unseen cards to the hidden active seats,
    /// consistent with the played cards and revealed voids.
    ///
    /// The agent's own hand is preserved exactly. The seat sitting out a loner and
    /// the buried cards (the up-card, the kitty, and a loner partner's hand) are
    /// simply left undealt — the solver only ever reads the active hands.
    fn determinize(&mut self, view: &GameView<'_>, trump: Suit) -> [Vec<Card>; 4] {
        let me = view.seat;
        let contract = view.contract.expect("a hand in play has a contract");
        let sitting_out = contract.sitting_out();
        let (seen, played, void) = Self::reconstruct(view, trump);

        let pool: Vec<Card> = Card::deck()
            .into_iter()
            .filter(|c| !seen[card_index(*c)] && !view.hand.contains(c))
            .collect();

        let mut need = [0usize; 4];
        for s in Seat::ALL {
            if s == me || Some(s) == sitting_out {
                continue;
            }
            need[seat_index(s)] = 5 - played[seat_index(s)];
        }
        debug_assert_eq!(
            pool.len(),
            need.iter().sum::<usize>() + 4 + if sitting_out.is_some() { 5 } else { 0 },
            "the unseen pool must cover the hidden needs plus the buried cards"
        );

        let mut hands = self
            .deal_constrained(&pool, need, &void, trump)
            .unwrap_or_else(|| self.deal_relaxed(&pool, need));
        hands[seat_index(me)] = view.hand.to_vec();
        hands
    }

    /// Deals the pool to the needy seats respecting voids, retrying with fresh
    /// randomness on a dead end. Returns `None` if every retry fails, leaving the
    /// caller to fall back to an unconstrained deal.
    fn deal_constrained(
        &mut self,
        pool: &[Card],
        need: [usize; 4],
        void: &[[bool; 4]; 4],
        trump: Suit,
    ) -> Option<[Vec<Card>; 4]> {
        for _ in 0..MAX_DEAL_RETRIES {
            if let Some(hands) = self.try_deal_once(pool, need, void, trump) {
                return Some(hands);
            }
        }
        None
    }

    /// One void-respecting deal. Assigns the most-constrained card (eligible for
    /// the fewest needy seats) first, which almost never paints itself into a
    /// corner at Euchre's size; returns `None` if it nonetheless does.
    fn try_deal_once(
        &mut self,
        pool: &[Card],
        mut need: [usize; 4],
        void: &[[bool; 4]; 4],
        trump: Suit,
    ) -> Option<[Vec<Card>; 4]> {
        let mut hands: [Vec<Card>; 4] = std::array::from_fn(|_| Vec::new());
        let mut available = pool.to_vec();
        available.shuffle(&mut self.rng);

        while need.iter().any(|&n| n > 0) {
            // Pick the available card eligible for the fewest needy seats.
            let mut best: Option<(usize, [usize; 4], usize)> = None;
            for (i, &c) in available.iter().enumerate() {
                let suit = suit_index(c.effective_suit(trump));
                let mut eligible = [0usize; 4];
                let mut len = 0;
                for (s, slot) in need.iter().enumerate() {
                    if *slot > 0 && !void[s][suit] {
                        eligible[len] = s;
                        len += 1;
                    }
                }
                if len == 0 {
                    continue; // nobody needy can take it; it stays buried
                }
                if best.is_none_or(|(_, _, blen)| len < blen) {
                    best = Some((i, eligible, len));
                    if len == 1 {
                        break;
                    }
                }
            }

            let (pos, eligible, len) = best?; // a need remains that no card can fill
            let card = available.swap_remove(pos);
            let &seat = eligible[..len]
                .choose(&mut self.rng)
                .expect("eligible is non-empty");
            hands[seat].push(card);
            need[seat] -= 1;
        }
        Some(hands)
    }

    /// Last-resort deal that ignores voids, used only when the constrained deal
    /// cannot satisfy the inferred voids. Perturbs at most one sample of many.
    fn deal_relaxed(&mut self, pool: &[Card], need: [usize; 4]) -> [Vec<Card>; 4] {
        let mut available = pool.to_vec();
        available.shuffle(&mut self.rng);
        let mut hands: [Vec<Card>; 4] = std::array::from_fn(|_| Vec::new());
        let mut cards = available.into_iter();
        for (s, &count) in need.iter().enumerate() {
            for _ in 0..count {
                hands[s].push(cards.next().expect("the pool covers every need"));
            }
        }
        hands
    }
}

impl Default for MonteCarloAgent {
    fn default() -> Self {
        MonteCarloAgent::new()
    }
}

impl Agent for MonteCarloAgent {
    fn bid_upcard(&mut self, view: &GameView<'_>, up_card: Card) -> UpcardBid {
        self.advanced.bid_upcard(view, up_card)
    }

    fn bid_call(&mut self, view: &GameView<'_>, turned_down: Suit) -> CallBid {
        self.advanced.bid_call(view, turned_down)
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        self.advanced.discard(view)
    }

    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card {
        if legal.len() == 1 {
            return legal[0];
        }
        let trump = view.trump().expect("trump is set during play");
        let contract = view.contract.expect("a hand in play has a contract");
        let me = view.seat;
        let maker_team = contract.maker.team();

        // The advanced agent's choice is the default; the search must beat it
        // convincingly to override it.
        let fallback = self.advanced.play_card(view, legal);

        // Score every candidate against the *same* sampled worlds (common random
        // numbers): the comparison is paired, so deal luck cancels and a small
        // sample separates close cards.
        let mut totals = vec![0i32; legal.len()];
        for _ in 0..self.determinizations {
            let world = self.determinize(view, trump);
            let base = DdState::from_world(&world, view, me, trump);
            for (i, &candidate) in legal.iter().enumerate() {
                let maker_tricks = solver::solve(&base.play(candidate));
                totals[i] += my_team_score(maker_tricks, maker_team, contract.alone, me);
            }
        }

        // Find the best-scoring card, then deviate from the fallback only when the
        // search prefers another by at least the confidence margin. Ties keep the
        // fallback, so the agent is never worse than the advanced heuristic.
        let fallback_idx = legal
            .iter()
            .position(|&c| c == fallback)
            .expect("the advanced agent returns a legal card");
        let mut best = fallback_idx;
        for i in 0..legal.len() {
            if totals[i] > totals[best] {
                best = i;
            }
        }
        let margin = (OVERRIDE_MARGIN_COEFF * (self.determinizations as f64).sqrt())
            .round()
            .max(1.0) as i32;
        if best != fallback_idx && totals[best] - totals[fallback_idx] >= margin {
            legal[best]
        } else {
            fallback
        }
    }
}

/// The signed match points the hand is worth to `me`'s team when the makers take
/// `maker_tricks`, replicating the engine's scoring exactly.
fn my_team_score(maker_tricks: u8, maker_team: Team, alone: bool, me: Seat) -> i32 {
    let (scoring_team, points) = if maker_tricks < 3 {
        (maker_team.opponent(), 2) // euchred
    } else if maker_tricks == 5 {
        (maker_team, if alone { 4 } else { 2 }) // march
    } else {
        (maker_team, 1)
    };
    if scoring_team == me.team() {
        points
    } else {
        -points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::{Contract, GameRules, Play, Rank, Scores, Trick};

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

    fn make_view<'a>(
        seat: Seat,
        dealer: Seat,
        hand: &'a [Card],
        contract: Option<Contract>,
        trick: &'a Trick,
        completed: &'a [(Trick, Seat)],
    ) -> GameView<'a> {
        GameView {
            seat,
            dealer,
            hand,
            contract,
            current_trick: trick,
            completed_tricks: completed,
            scores: Scores::default(),
            rules: GameRules::default(),
        }
    }

    /// Every card across the four reconstructed hands is distinct, and my own hand
    /// is preserved.
    fn assert_world_consistent(world: &[Vec<Card>; 4], view: &GameView<'_>) {
        let mut all: Vec<Card> = world.iter().flatten().copied().collect();
        let count = all.len();
        all.sort_by_key(|c| card_index(*c));
        all.dedup();
        assert_eq!(all.len(), count, "determinized hands overlap");
        assert_eq!(world[seat_index(view.seat)], view.hand.to_vec());
    }

    #[test]
    fn determinize_deals_full_hidden_hands_at_the_start_of_play() {
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Clubs),
        ];
        let empty = Trick::new();
        let view = make_view(
            Seat::North,
            Seat::West,
            &hand,
            Some(contract(Suit::Spades, Seat::East, false)),
            &empty,
            &[],
        );
        let mut agent = MonteCarloAgent::with_seed(1);
        for _ in 0..50 {
            let world = agent.determinize(&view, Suit::Spades);
            for s in [Seat::East, Seat::South, Seat::West] {
                assert_eq!(world[seat_index(s)].len(), 5);
            }
            assert_world_consistent(&world, &view);
        }
    }

    #[test]
    fn determinize_respects_a_revealed_void() {
        // A completed trick: North led the nine of hearts, East ruffed with the
        // right bower (so East is void in hearts), South and West followed.
        let mut done = Trick::new();
        for play in [
            (Seat::North, card(Rank::Nine, Suit::Hearts)),
            (Seat::East, card(Rank::Jack, Suit::Spades)),
            (Seat::South, card(Rank::Ten, Suit::Hearts)),
            (Seat::West, card(Rank::King, Suit::Hearts)),
        ] {
            done.push(Play {
                seat: play.0,
                card: play.1,
            });
        }
        let completed = [(done, Seat::East)];
        // North's four remaining cards, none of them hearts (those are in play).
        let hand = [
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Ace, Suit::Clubs),
            card(Rank::Nine, Suit::Diamonds),
        ];
        let empty = Trick::new();
        let view = make_view(
            Seat::North,
            Seat::West,
            &hand,
            Some(contract(Suit::Spades, Seat::East, false)),
            &empty,
            &completed,
        );
        let mut agent = MonteCarloAgent::with_seed(7);
        for _ in 0..200 {
            let world = agent.determinize(&view, Suit::Spades);
            for c in &world[seat_index(Seat::East)] {
                assert_ne!(
                    c.effective_suit(Suit::Spades),
                    Suit::Hearts,
                    "East was dealt a heart despite being void"
                );
            }
            assert_world_consistent(&world, &view);
        }
    }

    #[test]
    fn determinize_leaves_a_loner_partner_out() {
        let hand = [
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
            card(Rank::Ace, Suit::Clubs),
        ];
        let empty = Trick::new();
        // East goes alone; East's partner West sits out.
        let view = make_view(
            Seat::North,
            Seat::West,
            &hand,
            Some(contract(Suit::Spades, Seat::East, true)),
            &empty,
            &[],
        );
        let mut agent = MonteCarloAgent::with_seed(3);
        for _ in 0..50 {
            let world = agent.determinize(&view, Suit::Spades);
            assert!(world[seat_index(Seat::West)].is_empty());
            assert_eq!(world[seat_index(Seat::East)].len(), 5);
            assert_eq!(world[seat_index(Seat::South)].len(), 5);
            assert_world_consistent(&world, &view);
        }
    }

    #[test]
    fn a_single_legal_card_skips_the_search() {
        let only = card(Rank::Nine, Suit::Clubs);
        let hand = [only];
        let empty = Trick::new();
        let view = make_view(
            Seat::North,
            Seat::West,
            &hand,
            Some(contract(Suit::Spades, Seat::East, false)),
            &empty,
            &[],
        );
        // A huge determinization count would be ruinous if it were consulted.
        let mut agent = MonteCarloAgent::with_seed(1).with_determinizations(1_000_000);
        assert_eq!(agent.play_card(&view, &hand), only);
    }

    #[test]
    fn play_card_returns_a_legal_card() {
        // A full five-card hand leading the first trick: a consistent position
        // (nothing played yet), with the whole hand legal so the search runs.
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::King, Suit::Clubs),
            card(Rank::Ten, Suit::Spades),
        ];
        let empty = Trick::new();
        let view = make_view(
            Seat::South,
            Seat::North,
            &hand,
            Some(contract(Suit::Spades, Seat::South, false)),
            &empty,
            &[],
        );
        let mut agent = MonteCarloAgent::with_seed(99).with_determinizations(8);
        let chosen = agent.play_card(&view, &hand);
        assert!(hand.contains(&chosen));
    }

    #[test]
    fn my_team_score_matches_engine_scoring() {
        // Maker is North/South; viewed from North (a maker) and East (a defender).
        assert_eq!(my_team_score(5, Team::NorthSouth, true, Seat::North), 4); // alone march
        assert_eq!(my_team_score(5, Team::NorthSouth, false, Seat::North), 2); // march
        assert_eq!(my_team_score(3, Team::NorthSouth, false, Seat::North), 1); // bare make
        assert_eq!(my_team_score(2, Team::NorthSouth, false, Seat::North), -2); // euchred
        assert_eq!(my_team_score(2, Team::NorthSouth, false, Seat::East), 2); // euchred, defender
    }
}
