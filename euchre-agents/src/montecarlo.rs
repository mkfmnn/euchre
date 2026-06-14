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
//! **Bidding is anchored PIMC.** The embedded [`AdvancedAgent`] picks the suit and
//! the default bid (preserving conventions the double-dummy search cannot see),
//! and the search only adjusts it when confident: it tunes alone-versus-partner
//! (a loner has no partner to mis-model, so its value is trustworthy), vetoes a
//! make whose simulated value is clearly losing, and orders up a hand the
//! heuristic passed when the search is sure it profits. **Discarding stays
//! delegated.** The search can be turned off with [`MonteCarloAgent::play_only`].

use euchre_interface::{Agent, Bid, CallBid, Card, GameView, Rank, Seat, Suit, Team, UpcardBid};
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

/// Worlds sampled per *bidding* decision. A make integrates the whole five-trick
/// play, so the per-world score is noisier than a single card's; a few more
/// samples tighten the mean enough for the margins below to read real signal.
const DEFAULT_BID_DETERMINIZATIONS: usize = 48;

/// Mean-match-point edge (over the sampled worlds) required to switch a make
/// between alone and with-partner. Small, because both options are makes.
const LONER_MARGIN: f64 = 0.30;

/// Mean-match-point loss below which a make the advanced agent wanted is vetoed
/// down to a pass. Larger than `LONER_MARGIN`: overriding a call is higher-stakes.
const VETO_MARGIN: f64 = 0.50;

/// Mean-match-point gain above which a pass the advanced agent chose is upgraded
/// to a make. The largest margin: inventing a make the heuristic declined is the
/// riskiest deviation, so the search must be clearly confident.
const UPGRADE_MARGIN: f64 = 0.50;

/// Optimism correction subtracted from the *with-partner* contract value before
/// any comparison. The double-dummy solver assumes the partner plays perfectly
/// with full knowledge, so it overvalues makes that need partner help; the alone
/// value is immune (the partner sits out) and takes no haircut.
const HAIRCUT: f64 = 0.40;

/// A Perfect-Information Monte Carlo (PIMC) agent.
///
/// It samples [`determinizations`](Self::with_determinizations) full deals at each
/// play and solves each to a double-dummy optimum, playing the card that scores
/// best on average. Construct one with [`MonteCarloAgent::new`] or
/// [`MonteCarloAgent::with_seed`]; tune the search width with
/// [`MonteCarloAgent::with_determinizations`].
#[derive(Debug)]
pub struct MonteCarloAgent {
    /// Delegate for the default bid and for discarding.
    advanced: AdvancedAgent,
    /// Source of randomness for determinization sampling.
    rng: SmallRng,
    /// Worlds sampled per play decision.
    determinizations: usize,
    /// Worlds sampled per bidding decision.
    bid_determinizations: usize,
    /// Whether to search the bidding; when false, bidding is delegated wholesale.
    bid_search: bool,
}

impl MonteCarloAgent {
    /// Creates an agent seeded from system entropy with the default search width.
    pub fn new() -> Self {
        MonteCarloAgent {
            advanced: AdvancedAgent::new(),
            rng: SmallRng::from_rng(&mut rand::rng()),
            determinizations: DEFAULT_DETERMINIZATIONS,
            bid_determinizations: DEFAULT_BID_DETERMINIZATIONS,
            bid_search: true,
        }
    }

    /// Creates an agent with a fixed seed, for reproducible play.
    pub fn with_seed(seed: u64) -> Self {
        MonteCarloAgent {
            advanced: AdvancedAgent::new(),
            rng: SmallRng::seed_from_u64(seed),
            determinizations: DEFAULT_DETERMINIZATIONS,
            bid_determinizations: DEFAULT_BID_DETERMINIZATIONS,
            bid_search: true,
        }
    }

    /// Sets the number of determinizations sampled per play (clamped to at least
    /// one). Fewer is faster but noisier; more is stronger but slower. Tests use a
    /// small value to keep the suite quick.
    pub fn with_determinizations(mut self, n: usize) -> Self {
        self.determinizations = n.max(1);
        self
    }

    /// Sets the number of determinizations sampled per bidding decision (clamped
    /// to at least one).
    pub fn with_bid_determinizations(mut self, n: usize) -> Self {
        self.bid_determinizations = n.max(1);
        self
    }

    /// Disables the bidding search, delegating every bid to the advanced agent.
    /// Useful for isolating the value of search in the play.
    pub fn play_only(mut self) -> Self {
        self.bid_search = false;
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

    /// Samples one full deal for evaluating a candidate contract during bidding:
    /// my five cards stay fixed, the other playing seats get five each from the
    /// unseen pool, and (in round one) the dealer takes the up-card and buries its
    /// weakest. No trick has been played, so there are no voids to respect.
    /// Returns the hands and the seat that leads the play.
    fn determinize_bid(
        &mut self,
        view: &GameView<'_>,
        trump: Suit,
        alone: bool,
        up_card: Option<Card>,
    ) -> ([Vec<Card>; 4], Seat) {
        let me = view.seat;
        let dealer = view.dealer;
        let sitting_out = alone.then(|| me.partner());

        let mut pool: Vec<Card> = Card::deck()
            .into_iter()
            .filter(|c| !view.hand.contains(c) && Some(*c) != up_card)
            .collect();
        pool.shuffle(&mut self.rng);

        let mut hands: [Vec<Card>; 4] = std::array::from_fn(|_| Vec::new());
        hands[seat_index(me)] = view.hand.to_vec();
        let mut cards = pool.into_iter();
        for s in Seat::ALL {
            if s == me || Some(s) == sitting_out {
                continue;
            }
            for _ in 0..5 {
                hands[seat_index(s)].push(cards.next().expect("the pool covers the deal"));
            }
        }

        // Round one: the dealer picks up the up-card and discards, unless the
        // dealer is the seat sitting out a loner (then there is no pickup).
        if let Some(up) = up_card
            && Some(dealer) != sitting_out
        {
            let d = seat_index(dealer);
            hands[d].push(up);
            let buried = modeled_discard(&hands[d], trump);
            hands[d].retain(|&c| c != buried);
        }

        (hands, first_leader(dealer, sitting_out))
    }

    /// The mean signed match points my team scores by making `trump` (alone or
    /// with a partner), estimated by determinized double-dummy sampling.
    fn contract_ev(
        &mut self,
        view: &GameView<'_>,
        trump: Suit,
        alone: bool,
        up_card: Option<Card>,
    ) -> f64 {
        let me = view.seat;
        let maker_team = me.team();
        let sitting_out = alone.then(|| me.partner());
        let mut total = 0i32;
        for _ in 0..self.bid_determinizations {
            let (hands, leader) = self.determinize_bid(view, trump, alone, up_card);
            let st = DdState::new_play(&hands, trump, sitting_out, maker_team, leader);
            total += my_team_score(solver::solve(&st), maker_team, alone, me);
        }
        total as f64 / self.bid_determinizations as f64
    }

    /// Refines a make the advanced agent wants (in `trump`, defaulting to
    /// `default_bid`) using search: veto it down to a pass when its value is
    /// clearly losing (only if `can_pass`), otherwise pick alone versus partner by
    /// estimated value. Returns `None` to pass, `Some(bid)` to make.
    fn refine_make(
        &mut self,
        view: &GameView<'_>,
        trump: Suit,
        default_bid: Bid,
        up_card: Option<Card>,
        can_pass: bool,
    ) -> Option<Bid> {
        // The with-partner value is optimistic (double-dummy trusts the partner);
        // the alone value is not, so only the former takes the haircut.
        let ev_partner = self.contract_ev(view, trump, false, up_card) - HAIRCUT;
        let ev_alone = self.contract_ev(view, trump, true, up_card);

        if can_pass && ev_partner.max(ev_alone) < -VETO_MARGIN {
            return None;
        }
        let bid = if ev_alone >= ev_partner + LONER_MARGIN {
            Bid::Alone
        } else if ev_partner >= ev_alone + LONER_MARGIN {
            Bid::WithPartner
        } else {
            default_bid
        };
        Some(bid)
    }
}

impl Default for MonteCarloAgent {
    fn default() -> Self {
        MonteCarloAgent::new()
    }
}

impl Agent for MonteCarloAgent {
    fn bid_upcard(&mut self, view: &GameView<'_>, up_card: Card) -> UpcardBid {
        let base = self.advanced.bid_upcard(view, up_card);
        if !self.bid_search {
            return base;
        }
        let trump = up_card.suit;
        match base {
            // Refine a make: veto a clear loser (round-one passing is always safe)
            // or retune alone-vs-partner.
            UpcardBid::OrderUp(bid) => {
                match self.refine_make(view, trump, bid, Some(up_card), true) {
                    Some(b) => UpcardBid::OrderUp(b),
                    None => UpcardBid::Pass,
                }
            }
            // Upgrade a pass to a make when the hand has enough trump to plausibly
            // profit (a cheap filter that avoids searching hopeless passes) and the
            // search is clearly confident.
            UpcardBid::Pass => {
                let mine = trump_count(view.hand, trump) + usize::from(view.seat == view.dealer); // I would take the up-card
                if mine < 2 {
                    return UpcardBid::Pass;
                }
                let ev_partner = self.contract_ev(view, trump, false, Some(up_card)) - HAIRCUT;
                let ev_alone = self.contract_ev(view, trump, true, Some(up_card));
                if ev_partner.max(ev_alone) < UPGRADE_MARGIN {
                    UpcardBid::Pass
                } else if ev_alone >= ev_partner + LONER_MARGIN {
                    UpcardBid::OrderUp(Bid::Alone)
                } else {
                    UpcardBid::OrderUp(Bid::WithPartner)
                }
            }
        }
    }

    fn bid_call(&mut self, view: &GameView<'_>, turned_down: Suit) -> CallBid {
        let base = self.advanced.bid_call(view, turned_down);
        match base {
            CallBid::Call { suit, bid } if self.bid_search => {
                // "Stick the dealer" forbids a forced dealer from passing.
                let can_pass = !(view.rules.stick_the_dealer && view.seat == view.dealer);
                match self.refine_make(view, suit, bid, None, can_pass) {
                    Some(b) => CallBid::Call { suit, bid: b },
                    None => CallBid::Pass,
                }
            }
            other => other,
        }
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

/// How many of `hand`'s cards are trump (including the left bower).
fn trump_count(hand: &[Card], trump: Suit) -> usize {
    hand.iter().filter(|c| c.is_trump(trump)).count()
}

/// The seat that leads the first trick: the dealer's left, skipping a seat
/// sitting out a loner. Mirrors the engine's leader rule.
fn first_leader(dealer: Seat, sitting_out: Option<Seat>) -> Seat {
    let candidate = dealer.next();
    if Some(candidate) == sitting_out {
        candidate.next()
    } else {
        candidate
    }
}

/// Picks which card a dealer buries after taking the up-card into a six-card hand:
/// shed the weakest off-trump card, sparing aces and leaning toward voiding a
/// short side suit; if the hand is all trump, drop the lowest trump. A coarse
/// model of [`AdvancedAgent::discard`], good enough for bid-time sampling.
fn modeled_discard(hand: &[Card], trump: Suit) -> Card {
    let non_trump: Vec<Card> = hand
        .iter()
        .copied()
        .filter(|c| !c.is_trump(trump))
        .collect();
    if non_trump.is_empty() {
        return *hand
            .iter()
            .min_by_key(|c| c.trump_strength(trump, trump))
            .expect("hand is non-empty");
    }
    let count_in = |suit: Suit| {
        non_trump
            .iter()
            .filter(|c| c.effective_suit(trump) == suit)
            .count()
    };
    let keep_ace = non_trump.iter().any(|c| c.rank != Rank::Ace);
    *non_trump
        .iter()
        .filter(|c| !keep_ace || c.rank != Rank::Ace)
        .min_by_key(|c| {
            let suit = c.effective_suit(trump);
            (count_in(suit), c.trump_strength(trump, suit))
        })
        .expect("non_trump is non-empty")
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

    // --- Bidding -------------------------------------------------------------

    fn five() -> [Card; 5] {
        [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Clubs),
        ]
    }

    #[test]
    fn determinize_bid_round1_deals_a_full_table() {
        let hand = five();
        let empty = Trick::new();
        let view = make_view(Seat::North, Seat::West, &hand, None, &empty, &[]);
        let up = card(Rank::Queen, Suit::Spades); // not in hand
        let mut agent = MonteCarloAgent::with_seed(5);
        for _ in 0..50 {
            let (world, leader) = agent.determinize_bid(&view, Suit::Spades, false, Some(up));
            for s in [Seat::East, Seat::South, Seat::West] {
                assert_eq!(world[seat_index(s)].len(), 5);
            }
            assert_world_consistent(&world, &view);
            assert_eq!(leader, Seat::North); // dealer West's left
        }
    }

    #[test]
    fn determinize_bid_alone_sits_the_partner_out() {
        let hand = five();
        let empty = Trick::new();
        // East deals; North goes alone, so South (North's partner) sits out and the
        // lead, normally South, skips to West.
        let view = make_view(Seat::North, Seat::East, &hand, None, &empty, &[]);
        let up = card(Rank::Queen, Suit::Spades);
        let mut agent = MonteCarloAgent::with_seed(6);
        for _ in 0..50 {
            let (world, leader) = agent.determinize_bid(&view, Suit::Spades, true, Some(up));
            assert!(world[seat_index(Seat::South)].is_empty());
            assert_eq!(world[seat_index(Seat::East)].len(), 5);
            assert_eq!(world[seat_index(Seat::West)].len(), 5);
            assert_world_consistent(&world, &view);
            assert_eq!(leader, Seat::West);
        }
    }

    #[test]
    fn determinize_bid_round2_has_no_pickup() {
        let hand = five();
        let empty = Trick::new();
        let view = make_view(Seat::North, Seat::West, &hand, None, &empty, &[]);
        let mut agent = MonteCarloAgent::with_seed(7);
        let (world, leader) = agent.determinize_bid(&view, Suit::Hearts, false, None);
        for s in [Seat::East, Seat::South, Seat::West] {
            assert_eq!(world[seat_index(s)].len(), 5);
        }
        assert_world_consistent(&world, &view);
        assert_eq!(leader, Seat::North);
    }

    #[test]
    fn passes_a_hopeless_upcard() {
        // No trump and nothing of value: below the pre-filter, so it never even
        // searches — and certainly does not order up.
        let junk = [
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Queen, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        let empty = Trick::new();
        let view = make_view(Seat::North, Seat::West, &junk, None, &empty, &[]);
        let mut agent = MonteCarloAgent::with_seed(1).with_bid_determinizations(8);
        assert_eq!(
            agent.bid_upcard(&view, card(Rank::Nine, Suit::Spades)),
            UpcardBid::Pass
        );
    }

    #[test]
    fn orders_up_a_monster() {
        let monster = [
            card(Rank::Jack, Suit::Spades), // right bower
            card(Rank::Jack, Suit::Clubs),  // left bower
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
        ];
        let empty = Trick::new();
        let view = make_view(Seat::North, Seat::West, &monster, None, &empty, &[]);
        let mut agent = MonteCarloAgent::with_seed(2).with_bid_determinizations(8);
        assert!(matches!(
            agent.bid_upcard(&view, card(Rank::Nine, Suit::Spades)),
            UpcardBid::OrderUp(_)
        ));
    }

    #[test]
    fn bid_call_never_passes_a_stuck_dealer() {
        // Junk hand, but the dealer is stuck: the veto must not fire (the engine
        // forbids a pass), so a suit is always named.
        let junk = [
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ten, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Queen, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        let empty = Trick::new();
        let view = GameView {
            seat: Seat::North,
            dealer: Seat::North,
            hand: &junk,
            contract: None,
            current_trick: &empty,
            completed_tricks: &[],
            scores: Scores::default(),
            rules: GameRules {
                stick_the_dealer: true,
            },
        };
        let mut agent = MonteCarloAgent::with_seed(3).with_bid_determinizations(8);
        assert!(matches!(
            agent.bid_call(&view, Suit::Spades),
            CallBid::Call { .. }
        ));
    }
}
