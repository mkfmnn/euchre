//! A fast, tactically sharper Euchre agent.
//!
//! [`OpenAiAdvancedAgent`] is still a lightweight bot: bidding and most play
//! decisions are deterministic evaluations, with a tiny bounded rollout only in
//! late-hand spots where the remaining tree is small enough to sample cheaply.

use euchre_interface::{
    Agent, CallBid, Card, Contract, GameView, Play, Rank, Seat, Suit, Trick, UpcardBid,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Flip this to `false` to disable every rollout and use only deterministic
/// tactical evaluation.
const ENABLE_ROLLOUTS: bool = true;
const MAX_ROLLOUTS_PER_DECISION: usize = 8;

const MAKE_THRESHOLD: i32 = 88;
const CALL_THRESHOLD: i32 = 82;
const NEXT_CALL_BONUS: i32 = 10;
const ALONE_THRESHOLD: i32 = 148;

/// A stronger, bounded-compute Euchre agent.
#[derive(Debug, Clone)]
pub struct OpenAiAdvancedAgent {
    rng: SmallRng,
}

impl OpenAiAdvancedAgent {
    /// Creates an agent seeded from system entropy.
    pub fn new() -> Self {
        OpenAiAdvancedAgent {
            rng: SmallRng::from_rng(&mut rand::rng()),
        }
    }

    /// Creates an agent with a fixed seed, for reproducible evaluations.
    pub fn with_seed(seed: u64) -> Self {
        OpenAiAdvancedAgent {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    fn is_stuck(view: &GameView<'_>) -> bool {
        view.rules.stick_the_dealer && view.seat == Seat::Dealer
    }

    fn upcard_score(view: &GameView<'_>, trump: Suit) -> i32 {
        let me = view.seat;
        if me == Seat::Dealer {
            return best_keep_score_with_optional_upcard(view.hand, Some(view.up_card), trump);
        }

        let base = hand_strength(view.hand, trump);
        let up = trump_card_value(view.up_card, trump);
        if me.partner() == Seat::Dealer {
            base + up / 2 + 8
        } else {
            base - up / 3 - 8
        }
    }

    fn call_score(view: &GameView<'_>, suit: Suit) -> i32 {
        let mut score = hand_strength(view.hand, suit);
        if suit == view.up_card.suit.same_color() {
            score += NEXT_CALL_BONUS;
        }
        if view.seat == Seat::Dealer {
            score += 4;
        }
        score
    }

    fn should_go_alone(view: &GameView<'_>, strength: i32) -> bool {
        if strength < ALONE_THRESHOLD {
            return false;
        }
        // A normal point wins the match; avoid converting strong make hands into
        // unnecessary high-variance loners.
        view.scores.us + 1 < 10 || strength >= ALONE_THRESHOLD + 18
    }

    fn deterministic_play(view: &GameView<'_>, legal: &[Card]) -> Card {
        let trump = view.trump().expect("trump is set during play");
        let contract = view.contract.expect("contract is set during play");

        legal
            .iter()
            .copied()
            .max_by_key(|&card| tactical_play_score(view, legal, card, trump, contract))
            .expect("legal is non-empty")
    }

    fn rollout_play(&mut self, view: &GameView<'_>, legal: &[Card]) -> Option<Card> {
        if !ENABLE_ROLLOUTS || legal.len() <= 1 || view.completed_tricks.len() < 3 {
            return None;
        }

        let trump = view.trump()?;
        let contract = view.contract?;
        let mut best = None;
        let mut best_score = i32::MIN;
        for &card in legal {
            let tactical = tactical_play_score(view, legal, card, trump, contract);
            let rollout = rollout_score(view, card, &mut self.rng);
            let score = tactical + rollout;
            if score > best_score
                || (score == best_score
                    && card_sort_key(card) < best.map(card_sort_key).unwrap_or(u32::MAX))
            {
                best_score = score;
                best = Some(card);
            }
        }
        best
    }
}

impl Default for OpenAiAdvancedAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Agent for OpenAiAdvancedAgent {
    fn bid_upcard(&mut self, view: &GameView<'_>) -> UpcardBid {
        let trump = view.up_card.suit;
        let mut strength = Self::upcard_score(view, trump);
        if view.seat == Seat::Dealer || view.seat.partner() == Seat::Dealer {
            strength += 4;
        }
        let threshold = if view.seat == Seat::Dealer {
            MAKE_THRESHOLD - 12
        } else if view.seat.partner() == Seat::Dealer {
            MAKE_THRESHOLD - 6
        } else {
            MAKE_THRESHOLD + 7
        };

        if Self::should_go_alone(view, strength) {
            UpcardBid::OrderUp { alone: true }
        } else if strength >= threshold {
            UpcardBid::OrderUp { alone: false }
        } else {
            UpcardBid::Pass
        }
    }

    fn bid_call(&mut self, view: &GameView<'_>) -> CallBid {
        let turned_down = view.up_card.suit;
        let (suit, strength) = Suit::ALL
            .into_iter()
            .filter(|&s| s != turned_down)
            .map(|s| (s, Self::call_score(view, s)))
            .max_by_key(|&(s, score)| (score, suit_tiebreak(s, turned_down)))
            .expect("three suits are callable");

        let threshold = if view.seat == Seat::Dealer {
            CALL_THRESHOLD - 8
        } else if suit == turned_down.same_color() {
            CALL_THRESHOLD - 5
        } else {
            CALL_THRESHOLD
        };

        if Self::should_go_alone(view, strength) {
            CallBid::Call { suit, alone: true }
        } else if strength >= threshold || Self::is_stuck(view) {
            CallBid::Call { suit, alone: false }
        } else {
            CallBid::Pass
        }
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        let trump = view.trump().expect("trump is set before discard");
        choose_discard(view.hand, trump)
    }

    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card {
        self.rollout_play(view, legal)
            .unwrap_or_else(|| Self::deterministic_play(view, legal))
    }
}

fn hand_strength(cards: &[Card], trump: Suit) -> i32 {
    let mut score = 0;
    let mut trump_count = 0;
    let mut suit_counts = [0i32; 4];

    for &card in cards {
        if card.is_trump(trump) {
            trump_count += 1;
            score += trump_card_value(card, trump);
        } else {
            score += off_card_value(card);
        }
        suit_counts[suit_index(card.effective_suit(trump))] += 1;
    }

    if trump_count >= 1 {
        for suit in Suit::ALL {
            if suit != trump && suit_counts[suit_index(suit)] == 0 {
                score += 5 + 4 * trump_count.min(3);
            }
        }
    }
    if trump_count >= 3 {
        score += 8;
    }
    if has_right(cards, trump) && has_left(cards, trump) {
        score += 14;
    }
    score
}

fn trump_card_value(card: Card, trump: Suit) -> i32 {
    if card.is_right_bower(trump) {
        44
    } else if card.is_left_bower(trump) {
        38
    } else {
        match card.rank {
            Rank::Ace => 30,
            Rank::King => 24,
            Rank::Queen => 18,
            Rank::Jack => 0,
            Rank::Ten => 13,
            Rank::Nine => 10,
        }
    }
}

fn off_card_value(card: Card) -> i32 {
    match card.rank {
        Rank::Ace => 20,
        Rank::King => 10,
        Rank::Queen => 5,
        Rank::Jack => 3,
        Rank::Ten => 1,
        Rank::Nine => 0,
    }
}

fn best_keep_score_with_optional_upcard(hand: &[Card], up_card: Option<Card>, trump: Suit) -> i32 {
    let mut cards = hand.to_vec();
    if let Some(up_card) = up_card {
        cards.push(up_card);
    }
    if cards.len() <= 5 {
        return hand_strength(&cards, trump);
    }

    cards
        .iter()
        .copied()
        .map(|discard| {
            let kept: Vec<Card> = cards.iter().copied().filter(|&c| c != discard).collect();
            hand_strength(&kept, trump) + discard_tiebreak(discard, &kept, trump)
        })
        .max()
        .expect("cards is non-empty")
}

fn choose_discard(hand: &[Card], trump: Suit) -> Card {
    hand.iter()
        .copied()
        .min_by_key(|&discard| {
            let kept: Vec<Card> = hand.iter().copied().filter(|&c| c != discard).collect();
            -hand_strength(&kept, trump) - discard_tiebreak(discard, &kept, trump)
        })
        .expect("hand is non-empty")
}

fn discard_tiebreak(discard: Card, kept: &[Card], trump: Suit) -> i32 {
    let mut score = 0;
    let mut counts = [0; 4];
    for &card in kept {
        counts[suit_index(card.effective_suit(trump))] += 1;
    }
    if !discard.is_trump(trump) && counts[suit_index(discard.effective_suit(trump))] == 0 {
        score += 6;
    }
    if discard.rank == Rank::Ace && !discard.is_trump(trump) {
        score -= 8;
    }
    score
}

fn tactical_play_score(
    view: &GameView<'_>,
    legal: &[Card],
    card: Card,
    trump: Suit,
    contract: Contract,
) -> i32 {
    if view.current_trick.is_empty() {
        return lead_score(view, legal, card, trump, contract);
    }

    let led = view
        .current_trick
        .led_suit(trump)
        .expect("non-empty trick has led suit");
    let winning = current_winner(view.current_trick, trump).expect("non-empty trick has winner");
    let winning_strength = winning.card.trump_strength(trump, led);
    let strength = card.trump_strength(trump, led);
    let partner_winning = winning.seat == view.seat.partner();
    let last_to_play = view.current_trick.len() + 1 == active_count(contract);
    let wins_now = strength > winning_strength;

    if partner_winning {
        let mut score = -keep_value(card, trump);
        if last_to_play && card.rank == Rank::Ace && !card.is_trump(trump) {
            score -= 18;
        }
        return score;
    }

    if wins_now {
        let mut score = 180 - keep_value(card, trump);
        if last_to_play {
            score += 55;
        }
        if view.seat.same_team(contract.maker) {
            score += 8;
        }
        score
    } else {
        -keep_value(card, trump) + if card.is_trump(trump) { -20 } else { 0 }
    }
}

fn lead_score(
    view: &GameView<'_>,
    legal: &[Card],
    card: Card,
    trump: Suit,
    contract: Contract,
) -> i32 {
    let we_made = view.seat.same_team(contract.maker);
    let trump_count = legal.iter().filter(|c| c.is_trump(trump)).count();
    let mut score = 0;

    if card.is_trump(trump) {
        score += card.trump_strength(trump, trump) as i32 / 8;
        if we_made && trump_count >= 2 {
            score += 65;
        } else if trump_count == 1 {
            score -= 24;
        }
    } else {
        score += match card.rank {
            Rank::Ace => 70,
            Rank::King => 24,
            Rank::Queen => 8,
            _ => 0,
        };
        if !we_made {
            score += 10;
        }
        score -= keep_value(card, trump) / 4;
    }

    if is_boss_among_visible(view, card, trump) {
        score += 20;
    }
    if card.is_right_bower(trump) && we_made {
        score += 30;
    }
    score
}

fn keep_value(card: Card, trump: Suit) -> i32 {
    if card.is_trump(trump) {
        trump_card_value(card, trump) + 30
    } else {
        off_card_value(card) + if card.rank == Rank::Ace { 20 } else { 0 }
    }
}

fn current_winner(trick: &Trick, trump: Suit) -> Option<Play> {
    let led = trick.led_suit(trump)?;
    trick
        .plays()
        .iter()
        .copied()
        .max_by_key(|p| p.card.trump_strength(trump, led))
}

fn rollout_score(view: &GameView<'_>, candidate: Card, rng: &mut SmallRng) -> i32 {
    let mut total = 0;
    let mut samples = 0;
    for _ in 0..MAX_ROLLOUTS_PER_DECISION {
        if let Some(mut sim) = SimState::from_view(view, candidate, rng) {
            total += sim.play_out();
            samples += 1;
        }
    }
    if samples == 0 { 0 } else { total / samples }
}

#[derive(Clone)]
struct SimState {
    perspective: Seat,
    contract: Contract,
    hands: [Vec<Card>; 4],
    current_trick: Trick,
    completed_winners: Vec<Seat>,
    turn: Seat,
}

impl SimState {
    fn from_view(view: &GameView<'_>, candidate: Card, rng: &mut SmallRng) -> Option<Self> {
        let contract = view.contract?;
        let trump = contract.trump;
        let mut current_trick = view.current_trick.clone();
        current_trick.push(Play {
            seat: view.seat,
            card: candidate,
        });

        let mut hands: [Vec<Card>; 4] = std::array::from_fn(|_| Vec::new());
        hands[seat_index_rel(view.seat)] = view
            .hand
            .iter()
            .copied()
            .filter(|&c| c != candidate)
            .collect();

        let mut needed = [0usize; 4];
        for seat in Seat::ALL {
            let idx = seat_index_rel(seat);
            if contract.sitting_out() == Some(seat) {
                needed[idx] = 0;
            } else if seat == view.seat {
                needed[idx] = hands[idx].len();
            } else {
                let already_in_current = current_trick.plays().iter().any(|p| p.seat == seat);
                needed[idx] = 5usize
                    .saturating_sub(view.completed_tricks.len())
                    .saturating_sub(usize::from(already_in_current));
            }
        }

        let mut unknown = Card::deck();
        remove_cards(&mut unknown, view.hand);
        for (trick, _) in view.completed_tricks {
            for play in trick.plays() {
                remove_card(&mut unknown, play.card);
            }
        }
        for play in view.current_trick.plays() {
            remove_card(&mut unknown, play.card);
        }
        if let Some(discarded) = view.discarded {
            remove_card(&mut unknown, discarded);
        }
        if contract.trump != view.up_card.suit {
            remove_card(&mut unknown, view.up_card);
        }

        let voids = known_voids(view, &current_trick, trump);
        for seat in Seat::ALL {
            if seat == view.seat {
                continue;
            }
            let idx = seat_index_rel(seat);
            for _ in 0..needed[idx] {
                let card = draw_plausible(&mut unknown, &voids[idx], trump, rng)?;
                hands[idx].push(card);
            }
        }

        let completed_winners = view
            .completed_tricks
            .iter()
            .map(|(_, winner)| *winner)
            .collect();
        let active = active_count(contract);
        let turn = if current_trick.len() == active {
            current_winner(&current_trick, trump)?.seat
        } else {
            next_active(view.seat, contract)
        };

        Some(SimState {
            perspective: view.seat,
            contract,
            hands,
            current_trick,
            completed_winners,
            turn,
        })
    }

    fn play_out(&mut self) -> i32 {
        while self.completed_winners.len() < 5 {
            if self.current_trick.len() == active_count(self.contract) {
                self.resolve_trick();
                continue;
            }
            let idx = seat_index_rel(self.turn);
            let legal = legal_cards(&self.hands[idx], &self.current_trick, self.contract.trump);
            if legal.is_empty() {
                self.resolve_trick();
                continue;
            }
            let card = rollout_policy(self.turn, &legal, &self.current_trick, self.contract);
            remove_card(&mut self.hands[idx], card);
            self.current_trick.push(Play {
                seat: self.turn,
                card,
            });
            if self.current_trick.len() == active_count(self.contract) {
                self.resolve_trick();
            } else {
                self.turn = next_active(self.turn, self.contract);
            }
        }
        hand_utility(self.perspective, self.contract, &self.completed_winners)
    }

    fn resolve_trick(&mut self) {
        let winner = current_winner(&self.current_trick, self.contract.trump)
            .expect("completed trick has a winner")
            .seat;
        self.completed_winners.push(winner);
        self.current_trick = Trick::new();
        self.turn = winner;
    }
}

fn rollout_policy(seat: Seat, legal: &[Card], trick: &Trick, contract: Contract) -> Card {
    let trump = contract.trump;
    if trick.is_empty() {
        return legal
            .iter()
            .copied()
            .max_by_key(|&c| {
                let mut score = if c.is_trump(trump) {
                    c.trump_strength(trump, trump) as i32 / 8
                } else {
                    off_card_value(c)
                };
                if c.rank == Rank::Ace && !c.is_trump(trump) {
                    score += 40;
                }
                if seat.same_team(contract.maker) && c.is_trump(trump) {
                    score += 25;
                }
                score
            })
            .expect("legal is non-empty");
    }

    let led = trick.led_suit(trump).expect("non-empty trick");
    let winning = current_winner(trick, trump).expect("non-empty trick");
    if winning.seat == seat.partner() {
        return weakest_card(legal, trump, led);
    }
    let winning_strength = winning.card.trump_strength(trump, led);
    legal
        .iter()
        .copied()
        .filter(|c| c.trump_strength(trump, led) > winning_strength)
        .min_by_key(|c| c.trump_strength(trump, led))
        .unwrap_or_else(|| weakest_card(legal, trump, led))
}

fn legal_cards(hand: &[Card], trick: &Trick, trump: Suit) -> Vec<Card> {
    let Some(led) = trick.led_suit(trump) else {
        return hand.to_vec();
    };
    let following: Vec<Card> = hand
        .iter()
        .copied()
        .filter(|c| c.effective_suit(trump) == led)
        .collect();
    if following.is_empty() {
        hand.to_vec()
    } else {
        following
    }
}

fn hand_utility(perspective: Seat, contract: Contract, winners: &[Seat]) -> i32 {
    let maker_tricks = winners
        .iter()
        .filter(|&&seat| seat.same_team(contract.maker))
        .count();
    let us_tricks = winners
        .iter()
        .filter(|&&seat| seat.same_team(perspective))
        .count() as i32;
    let them_tricks = winners.len() as i32 - us_tricks;
    let makers_us = perspective.same_team(contract.maker);

    let points = if maker_tricks < 3 {
        if makers_us { -220 } else { 220 }
    } else if maker_tricks == 5 {
        let value = if contract.alone { 420 } else { 240 };
        if makers_us { value } else { -value }
    } else if makers_us {
        120
    } else {
        -120
    };
    points + 12 * (us_tricks - them_tricks)
}

fn known_voids(
    view: &GameView<'_>,
    current_after_candidate: &Trick,
    trump: Suit,
) -> [[bool; 4]; 4] {
    let mut voids = [[false; 4]; 4];
    for (trick, _) in view.completed_tricks {
        mark_voids(&mut voids, trick, trump);
    }
    mark_voids(&mut voids, current_after_candidate, trump);
    voids
}

fn mark_voids(voids: &mut [[bool; 4]; 4], trick: &Trick, trump: Suit) {
    let Some(led) = trick.led_suit(trump) else {
        return;
    };
    for play in trick.plays().iter().skip(1) {
        if play.card.effective_suit(trump) != led {
            voids[seat_index_rel(play.seat)][suit_index(led)] = true;
        }
    }
}

fn draw_plausible(
    cards: &mut Vec<Card>,
    voids: &[bool; 4],
    trump: Suit,
    rng: &mut SmallRng,
) -> Option<Card> {
    if cards.is_empty() {
        return None;
    }
    let valid: Vec<usize> = cards
        .iter()
        .enumerate()
        .filter_map(|(idx, card)| (!voids[suit_index(card.effective_suit(trump))]).then_some(idx))
        .collect();
    let idx = if valid.is_empty() {
        rng.random_range(0..cards.len())
    } else {
        valid[rng.random_range(0..valid.len())]
    };
    Some(cards.swap_remove(idx))
}

fn is_boss_among_visible(view: &GameView<'_>, card: Card, trump: Suit) -> bool {
    if !card.is_trump(trump) && card.rank != Rank::Ace {
        return false;
    }
    let led = card.effective_suit(trump);
    let strength = card.trump_strength(trump, led);
    visible_cards(view)
        .into_iter()
        .filter(|&c| c != card)
        .filter(|&c| c.effective_suit(trump) == led || c.is_trump(trump))
        .all(|c| c.trump_strength(trump, led) < strength)
}

fn visible_cards(view: &GameView<'_>) -> Vec<Card> {
    let mut cards = view.hand.to_vec();
    for (trick, _) in view.completed_tricks {
        for play in trick.plays() {
            cards.push(play.card);
        }
    }
    for play in view.current_trick.plays() {
        cards.push(play.card);
    }
    if let Some(discarded) = view.discarded {
        cards.push(discarded);
    }
    cards
}

fn weakest_card(cards: &[Card], trump: Suit, led: Suit) -> Card {
    cards
        .iter()
        .copied()
        .min_by_key(|c| (c.trump_strength(trump, led), card_sort_key(*c)))
        .expect("cards is non-empty")
}

fn remove_cards(cards: &mut Vec<Card>, remove: &[Card]) {
    for &card in remove {
        remove_card(cards, card);
    }
}

fn remove_card(cards: &mut Vec<Card>, card: Card) -> bool {
    if let Some(pos) = cards.iter().position(|&c| c == card) {
        cards.swap_remove(pos);
        true
    } else {
        false
    }
}

fn has_right(cards: &[Card], trump: Suit) -> bool {
    cards.iter().any(|c| c.is_right_bower(trump))
}

fn has_left(cards: &[Card], trump: Suit) -> bool {
    cards.iter().any(|c| c.is_left_bower(trump))
}

fn active_count(contract: Contract) -> usize {
    if contract.sitting_out().is_some() {
        3
    } else {
        4
    }
}

fn next_active(seat: Seat, contract: Contract) -> Seat {
    let candidate = seat.next();
    if contract.sitting_out() == Some(candidate) {
        candidate.next()
    } else {
        candidate
    }
}

fn seat_index_rel(seat: Seat) -> usize {
    match seat {
        Seat::First => 0,
        Seat::Second => 1,
        Seat::Third => 2,
        Seat::Dealer => 3,
    }
}

fn suit_index(suit: Suit) -> usize {
    match suit {
        Suit::Clubs => 0,
        Suit::Diamonds => 1,
        Suit::Hearts => 2,
        Suit::Spades => 3,
    }
}

fn suit_tiebreak(suit: Suit, turned_down: Suit) -> i32 {
    if suit == turned_down.same_color() {
        3
    } else {
        suit_index(suit) as i32
    }
}

fn card_sort_key(card: Card) -> u32 {
    (suit_index(card.suit) as u32) * 8 + card.rank as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::{Contract, GameRules, Play, Scores};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    fn view_for<'a>(
        hand: &'a [Card],
        trick: &'a Trick,
        seat: Seat,
        contract: Option<Contract>,
    ) -> GameView<'a> {
        GameView {
            seat,
            up_card: card(Rank::Nine, Suit::Spades),
            hand,
            contract,
            discarded: None,
            current_trick: trick,
            completed_tricks: &[],
            scores: Scores::default(),
            rules: GameRules::default(),
        }
    }

    #[test]
    fn stuck_dealer_calls_instead_of_passing() {
        let hand = [
            card(Rank::Nine, Suit::Clubs),
            card(Rank::Ten, Suit::Diamonds),
            card(Rank::Queen, Suit::Hearts),
        ];
        let trick = Trick::new();
        let mut view = view_for(&hand, &trick, Seat::Dealer, None);
        view.rules = GameRules {
            stick_the_dealer: true,
        };
        let mut agent = OpenAiAdvancedAgent::with_seed(1);
        assert!(matches!(agent.bid_call(&view), CallBid::Call { .. }));
    }

    #[test]
    fn discard_preserves_bowers_and_creates_void() {
        let trump = Suit::Spades;
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Jack, Suit::Clubs),
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::King, Suit::Hearts),
        ];
        let discard = choose_discard(&hand, trump);
        assert_eq!(discard, card(Rank::Nine, Suit::Diamonds));
    }

    #[test]
    fn follows_by_winning_cheaply() {
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        trick.push(Play {
            seat: Seat::Second,
            card: card(Rank::King, Suit::Hearts),
        });
        let hand = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Nine, Suit::Hearts),
        ];
        let legal = hand;
        let contract = Contract {
            trump,
            maker: Seat::Second,
            alone: false,
        };
        let view = view_for(&hand, &trick, Seat::Third, Some(contract));
        let mut agent = OpenAiAdvancedAgent::with_seed(2);
        assert_eq!(
            agent.play_card(&view, &legal),
            card(Rank::Ace, Suit::Hearts)
        );
    }

    #[test]
    fn ducks_when_partner_is_winning() {
        let trump = Suit::Spades;
        let mut trick = Trick::new();
        trick.push(Play {
            seat: Seat::First,
            card: card(Rank::Jack, Suit::Spades),
        });
        trick.push(Play {
            seat: Seat::Second,
            card: card(Rank::Ace, Suit::Hearts),
        });
        let hand = [
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::Nine, Suit::Diamonds),
        ];
        let contract = Contract {
            trump,
            maker: Seat::First,
            alone: false,
        };
        let view = view_for(&hand, &trick, Seat::Third, Some(contract));
        let mut agent = OpenAiAdvancedAgent::with_seed(3);
        assert_eq!(
            agent.play_card(&view, &hand),
            card(Rank::Nine, Suit::Diamonds)
        );
    }

    #[test]
    fn leads_boss_off_ace() {
        let trump = Suit::Spades;
        let hand = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Clubs),
        ];
        let trick = Trick::new();
        let contract = Contract {
            trump,
            maker: Seat::Dealer,
            alone: false,
        };
        let view = view_for(&hand, &trick, Seat::First, Some(contract));
        let mut agent = OpenAiAdvancedAgent::with_seed(4);
        assert_eq!(agent.play_card(&view, &hand), card(Rank::Ace, Suit::Hearts));
    }
}
