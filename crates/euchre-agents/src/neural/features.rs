//! Turning a [`GameView`] into the fixed-width feature vectors the policy
//! networks consume, and mapping network outputs back to legal actions.
//!
//! ## Trump-relative encoding
//!
//! Every card is encoded by a *trump-relative slot* rather than its printed
//! identity (see [`card_slot`]). The 24 slots run: right bower, left bower, the
//! five remaining trumps (A K Q T 9), the five cards of the same-colour suit
//! (its jack is the left bower, already counted), then the six cards of each of
//! the two off-colour suits. Encoding relative to trump means the network learns
//! the value of, say, "the right bower" or "the off-suit ace" exactly once
//! instead of separately for each of the four suits it could be — the single
//! biggest lever on how much a small net can learn from a given amount of data.
//! It also makes the left-bower-is-trump rule a property of the *encoding*, so
//! the net never has to rediscover it.
//!
//! Suits, likewise, are numbered relative to trump: 0 = trump, 1 = "next" (the
//! same colour as trump), 2 and 3 = the two off-colour suits in suit order.
//!
//! ## Heads
//!
//! Each of the four decisions is a separate classifier ([`Head`]). Bidding heads
//! choose among a handful of abstract actions; the card-playing heads
//! ([`Head::Discard`], [`Head::Play`]) score all 24 slots and the agent takes the
//! best *legal* card. Legality is expressed as a bitmask so the same masking
//! works for training (masked softmax) and inference (masked arg-max).

use euchre_interface::{Bid, CallBid, Card, GameView, Rank, Seat, Suit, UpcardBid};

/// The four decision points, each backed by its own policy network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head {
    /// First-round bid on the up-card.
    Upcard,
    /// Second-round bid naming a suit.
    Call,
    /// The dealer's discard after picking up the up-card.
    Discard,
    /// A card played to a trick.
    Play,
}

impl Head {
    /// All heads in a stable order, matching their serialization order.
    pub const ALL: [Head; 4] = [Head::Upcard, Head::Call, Head::Discard, Head::Play];

    /// A stable index into per-head arrays.
    pub fn index(self) -> usize {
        match self {
            Head::Upcard => 0,
            Head::Call => 1,
            Head::Discard => 2,
            Head::Play => 3,
        }
    }

    /// The number of input features this head's network expects.
    pub fn input_dim(self) -> usize {
        match self {
            Head::Upcard => UPCARD_IN,
            Head::Call => CALL_IN,
            Head::Discard => DISCARD_IN,
            Head::Play => PLAY_IN,
        }
    }

    /// The number of output logits (classes) this head produces.
    pub fn output_dim(self) -> usize {
        match self {
            Head::Upcard => 3,
            Head::Call => 7,
            Head::Discard | Head::Play => 24,
        }
    }
}

// Input widths, verified against the builders by a test below.
const UPCARD_IN: usize = 59;
const CALL_IN: usize = 83;
const DISCARD_IN: usize = 32;
const PLAY_IN: usize = 150;

/// The conventional target score. [`GameView`] does not carry the match's target,
/// so — like [`AdvancedAgent`](crate::AdvancedAgent) — the score features assume
/// the standard race to 10.
const TARGET_SCORE: f32 = 10.0;

// --- Card / suit slots -------------------------------------------------------

/// Maps a card to its trump-relative slot in `0..24` (see the [module
/// docs](self)). The mapping is a bijection over the 24-card deck for any fixed
/// `trump`.
pub fn card_slot(card: Card, trump: Suit) -> usize {
    if card.is_right_bower(trump) {
        return 0;
    }
    if card.is_left_bower(trump) {
        return 1;
    }
    if card.suit == trump {
        // The five trumps that are not bowers: A K Q T 9 -> slots 2..=6.
        return 2 + trump_honor_index(card.rank);
    }
    let next = trump.same_color();
    if card.suit == next {
        // The same-colour suit, minus its jack (the left bower): A K Q T 9.
        return 7 + next_suit_index(card.rank);
    }
    let offs = off_suits(trump);
    let base = if card.suit == offs[0] { 12 } else { 18 };
    base + rank_desc_index(card.rank)
}

/// The inverse of [`card_slot`]: the card occupying `slot` under `trump`.
pub fn slot_to_card(slot: usize, trump: Suit) -> Card {
    match slot {
        0 => Card::new(Rank::Jack, trump),
        1 => Card::new(Rank::Jack, trump.same_color()),
        2..=6 => Card::new(trump_honor_rank(slot - 2), trump),
        7..=11 => Card::new(next_suit_rank(slot - 7), trump.same_color()),
        12..=17 => Card::new(rank_from_desc(slot - 12), off_suits(trump)[0]),
        18..=23 => Card::new(rank_from_desc(slot - 18), off_suits(trump)[1]),
        _ => panic!("card slot out of range: {slot}"),
    }
}

/// The two suits whose colour is opposite to `trump`'s, in suit order.
fn off_suits(trump: Suit) -> [Suit; 2] {
    let mut out = [Suit::Clubs; 2];
    let mut n = 0;
    for s in Suit::ALL {
        if s.color() != trump.color() {
            out[n] = s;
            n += 1;
        }
    }
    debug_assert_eq!(n, 2);
    out
}

/// Slot offset for a non-bower trump card (A K Q T 9 -> 0..=4).
fn trump_honor_index(rank: Rank) -> usize {
    match rank {
        Rank::Ace => 0,
        Rank::King => 1,
        Rank::Queen => 2,
        Rank::Ten => 3,
        Rank::Nine => 4,
        Rank::Jack => unreachable!("the trump jack is the right bower"),
    }
}

fn trump_honor_rank(index: usize) -> Rank {
    [Rank::Ace, Rank::King, Rank::Queen, Rank::Ten, Rank::Nine][index]
}

/// Slot offset within the same-colour suit (A K Q T 9 -> 0..=4; its jack is the
/// left bower and is encoded as trump).
fn next_suit_index(rank: Rank) -> usize {
    match rank {
        Rank::Ace => 0,
        Rank::King => 1,
        Rank::Queen => 2,
        Rank::Ten => 3,
        Rank::Nine => 4,
        Rank::Jack => unreachable!("the same-colour jack is the left bower"),
    }
}

fn next_suit_rank(index: usize) -> Rank {
    [Rank::Ace, Rank::King, Rank::Queen, Rank::Ten, Rank::Nine][index]
}

/// Slot offset within an off-colour suit, ranks high to low (A K Q J T 9).
fn rank_desc_index(rank: Rank) -> usize {
    match rank {
        Rank::Ace => 0,
        Rank::King => 1,
        Rank::Queen => 2,
        Rank::Jack => 3,
        Rank::Ten => 4,
        Rank::Nine => 5,
    }
}

fn rank_from_desc(index: usize) -> Rank {
    [
        Rank::Ace,
        Rank::King,
        Rank::Queen,
        Rank::Jack,
        Rank::Ten,
        Rank::Nine,
    ][index]
}

/// The suit at trump-relative position `q` (0 = trump, 1 = next, 2/3 = off).
fn suit_at_rel(q: usize, trump: Suit) -> Suit {
    match q {
        0 => trump,
        1 => trump.same_color(),
        2 => off_suits(trump)[0],
        3 => off_suits(trump)[1],
        _ => panic!("relative suit out of range: {q}"),
    }
}

/// The trump-relative position of `suit` (inverse of [`suit_at_rel`]).
fn rel_of_suit(suit: Suit, trump: Suit) -> usize {
    (0..4)
        .find(|&q| suit_at_rel(q, trump) == suit)
        .expect("every suit has a relative position")
}

/// The number of clockwise steps from `me` to `other` (`me` -> 0, left -> 1,
/// partner -> 2, right -> 3).
fn rel_seat(me: Seat, other: Seat) -> usize {
    let mut s = me;
    for i in 0..4 {
        if s == other {
            return i;
        }
        s = s.next();
    }
    unreachable!("a seat is reached within four steps")
}

// --- Small feature helpers ---------------------------------------------------

/// Appends a 24-slot multi-hot of `cards` (a 1.0 at each card's slot).
fn push_hand(out: &mut Vec<f32>, cards: &[Card], trump: Suit) {
    let start = out.len();
    out.resize(start + 24, 0.0);
    for &c in cards {
        out[start + card_slot(c, trump)] = 1.0;
    }
}

/// Appends a 24-slot one-hot for a single `card`.
fn push_card(out: &mut Vec<f32>, card: Card, trump: Suit) {
    let start = out.len();
    out.resize(start + 24, 0.0);
    out[start + card_slot(card, trump)] = 1.0;
}

/// Appends a one-hot of length `n` with `index` set (or all zeros if out of
/// range, e.g. an absent value).
fn push_onehot(out: &mut Vec<f32>, n: usize, index: usize) {
    let start = out.len();
    out.resize(start + n, 0.0);
    if index < n {
        out[start + index] = 1.0;
    }
}

/// Appends the four score features: own and opponent scores (scaled by the
/// target) and "on the hill" flags for each side being one hand from the win.
fn push_score(out: &mut Vec<f32>, view: &GameView<'_>) {
    let me = view.scores.for_team(view.seat.team()) as f32;
    let them = view.scores.for_team(view.seat.team().opponent()) as f32;
    out.push(me / TARGET_SCORE);
    out.push(them / TARGET_SCORE);
    out.push(if me + 1.0 >= TARGET_SCORE { 1.0 } else { 0.0 });
    out.push(if them + 1.0 >= TARGET_SCORE { 1.0 } else { 0.0 });
}

/// Reconstructs, from the public record, every card already played (as a set of
/// trump-relative slots) and which suits each seat has revealed a void in.
fn played_and_voids(view: &GameView<'_>, trump: Suit) -> ([bool; 24], [[bool; 4]; 4]) {
    let mut seen = [false; 24];
    let mut void = [[false; 4]; 4];
    let tricks = view
        .completed_tricks
        .iter()
        .map(|(t, _)| t)
        .chain(std::iter::once(view.current_trick));
    for trick in tricks {
        let led = trick.led_suit(trump);
        for play in trick.plays() {
            seen[card_slot(play.card, trump)] = true;
            if let Some(led) = led {
                let eff = play.card.effective_suit(trump);
                if eff != led {
                    void[seat_idx(play.seat)][rel_of_suit(led, trump)] = true;
                }
            }
        }
    }
    (seen, void)
}

fn seat_idx(seat: Seat) -> usize {
    match seat {
        Seat::North => 0,
        Seat::East => 1,
        Seat::South => 2,
        Seat::West => 3,
    }
}

// --- Per-head feature builders -----------------------------------------------

/// Features for the first-round up-card bid. The candidate trump is the
/// up-card's suit.
pub fn upcard_features(view: &GameView<'_>, up_card: Card) -> Vec<f32> {
    let trump = up_card.suit;
    let mut out = Vec::with_capacity(UPCARD_IN);
    push_hand(&mut out, view.hand, trump);
    push_card(&mut out, up_card, trump);
    let rd = rel_seat(view.seat, view.dealer);
    push_onehot(&mut out, 4, rd);
    out.push(if rd == 0 { 1.0 } else { 0.0 }); // I am the dealer
    out.push(if rd == 2 { 1.0 } else { 0.0 }); // the dealer is my partner
    let trumps = view.hand.iter().filter(|c| c.is_trump(trump)).count();
    out.push(trumps as f32 / 5.0);
    push_score(&mut out, view);
    debug_assert_eq!(out.len(), UPCARD_IN);
    out
}

/// Features for the second-round call. The hand is encoded once per *candidate*
/// suit (in that suit's trump frame), in the canonical order [next, off0, off1]
/// matching the action classes.
pub fn call_features(view: &GameView<'_>, turned_down: Suit, stuck: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(CALL_IN);
    for &suit in &call_candidates(turned_down) {
        push_hand(&mut out, view.hand, suit);
    }
    let rd = rel_seat(view.seat, view.dealer);
    push_onehot(&mut out, 4, rd);
    out.push(if rd == 0 { 1.0 } else { 0.0 });
    out.push(if rd == 2 { 1.0 } else { 0.0 });
    out.push(if stuck { 1.0 } else { 0.0 });
    push_score(&mut out, view);
    debug_assert_eq!(out.len(), CALL_IN);
    out
}

/// Features for the dealer's discard (trump is set; the hand holds six cards).
pub fn discard_features(view: &GameView<'_>) -> Vec<f32> {
    let trump = view.trump().expect("trump is set before the discard");
    let mut out = Vec::with_capacity(DISCARD_IN);
    push_hand(&mut out, view.hand, trump);
    // Per relative-suit counts give the net the shape of the hand directly.
    for q in 0..4 {
        let suit = suit_at_rel(q, trump);
        let count = view
            .hand
            .iter()
            .filter(|c| c.effective_suit(trump) == suit)
            .count();
        out.push(count as f32 / 6.0);
    }
    push_score(&mut out, view);
    debug_assert_eq!(out.len(), DISCARD_IN);
    out
}

/// Features for a card play. The richest head: it sees the hand, every card
/// seen so far, the cards on the table by relative seat, revealed voids, the
/// contract, position in the trick, and the score.
pub fn play_features(view: &GameView<'_>) -> Vec<f32> {
    let trump = view.trump().expect("trump is set during play");
    let contract = view.contract.expect("a hand in play has a contract");
    let me = view.seat;
    let mut out = Vec::with_capacity(PLAY_IN);

    push_hand(&mut out, view.hand, trump);

    let (seen, void) = played_and_voids(view, trump);
    let start = out.len();
    out.resize(start + 24, 0.0);
    for (slot, &s) in seen.iter().enumerate() {
        if s {
            out[start + slot] = 1.0;
        }
    }

    // Cards on the table in the current trick, by relative seat (left, partner,
    // right); my own slot is never populated because I have not yet played.
    let mut trick_blocks = [[0.0f32; 24]; 3];
    for play in view.current_trick.plays() {
        let r = rel_seat(me, play.seat);
        if (1..=3).contains(&r) {
            trick_blocks[r - 1][card_slot(play.card, trump)] = 1.0;
        }
    }
    for block in &trick_blocks {
        out.extend_from_slice(block);
    }

    // The led suit (relative), or all-zero plus a flag when I am leading.
    match view.current_trick.led_suit(trump) {
        Some(led) => push_onehot(&mut out, 4, rel_of_suit(led, trump)),
        None => push_onehot(&mut out, 4, usize::MAX),
    }
    out.push(if view.current_trick.is_empty() {
        1.0
    } else {
        0.0
    });

    // The contract, from my point of view.
    push_onehot(&mut out, 4, rel_seat(me, contract.maker));
    out.push(if contract.alone { 1.0 } else { 0.0 });
    out.push(if contract.maker.team() == me.team() {
        1.0
    } else {
        0.0
    });

    // Revealed voids for the other three seats, by relative seat and suit.
    for r in 1..=3 {
        let mut s = me;
        for _ in 0..r {
            s = s.next();
        }
        for &voided in &void[seat_idx(s)] {
            out.push(if voided { 1.0 } else { 0.0 });
        }
    }

    out.push(view.current_trick.len() as f32 / 4.0);
    out.push(view.completed_tricks.len() as f32 / 5.0);
    out.push(trump_outstanding(view, trump) as f32 / 7.0);
    push_score(&mut out, view);

    debug_assert_eq!(out.len(), PLAY_IN);
    out
}

/// How many trumps are still unaccounted for: not in my hand and not yet played.
fn trump_outstanding(view: &GameView<'_>, trump: Suit) -> usize {
    let (seen, _) = played_and_voids(view, trump);
    let mut count = 0;
    for s in Suit::ALL {
        for r in Rank::ALL {
            let c = Card::new(r, s);
            if c.is_trump(trump) && !seen[card_slot(c, trump)] && !view.hand.contains(&c) {
                count += 1;
            }
        }
    }
    count
}

// --- Action encoding / decoding ----------------------------------------------

/// The legal-class mask for the up-card bid: pass, order up, order up alone.
pub fn upcard_legal() -> u32 {
    0b111
}

/// The class index of an up-card bid.
pub fn upcard_class(bid: UpcardBid) -> usize {
    match bid {
        UpcardBid::Pass => 0,
        UpcardBid::OrderUp(Bid::WithPartner) => 1,
        UpcardBid::OrderUp(Bid::Alone) => 2,
    }
}

/// The up-card bid for a class index.
pub fn upcard_action(class: usize) -> UpcardBid {
    match class {
        0 => UpcardBid::Pass,
        1 => UpcardBid::OrderUp(Bid::WithPartner),
        2 => UpcardBid::OrderUp(Bid::Alone),
        _ => panic!("up-card class out of range: {class}"),
    }
}

/// The three nameable suits in canonical order: "next" (same colour as the
/// turned-down suit) then the two off-colour suits in suit order.
fn call_candidates(turned_down: Suit) -> [Suit; 3] {
    let offs = off_suits(turned_down);
    [turned_down.same_color(), offs[0], offs[1]]
}

/// The legal-class mask for the second-round call. Class 0 is pass, then each of
/// the three candidate suits with partner / alone. Pass is dropped when the
/// dealer is `stuck` under stick-the-dealer.
pub fn call_legal(stuck: bool) -> u32 {
    if stuck { 0b1111110 } else { 0b1111111 }
}

/// The class index of a call bid, given the turned-down suit.
pub fn call_class(bid: CallBid, turned_down: Suit) -> usize {
    match bid {
        CallBid::Pass => 0,
        CallBid::Call { suit, bid } => {
            let ci = call_candidates(turned_down)
                .iter()
                .position(|&s| s == suit)
                .expect("a called suit is one of the three candidates");
            1 + ci * 2 + usize::from(bid.is_alone())
        }
    }
}

/// The call bid for a class index, given the turned-down suit.
pub fn call_action(class: usize, turned_down: Suit) -> CallBid {
    if class == 0 {
        return CallBid::Pass;
    }
    let ci = (class - 1) / 2;
    let alone = (class - 1) % 2 == 1;
    CallBid::Call {
        suit: call_candidates(turned_down)[ci],
        bid: if alone { Bid::Alone } else { Bid::WithPartner },
    }
}

/// A bitmask over the 24 card slots marking exactly the cards in `cards` as
/// legal (used for both [`Head::Discard`] and [`Head::Play`]).
pub fn card_mask(cards: &[Card], trump: Suit) -> u32 {
    cards
        .iter()
        .fold(0u32, |m, &c| m | (1 << card_slot(c, trump)))
}

/// Picks the card among `candidates` whose slot scores highest under `logits`.
pub fn best_card(logits: &[f32], candidates: &[Card], trump: Suit) -> Card {
    *candidates
        .iter()
        .max_by(|&&a, &&b| logits[card_slot(a, trump)].total_cmp(&logits[card_slot(b, trump)]))
        .expect("candidates is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::{Contract, GameRules, Play, Scores, Trick};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn card_slot_is_a_bijection_for_every_trump() {
        for trump in Suit::ALL {
            let mut seen = [false; 24];
            for c in Card::deck() {
                let slot = card_slot(c, trump);
                assert!(slot < 24);
                assert!(!seen[slot], "slot {slot} collided under trump {trump:?}");
                seen[slot] = true;
                assert_eq!(slot_to_card(slot, trump), c, "round trip under {trump:?}");
            }
            assert!(seen.iter().all(|&s| s), "every slot filled under {trump:?}");
        }
    }

    #[test]
    fn bowers_take_the_top_two_slots() {
        let trump = Suit::Hearts;
        assert_eq!(card_slot(card(Rank::Jack, Suit::Hearts), trump), 0); // right
        assert_eq!(card_slot(card(Rank::Jack, Suit::Diamonds), trump), 1); // left
        assert_eq!(card_slot(card(Rank::Ace, Suit::Hearts), trump), 2);
    }

    fn view<'a>(hand: &'a [Card], trick: &'a Trick, contract: Option<Contract>) -> GameView<'a> {
        GameView {
            seat: Seat::North,
            dealer: Seat::West,
            hand,
            contract,
            current_trick: trick,
            completed_tricks: &[],
            scores: Scores::default(),
            rules: GameRules::default(),
        }
    }

    #[test]
    fn feature_lengths_match_declared_dims() {
        let hand = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Clubs),
        ];
        let trick = Trick::new();
        let bidding = view(&hand, &trick, None);
        assert_eq!(
            upcard_features(&bidding, card(Rank::Nine, Suit::Spades)).len(),
            UPCARD_IN
        );
        assert_eq!(call_features(&bidding, Suit::Spades, false).len(), CALL_IN);

        let contract = Contract {
            trump: Suit::Spades,
            maker: Seat::North,
            alone: false,
        };
        let six = [
            card(Rank::Jack, Suit::Spades),
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Hearts),
            card(Rank::Nine, Suit::Diamonds),
            card(Rank::Ten, Suit::Clubs),
            card(Rank::Nine, Suit::Spades),
        ];
        let playing = view(&six[..5], &trick, Some(contract));
        let discard = view(&six, &trick, Some(contract));
        assert_eq!(discard_features(&discard).len(), DISCARD_IN);
        assert_eq!(play_features(&playing).len(), PLAY_IN);
    }

    #[test]
    fn call_class_round_trips() {
        let turned_down = Suit::Diamonds;
        for class in 0..7 {
            let action = call_action(class, turned_down);
            assert_eq!(call_class(action, turned_down), class);
        }
        // Pass is illegal exactly when stuck.
        assert_eq!(call_legal(false) & 1, 1);
        assert_eq!(call_legal(true) & 1, 0);
    }

    #[test]
    fn upcard_class_round_trips() {
        for class in 0..3 {
            assert_eq!(upcard_class(upcard_action(class)), class);
        }
    }

    #[test]
    fn best_card_respects_logits_and_slots() {
        let trump = Suit::Spades;
        let candidates = [
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ace, Suit::Hearts),
        ];
        let mut logits = vec![0.0f32; 24];
        // Favour the ace of hearts' slot.
        logits[card_slot(candidates[1], trump)] = 5.0;
        assert_eq!(best_card(&logits, &candidates, trump), candidates[1]);
        assert_eq!(card_mask(&candidates, trump).count_ones(), 2);
    }

    #[test]
    fn play_features_flag_a_revealed_void() {
        // East fails to follow North's heart lead, revealing East void in hearts.
        let trump = Suit::Spades;
        let mut done = Trick::new();
        for (seat, c) in [
            (Seat::North, card(Rank::Ace, Suit::Hearts)),
            (Seat::East, card(Rank::Nine, Suit::Clubs)),
            (Seat::South, card(Rank::King, Suit::Hearts)),
            (Seat::West, card(Rank::Ten, Suit::Hearts)),
        ] {
            done.push(Play { seat, card: c });
        }
        let completed = [(done, Seat::South)];
        let hand = [card(Rank::Jack, Suit::Spades)];
        let empty = Trick::new();
        let contract = Contract {
            trump,
            maker: Seat::North,
            alone: false,
        };
        let v = GameView {
            seat: Seat::North,
            dealer: Seat::West,
            hand: &hand,
            contract: Some(contract),
            current_trick: &empty,
            completed_tricks: &completed,
            scores: Scores::default(),
            rules: GameRules::default(),
        };
        let (_seen, void) = played_and_voids(&v, trump);
        // East is North's left (relative 1); hearts is the "next" suit (relative 1)
        // since trump is spades.
        assert!(void[seat_idx(Seat::East)][rel_of_suit(Suit::Hearts, trump)]);
    }
}
