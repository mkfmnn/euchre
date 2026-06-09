//! The Euchre game engine: deals, runs the auction, plays the tricks, and
//! keeps score.
//!
//! The engine owns the rules. It holds four [`Agent`]s — one per [`Seat`] — and
//! drives a complete game by calling into them at each decision point, handing
//! each a [`GameView`] restricted to what that seat may legitimately know. The
//! engine validates every choice an agent makes and rejects illegal ones, so a
//! buggy or hostile agent can spoil its own game but never corrupt the engine's
//! state.
//!
//! ## Quick start
//!
//! ```
//! use euchre_engine::{Engine, EngineConfig};
//! use euchre_engine::agents::RandomAgent;
//! use euchre_engine::rng::Rng;
//! use euchre_interface::Agent;
//!
//! // Four independent random bots, one per seat.
//! let agents: [Box<dyn Agent>; 4] = [
//!     Box::new(RandomAgent::new(Rng::seed_from_u64(1))),
//!     Box::new(RandomAgent::new(Rng::seed_from_u64(2))),
//!     Box::new(RandomAgent::new(Rng::seed_from_u64(3))),
//!     Box::new(RandomAgent::new(Rng::seed_from_u64(4))),
//! ];
//! let config = EngineConfig { seed: Some(7), ..EngineConfig::default() };
//! let mut engine = Engine::new(agents, config);
//! let outcome = engine.play_match();
//! assert!(outcome.winner_score() >= config.target_score);
//! ```

use euchre_interface::{
    Agent, CallBid, Card, Contract, GameView, HandResult, Play, Scores, Seat, Team, Trick,
    UpcardBid,
};

use crate::rng::Rng;

/// Tunable rules for a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    /// Points needed to win the match. Standard Euchre plays to 10.
    pub target_score: u8,
    /// "Stick the dealer": if every seat passes both rounds, the dealer is
    /// forced to name a trump suit rather than the hand being thrown in.
    ///
    /// When `false`, a passed-out hand is redealt by the next dealer.
    pub stick_the_dealer: bool,
    /// An optional fixed seed for the deck shuffler, making a whole match
    /// reproducible. `None` seeds from system entropy.
    pub seed: Option<u64>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            target_score: 10,
            stick_the_dealer: true,
            seed: None,
        }
    }
}

/// Something an agent did that the rules forbid.
///
/// The engine never produces these on its own; each corresponds to an [`Agent`]
/// returning a choice outside the legal set it was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineError {
    /// A dealer's [`discard`](Agent::discard) named a card not in hand.
    IllegalDiscard { seat: Seat, card: Card },
    /// A [`play_card`](Agent::play_card) returned a card outside the legal set
    /// (either not held or failing to follow suit when able).
    IllegalPlay { seat: Seat, card: Card },
    /// A second-round [`bid_call`](Agent::bid_call) named the turned-down suit,
    /// which is not allowed.
    CalledTurnedDownSuit {
        seat: Seat,
        suit: euchre_interface::Suit,
    },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::IllegalDiscard { seat, card } => {
                write!(
                    f,
                    "{seat:?} tried to discard {card}, which it does not hold"
                )
            }
            EngineError::IllegalPlay { seat, card } => {
                write!(
                    f,
                    "{seat:?} tried to play {card}, which is not a legal play"
                )
            }
            EngineError::CalledTurnedDownSuit { seat, suit } => {
                write!(f, "{seat:?} tried to call {suit}, the turned-down suit")
            }
        }
    }
}

impl std::error::Error for EngineError {}

/// How a single hand turned out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandOutcome {
    /// Trump was named and the hand was played to completion.
    Played(HandResult),
    /// Every seat passed both bidding rounds and the hand was thrown in.
    ///
    /// Only possible when [`EngineConfig::stick_the_dealer`] is `false`.
    PassedOut,
}

/// The result of playing a full match to the target score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchOutcome {
    /// The team that reached the target score first.
    pub winner: Team,
    /// Final cumulative scores.
    pub scores: Scores,
    /// The number of hands played.
    pub hands_played: u32,
}

impl MatchOutcome {
    /// The winning team's final score.
    pub fn winner_score(&self) -> u8 {
        self.scores.for_team(self.winner)
    }
}

/// Maps a [`Seat`] to a `0..4` array index in clockwise order.
const fn seat_index(seat: Seat) -> usize {
    match seat {
        Seat::North => 0,
        Seat::East => 1,
        Seat::South => 2,
        Seat::West => 3,
    }
}

/// The driver that owns the agents and runs the game.
///
/// Construct with [`Engine::new`], then call [`Engine::play_match`] for a full
/// game or [`Engine::play_hand`] to step one hand at a time.
pub struct Engine {
    agents: [Box<dyn Agent>; 4],
    config: EngineConfig,
    scores: Scores,
    dealer: Seat,
    rng: Rng,
}

impl Engine {
    /// Creates an engine from four agents — indexed `[North, East, South,
    /// West]` — and a configuration.
    ///
    /// The opening dealer is `North`; it rotates clockwise after every hand
    /// (including passed-out redeals).
    pub fn new(agents: [Box<dyn Agent>; 4], config: EngineConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => Rng::seed_from_u64(seed),
            None => Rng::from_entropy(),
        };
        Engine {
            agents,
            config,
            scores: Scores::default(),
            dealer: Seat::North,
            rng,
        }
    }

    /// The current cumulative scores.
    pub fn scores(&self) -> Scores {
        self.scores
    }

    /// The seat that will deal the next hand.
    pub fn dealer(&self) -> Seat {
        self.dealer
    }

    /// Borrows the agent occupying `seat`.
    pub fn agent(&self, seat: Seat) -> &dyn Agent {
        &*self.agents[seat_index(seat)]
    }

    /// Whether either team has reached the target score.
    pub fn is_over(&self) -> bool {
        self.scores.north_south >= self.config.target_score
            || self.scores.east_west >= self.config.target_score
    }

    /// Plays hands until one team reaches the target score, then reports the
    /// outcome.
    ///
    /// Panics only if an agent makes an illegal choice; see [`Engine::play_hand`]
    /// for the fallible variant.
    pub fn play_match(&mut self) -> MatchOutcome {
        let mut hands_played = 0;
        while !self.is_over() {
            self.play_hand()
                .expect("agent made an illegal choice during the match");
            hands_played += 1;
        }
        let winner = if self.scores.north_south >= self.config.target_score {
            Team::NorthSouth
        } else {
            Team::EastWest
        };
        MatchOutcome {
            winner,
            scores: self.scores,
            hands_played,
        }
    }

    /// Deals and plays a single hand, updating the scores and rotating the
    /// dealer.
    ///
    /// Returns the [`HandOutcome`], or an [`EngineError`] if an agent returned
    /// an illegal choice — in which case the hand is abandoned and neither the
    /// scores nor the dealer change.
    pub fn play_hand(&mut self) -> Result<HandOutcome, EngineError> {
        let dealer = self.dealer;
        let outcome = self.run_hand(dealer)?;
        // Apply scoring.
        if let HandOutcome::Played(result) = &outcome {
            let (team, points) = result.points_awarded;
            match team {
                Team::NorthSouth => self.scores.north_south += points,
                Team::EastWest => self.scores.east_west += points,
            }
        }
        self.dealer = dealer.next();
        Ok(outcome)
    }

    /// Runs the full life cycle of one hand without touching match-level state
    /// (scores, dealer). Split out so the borrow of `self.rng` and the agents
    /// stays tidy.
    fn run_hand(&mut self, dealer: Seat) -> Result<HandOutcome, EngineError> {
        // ---- Deal ----
        let mut deck = Card::deck();
        self.rng.shuffle(&mut deck);
        let mut hands: [Vec<Card>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        // Deal five cards to each seat, clockwise from the dealer's left.
        let mut idx = 0;
        for _round in 0..5 {
            let mut seat = dealer.next();
            for _ in 0..4 {
                hands[seat_index(seat)].push(deck[idx]);
                idx += 1;
                seat = seat.next();
            }
        }
        // The next card is turned up; the rest form the (unseen) kitty.
        let up_card = deck[idx];

        // Sort each hand for stable, readable presentation to agents/humans.
        for hand in &mut hands {
            hand.sort_by_key(|c| (c.suit, c.rank));
        }

        // ---- Bidding ----
        let Some(contract) = self.run_bidding(dealer, &mut hands, up_card)? else {
            return Ok(HandOutcome::PassedOut);
        };

        // ---- Play ----
        let (maker_tricks, completed) = self.run_play(dealer, contract, &mut hands)?;

        // ---- Score ----
        let result = score_hand(contract, maker_tricks);
        self.notify_hand_end(dealer, contract, &hands, &completed, &result);
        Ok(HandOutcome::Played(result))
    }

    /// Runs both bidding rounds. Returns the agreed [`Contract`], or `None` if
    /// the hand was passed out (only possible without "stick the dealer").
    fn run_bidding(
        &mut self,
        dealer: Seat,
        hands: &mut [Vec<Card>; 4],
        up_card: Card,
    ) -> Result<Option<Contract>, EngineError> {
        let empty_trick = Trick::new();

        // Round 1: order up the up-card's suit, or pass.
        let mut seat = dealer.next();
        for _ in 0..4 {
            let view = self.view(
                seat,
                dealer,
                &hands[seat_index(seat)],
                None,
                &empty_trick,
                &[],
            );
            let bid = self.agents[seat_index(seat)].bid_upcard(&view, up_card);
            if let UpcardBid::OrderUp(bid) = bid {
                let contract = Contract {
                    trump: up_card.suit,
                    maker: seat,
                    alone: bid.is_alone(),
                };
                self.dealer_takes_up_card(dealer, contract, hands, up_card)?;
                return Ok(Some(contract));
            }
            seat = seat.next();
        }

        // Round 2: name any other suit, or pass.
        let turned_down = up_card.suit;
        let mut seat = dealer.next();
        for _ in 0..4 {
            let is_dealer = seat == dealer;
            let forced = is_dealer && self.config.stick_the_dealer;
            let view = self.view(
                seat,
                dealer,
                &hands[seat_index(seat)],
                None,
                &empty_trick,
                &[],
            );
            let bid = self.agents[seat_index(seat)].bid_call(&view, turned_down);
            match bid {
                CallBid::Call { suit, bid } => {
                    if suit == turned_down {
                        return Err(EngineError::CalledTurnedDownSuit { seat, suit });
                    }
                    return Ok(Some(Contract {
                        trump: suit,
                        maker: seat,
                        alone: bid.is_alone(),
                    }));
                }
                CallBid::Pass if forced => {
                    // Stuck dealer must call: pick the strongest legal suit on
                    // their behalf rather than honoring the illegal pass.
                    let suit = best_trump_for(&hands[seat_index(seat)], turned_down);
                    return Ok(Some(Contract {
                        trump: suit,
                        maker: seat,
                        alone: false,
                    }));
                }
                CallBid::Pass => {}
            }
            seat = seat.next();
        }

        Ok(None)
    }

    /// The dealer picks up the ordered-up card and discards one, leaving five.
    fn dealer_takes_up_card(
        &mut self,
        dealer: Seat,
        contract: Contract,
        hands: &mut [Vec<Card>; 4],
        up_card: Card,
    ) -> Result<(), EngineError> {
        let di = seat_index(dealer);
        hands[di].push(up_card);
        hands[di].sort_by_key(|c| (c.suit, c.rank));
        let empty_trick = Trick::new();
        let view = self.view(
            dealer,
            dealer,
            &hands[di],
            Some(contract),
            &empty_trick,
            &[],
        );
        let discard = self.agents[di].discard(&view);
        let pos =
            hands[di]
                .iter()
                .position(|&c| c == discard)
                .ok_or(EngineError::IllegalDiscard {
                    seat: dealer,
                    card: discard,
                })?;
        hands[di].remove(pos);
        Ok(())
    }

    /// Plays the five tricks. Returns the makers' trick count and the record of
    /// completed tricks.
    fn run_play(
        &mut self,
        dealer: Seat,
        contract: Contract,
        hands: &mut [Vec<Card>; 4],
    ) -> Result<(u8, Vec<(Trick, Seat)>), EngineError> {
        let trump = contract.trump;
        let sitting_out = contract.sitting_out();
        let makers = contract.maker.team();
        let mut completed: Vec<(Trick, Seat)> = Vec::with_capacity(5);
        let mut maker_tricks = 0u8;

        // The first trick is led by the seat to the dealer's left.
        let mut leader = dealer.next();

        for _trick_num in 0..5 {
            let mut trick = Trick::new();
            let mut seat = leader;
            for _ in 0..4 {
                if Some(seat) == sitting_out {
                    seat = seat.next();
                    continue;
                }
                let si = seat_index(seat);
                let legal = legal_plays(&hands[si], &trick, trump);
                let view = self.view(seat, dealer, &hands[si], Some(contract), &trick, &completed);
                let card = self.agents[si].play_card(&view, &legal);
                if !legal.contains(&card) {
                    return Err(EngineError::IllegalPlay { seat, card });
                }
                let pos = hands[si].iter().position(|&c| c == card).unwrap();
                hands[si].remove(pos);
                trick.push(Play { seat, card });
                seat = seat.next();
            }
            let winner = trick.winner(trump).expect("a completed trick has a winner");
            if winner.team() == makers {
                maker_tricks += 1;
            }
            completed.push((trick, winner));
            leader = winner;
        }

        Ok((maker_tricks, completed))
    }

    /// Delivers the end-of-hand summary to every agent that played.
    fn notify_hand_end(
        &mut self,
        dealer: Seat,
        contract: Contract,
        hands: &[Vec<Card>; 4],
        completed: &[(Trick, Seat)],
        result: &HandResult,
    ) {
        let empty_trick = Trick::new();
        for seat in Seat::ALL {
            if Some(seat) == contract.sitting_out() {
                continue;
            }
            let si = seat_index(seat);
            let view = self.view(
                seat,
                dealer,
                &hands[si],
                Some(contract),
                &empty_trick,
                completed,
            );
            self.agents[si].observe_hand_end(&view, result);
        }
    }

    /// Builds the read-only [`GameView`] handed to an agent at a decision point.
    fn view<'a>(
        &self,
        seat: Seat,
        dealer: Seat,
        hand: &'a [Card],
        contract: Option<Contract>,
        current_trick: &'a Trick,
        completed_tricks: &'a [(Trick, Seat)],
    ) -> GameView<'a> {
        GameView {
            seat,
            dealer,
            hand,
            contract,
            current_trick,
            completed_tricks,
            scores: self.scores,
        }
    }
}

/// The cards a seat may legally play to `trick` from `hand`.
///
/// If a suit was led and the hand holds a card of that effective suit, only
/// those cards are legal (the obligation to follow suit). Otherwise the whole
/// hand is legal. Leading to an empty trick allows any card.
pub fn legal_plays(hand: &[Card], trick: &Trick, trump: euchre_interface::Suit) -> Vec<Card> {
    match trick.led_suit(trump) {
        None => hand.to_vec(),
        Some(led) => {
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
    }
}

/// Scores a completed hand from the makers' trick count.
///
/// * 3 or 4 tricks: 1 point to the makers.
/// * All 5 (a *march*): 2 points, or 4 if the maker went alone.
/// * Fewer than 3 (*euchred*): 2 points to the defenders.
pub fn score_hand(contract: Contract, maker_tricks: u8) -> HandResult {
    let makers = contract.maker.team();
    let march = maker_tricks == 5;
    let euchred = maker_tricks < 3;

    let points_awarded = if euchred {
        (makers.opponent(), 2)
    } else if march {
        if contract.alone {
            (makers, 4)
        } else {
            (makers, 2)
        }
    } else {
        (makers, 1)
    };

    HandResult {
        makers,
        maker_tricks,
        euchred,
        march,
        alone: contract.alone,
        points_awarded,
    }
}

/// Picks the strongest suit (other than `turned_down`) to make trump, used when
/// a stuck dealer declines to call. Chooses the suit giving the most trump
/// cards, breaking ties by the combined rank of those cards.
fn best_trump_for(hand: &[Card], turned_down: euchre_interface::Suit) -> euchre_interface::Suit {
    use euchre_interface::Suit;
    Suit::ALL
        .into_iter()
        .filter(|&s| s != turned_down)
        .max_by_key(|&trump| {
            let count = hand.iter().filter(|c| c.is_trump(trump)).count();
            let strength: u32 = hand
                .iter()
                .filter(|c| c.is_trump(trump))
                .map(|c| c.trump_strength(trump, trump))
                .sum();
            (count, strength)
        })
        .expect("three candidate suits remain")
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::{Rank, Suit};

    #[test]
    fn scoring_rules() {
        let solo = Contract {
            trump: Suit::Hearts,
            maker: Seat::North,
            alone: true,
        };
        let team = Contract {
            trump: Suit::Hearts,
            maker: Seat::North,
            alone: false,
        };

        assert_eq!(score_hand(team, 1).points_awarded, (Team::EastWest, 2));
        assert_eq!(score_hand(team, 2).points_awarded, (Team::EastWest, 2));
        assert_eq!(score_hand(team, 3).points_awarded, (Team::NorthSouth, 1));
        assert_eq!(score_hand(team, 4).points_awarded, (Team::NorthSouth, 1));
        assert_eq!(score_hand(team, 5).points_awarded, (Team::NorthSouth, 2));
        assert_eq!(score_hand(solo, 5).points_awarded, (Team::NorthSouth, 4));
        assert_eq!(score_hand(solo, 4).points_awarded, (Team::NorthSouth, 1));

        assert!(score_hand(team, 2).euchred);
        assert!(score_hand(team, 5).march);
        assert!(!score_hand(team, 3).euchred);
    }

    #[test]
    fn must_follow_suit_when_able() {
        let trump = Suit::Spades;
        let hand = vec![
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::Nine, Suit::Hearts),
            Card::new(Rank::King, Suit::Clubs),
        ];
        let mut trick = Trick::new();
        trick.push(Play {
            seat: Seat::North,
            card: Card::new(Rank::Ten, Suit::Hearts),
        });
        let legal = legal_plays(&hand, &trick, trump);
        assert_eq!(legal.len(), 2);
        assert!(legal.iter().all(|c| c.suit == Suit::Hearts));
    }

    #[test]
    fn left_bower_must_follow_trump() {
        let trump = Suit::Spades;
        // The Jack of clubs is the left bower: it counts as a spade here.
        let hand = vec![
            Card::new(Rank::Jack, Suit::Clubs),
            Card::new(Rank::Ace, Suit::Hearts),
        ];
        let mut trick = Trick::new();
        trick.push(Play {
            seat: Seat::North,
            card: Card::new(Rank::Nine, Suit::Spades),
        });
        let legal = legal_plays(&hand, &trick, trump);
        assert_eq!(legal, vec![Card::new(Rank::Jack, Suit::Clubs)]);
    }

    #[test]
    fn can_play_anything_when_void() {
        let trump = Suit::Spades;
        let hand = vec![
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::King, Suit::Clubs),
        ];
        let mut trick = Trick::new();
        trick.push(Play {
            seat: Seat::North,
            card: Card::new(Rank::Nine, Suit::Diamonds),
        });
        let legal = legal_plays(&hand, &trick, trump);
        assert_eq!(legal.len(), 2);
    }

    #[test]
    fn best_trump_prefers_most_trumps() {
        // Three spades vs one heart: spades should win even though hearts is an
        // option.
        let hand = vec![
            Card::new(Rank::Nine, Suit::Spades),
            Card::new(Rank::Ten, Suit::Spades),
            Card::new(Rank::Queen, Suit::Spades),
            Card::new(Rank::Ace, Suit::Hearts),
        ];
        assert_eq!(best_trump_for(&hand, Suit::Diamonds), Suit::Spades);
    }
}
