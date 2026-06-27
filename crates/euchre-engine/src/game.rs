//! The Euchre game **core**: a deterministic state machine.
//!
//! [`Game`] owns the complete, authoritative state of a single match — every
//! seat's hand, the kitty, the running score — and advances through a hand one
//! decision at a time. It deliberately knows nothing about *who* makes those
//! decisions: it simply reports [what is needed next](Game::next_action) and
//! [accepts an answer](Game::apply). This makes it equally suitable for driving
//! a terminal game loop (see [`crate::driver`]) or a websocket server that
//! relays each request to whichever of up to four connected clients is on turn.
//!
//! ## The loop
//!
//! A caller repeatedly:
//!
//! 1. calls [`Game::next_action`] to learn what is required and from whom,
//! 2. builds a per-seat [`GameView`] with [`Game::view`] to hand to that player,
//! 3. collects the player's choice and submits it with [`Game::apply`],
//!
//! until [`Action::HandComplete`] is reported. At that point the caller checks
//! [`Game::is_over`]; if the match is decided it stops, otherwise it deals the
//! next hand with [`Game::start_next_hand`].
//!
//! ## Determinism and dealing
//!
//! The core contains no randomness. Each hand is dealt from a 24-card deck that
//! the caller supplies (already shuffled), which keeps the engine reproducible
//! and trivial to test. The [driver](crate::driver) is responsible for shuffling.

use std::fmt;

use euchre_interface::{
    CallBid, Card, Contract, GameRules, GameView, HandResult, HandScore, Play, Scores, Seat, Suit,
    Trick, UpcardBid,
};

/// The number of cards dealt to each seat.
const HAND_SIZE: usize = 5;
/// The number of tricks played in a hand.
const TRICKS_PER_HAND: usize = 5;

/// A fixed player identity, independent of the rotating deal: `0` = North,
/// `1` = East, `2` = South, `3` = West. Used to index the per-player arrays
/// (hands) so a player keeps the same slot from hand to hand even as the
/// dealer-relative [`Seat`] it occupies changes.
type Player = usize;

/// A fixed team identity: `0` = North/South (players 0 and 2), `1` = East/West
/// (players 1 and 3). Used to index the per-team score array.
type TeamId = usize;

/// The team a fixed player belongs to (`player % 2`: 0/2 → 0, 1/3 → 1).
fn team_of_player(player: Player) -> TeamId {
    player % 2
}

/// The fixed setup for a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GameConfig {
    /// The optional house rules in effect.
    pub rules: GameRules,
    /// The score a team must reach (or exceed) to win the match. Conventionally
    /// 10; some play to 5.
    pub target_score: u8,
    /// The fixed player who deals the very first hand (`0` = North, `1` = East,
    /// `2` = South, `3` = West). The deal rotates clockwise each subsequent hand.
    pub first_dealer: Player,
}

impl Default for GameConfig {
    fn default() -> Self {
        GameConfig {
            rules: GameRules::default(),
            target_score: 10,
            first_dealer: 0,
        }
    }
}

/// Where in a hand the state machine currently sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// First bidding round: `turn` decides whether to order up the up-card.
    BidRound1 { turn: Seat },
    /// Second bidding round: `turn` decides whether to name a trump suit.
    BidRound2 { turn: Seat },
    /// The dealer, having taken the up-card into hand, must discard a card.
    Discard,
    /// The play of the tricks: `turn` must play a card.
    Play { turn: Seat },
    /// The hand is over. [`Game::last_result`] holds how it was scored, and
    /// [`Game::winner`] is set if the match is now decided.
    HandComplete,
}

/// What the engine needs next in order to advance, as reported by
/// [`Game::next_action`].
///
/// Every variant except [`Action::HandComplete`] names the `seat` whose turn it
/// is; the caller should obtain that seat's [`GameView`] via [`Game::view`],
/// collect its decision, and submit the matching [`Decision`] to
/// [`Game::apply`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Action {
    /// First bidding round. `seat` may order up `up_card`'s suit as trump or
    /// pass. Answered with [`Decision::Upcard`].
    BidUpcard { seat: Seat, up_card: Card },
    /// Second bidding round. `seat` may name any suit other than `turned_down`
    /// as trump, or pass. When `may_pass` is `false` the "stick the dealer"
    /// rule forbids passing and the seat must name a suit. Answered with
    /// [`Decision::Call`].
    BidCall {
        seat: Seat,
        turned_down: Suit,
        may_pass: bool,
    },
    /// The dealer (`seat`) took the ordered-up `up_card` and must now discard
    /// down to five cards. Answered with [`Decision::Discard`].
    Discard { seat: Seat, up_card: Card },
    /// `seat` must play one of `legal` into the current trick. Answered with
    /// [`Decision::Play`].
    Play { seat: Seat, legal: Vec<Card> },
    /// The hand has ended. `dealer` is the fixed player who dealt it. The caller
    /// should consult [`Game::is_over`] to decide between stopping and dealing
    /// the next hand, and may fetch each seat's view of the scoring with
    /// [`Game::hand_result`].
    HandComplete { dealer: Player },
}

/// A player's answer to an [`Action`], submitted via [`Game::apply`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Decision {
    /// Answers [`Action::BidUpcard`].
    Upcard(UpcardBid),
    /// Answers [`Action::BidCall`].
    Call(CallBid),
    /// Answers [`Action::Discard`]: the card the dealer buries.
    Discard(Card),
    /// Answers [`Action::Play`]: the card played into the trick.
    Play(Card),
}

/// Why a [`Game::apply`] (or [`Game::start_next_hand`]) call was rejected.
///
/// A well-behaved caller that only ever submits the [`Decision`] matching the
/// current [`Action`] — and only plays cards drawn from the supplied legal set —
/// will never see one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyError {
    /// The submitted [`Decision`] does not match the action the engine is
    /// currently waiting for.
    WrongPhase,
    /// A discarded or played card is not held by the acting seat.
    NotInHand(Card),
    /// The played card is held but breaks the obligation to follow suit.
    MustFollowSuit(Card),
    /// The named trump suit is illegal — the turned-down up-card suit cannot be
    /// chosen in the second round.
    IllegalCall(Suit),
    /// The seat passed when "stick the dealer" forbids it.
    MustNotPass,
    /// A new hand was requested but the match is already decided.
    GameOver,
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplyError::WrongPhase => write!(f, "decision does not match the current game phase"),
            ApplyError::NotInHand(c) => write!(f, "card {c} is not in the acting seat's hand"),
            ApplyError::MustFollowSuit(c) => write!(f, "playing {c} fails to follow the led suit"),
            ApplyError::IllegalCall(s) => write!(f, "{s} may not be named (it was turned down)"),
            ApplyError::MustNotPass => write!(f, "the dealer is stuck and may not pass"),
            ApplyError::GameOver => write!(f, "the match is already over"),
        }
    }
}

impl std::error::Error for ApplyError {}

/// The authoritative state of a Euchre match.
///
/// See the [module documentation](self) for the intended request/response loop.
#[derive(Debug, Clone)]
pub struct Game {
    rules: GameRules,
    target_score: u8,
    /// Absolute team scores, indexed by [`TeamId`] (team 0 = North/South).
    scores: [u8; 2],
    /// The fixed player dealing the current hand.
    dealer: Player,

    // Per-hand state, reset by `deal`.
    /// Each fixed player's hand, indexed by [`Player`].
    hands: [Vec<Card>; 4],
    up_card: Card,
    kitty: Vec<Card>,
    /// The card the dealer buried after taking the up-card, remembered so the
    /// dealer's own view can show it.
    discarded: Option<Card>,
    contract: Option<Contract>,
    current_trick: Trick,
    completed_tricks: Vec<(Trick, Seat)>,
    /// Tricks taken so far this hand, indexed by [`TeamId`].
    team_tricks: [u8; 2],
    phase: Phase,
    last_result: Option<HandRecord>,

    /// The winning team, once decided.
    winner: Option<TeamId>,
}

/// The engine's absolute record of how a hand was scored, kept so a per-seat
/// [`HandResult`] can be built for any viewpoint (see [`Game::hand_result`]).
#[derive(Debug, Clone, Copy)]
enum HandRecord {
    /// Every seat passed; no points were awarded.
    PassedOut,
    /// Trump was named and the hand played out.
    Played {
        maker_tricks: u8,
        /// The fixed team that scored.
        scoring_team: TeamId,
        points: u8,
    },
}

impl Game {
    /// Starts a match and deals the first hand from `deck`.
    ///
    /// `deck` must be the full 24-card Euchre deck in the (already shuffled)
    /// order to deal; see [`Card::deck`]. The first five cards go to one seat,
    /// the next five to the next, and so on, with card 21 turned up and the rest
    /// buried in the kitty.
    pub fn new(config: GameConfig, deck: [Card; 24]) -> Self {
        let mut game = Game {
            rules: config.rules,
            target_score: config.target_score,
            scores: [0, 0],
            dealer: config.first_dealer % 4,
            hands: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            up_card: deck[HAND_SIZE * 4],
            kitty: Vec::new(),
            discarded: None,
            contract: None,
            current_trick: Trick::new(),
            completed_tricks: Vec::new(),
            team_tricks: [0, 0],
            phase: Phase::BidRound1 { turn: Seat::First },
            last_result: None,
            winner: None,
        };
        game.deal(deck);
        game
    }

    /// Rotates the deal one seat clockwise and deals a fresh hand from `deck`.
    ///
    /// Valid only once the previous hand is complete and the match is not yet
    /// decided; otherwise returns [`ApplyError`].
    pub fn start_next_hand(&mut self, deck: [Card; 24]) -> Result<(), ApplyError> {
        if self.winner.is_some() {
            return Err(ApplyError::GameOver);
        }
        if self.phase != Phase::HandComplete {
            return Err(ApplyError::WrongPhase);
        }
        self.dealer = (self.dealer + 1) % 4;
        self.deal(deck);
        Ok(())
    }

    /// Deals `deck` into hands/up-card/kitty and resets per-hand state, opening
    /// the first bidding round to the seat on the dealer's left.
    fn deal(&mut self, deck: [Card; 24]) {
        debug_assert!(
            is_full_deck(&deck),
            "deck must be the 24 unique Euchre cards"
        );
        for (i, chunk) in deck[..HAND_SIZE * 4].chunks(HAND_SIZE).enumerate() {
            // The dealer will pick up the up-card, so leave room for a sixth.
            let mut hand = Vec::with_capacity(HAND_SIZE + 1);
            hand.extend_from_slice(chunk);
            self.hands[i] = hand;
        }
        self.up_card = deck[HAND_SIZE * 4];
        self.kitty = deck[HAND_SIZE * 4 + 1..].to_vec();
        self.discarded = None;
        self.contract = None;
        self.current_trick = Trick::new();
        self.completed_tricks = Vec::with_capacity(TRICKS_PER_HAND);
        self.team_tricks = [0, 0];
        self.last_result = None;
        self.phase = Phase::BidRound1 { turn: Seat::First };
    }

    /// Reports what the engine needs next in order to advance.
    ///
    /// This is a pure query; call it as often as you like. The returned
    /// [`Action`] names the acting seat (except for [`Action::HandComplete`]).
    pub fn next_action(&self) -> Action {
        match self.phase {
            Phase::BidRound1 { turn } => Action::BidUpcard {
                seat: turn,
                up_card: self.up_card,
            },
            Phase::BidRound2 { turn } => Action::BidCall {
                seat: turn,
                turned_down: self.up_card.suit,
                may_pass: !(self.rules.stick_the_dealer && turn == Seat::Dealer),
            },
            Phase::Discard => Action::Discard {
                seat: Seat::Dealer,
                up_card: self.up_card,
            },
            Phase::Play { turn } => Action::Play {
                seat: turn,
                legal: self.legal_plays(turn),
            },
            Phase::HandComplete => Action::HandComplete {
                dealer: self.dealer,
            },
        }
    }

    /// Applies a player's [`Decision`], advancing the state machine.
    ///
    /// The decision must match the variant of the current [`Action`]; mismatches
    /// and illegal moves are rejected with an [`ApplyError`] and leave the game
    /// unchanged.
    pub fn apply(&mut self, decision: Decision) -> Result<(), ApplyError> {
        match (self.phase, decision) {
            (Phase::BidRound1 { turn }, Decision::Upcard(bid)) => self.apply_upcard(turn, bid),
            (Phase::BidRound2 { turn }, Decision::Call(bid)) => self.apply_call(turn, bid),
            (Phase::Discard, Decision::Discard(card)) => self.apply_discard(card),
            (Phase::Play { turn }, Decision::Play(card)) => self.apply_play(turn, card),
            _ => Err(ApplyError::WrongPhase),
        }
    }

    /// Builds the read-only [`GameView`] that `seat` is entitled to see right
    /// now: its own hand plus all public state.
    pub fn view(&self, seat: Seat) -> GameView<'_> {
        let team = self.team_of(seat);
        GameView {
            seat,
            up_card: self.up_card,
            hand: &self.hands[self.player_at(seat)],
            contract: self.contract,
            // The buried card is private to the dealer who chose it.
            discarded: if seat == Seat::Dealer {
                self.discarded
            } else {
                None
            },
            current_trick: &self.current_trick,
            completed_tricks: &self.completed_tricks,
            scores: Scores {
                us: self.scores[team],
                them: self.scores[1 - team],
            },
            rules: self.rules,
        }
    }

    // --- Accessors -----------------------------------------------------------

    /// The fixed player who dealt the current hand (`0` = North … `3` = West).
    pub fn dealer(&self) -> Player {
        self.dealer
    }

    /// The cumulative match score by fixed team (index 0 = North/South).
    pub fn scores(&self) -> [u8; 2] {
        self.scores
    }

    /// The fixed player occupying `seat` this hand. `Seat::Dealer` is the
    /// [dealer](Game::dealer); the others follow clockwise from the dealer's
    /// left.
    pub fn player_at(&self, seat: Seat) -> Player {
        (self.dealer + seat_offset(seat)) % 4
    }

    /// The dealer-relative seat a fixed `player` occupies this hand — the inverse
    /// of [`Game::player_at`].
    pub fn seat_of(&self, player: Player) -> Seat {
        seat_from_offset((player + 4 - self.dealer) % 4)
    }

    /// How the just-completed hand scored, told from `seat`'s point of view (its
    /// team's net points). Valid once the engine reports
    /// [`Action::HandComplete`].
    pub fn hand_result(&self, seat: Seat) -> HandResult {
        match self.last_result.expect("a completed hand has a result") {
            HandRecord::PassedOut => HandResult::PassedOut,
            HandRecord::Played {
                maker_tricks,
                scoring_team,
                points,
            } => {
                let signed = if scoring_team == self.team_of(seat) {
                    points as i8
                } else {
                    -(points as i8)
                };
                HandResult::Played(HandScore {
                    maker_tricks,
                    points_awarded: signed,
                })
            }
        }
    }

    /// The contract for the current hand, once trump has been named.
    pub fn contract(&self) -> Option<Contract> {
        self.contract
    }

    /// The turned-up card for the current hand.
    pub fn up_card(&self) -> Card {
        self.up_card
    }

    /// The trick currently in progress.
    pub fn current_trick(&self) -> &Trick {
        &self.current_trick
    }

    /// The completed tricks of the current hand, oldest first, each paired with
    /// the seat that won it.
    pub fn completed_tricks(&self) -> &[(Trick, Seat)] {
        &self.completed_tricks
    }

    /// The house rules in effect.
    pub fn rules(&self) -> GameRules {
        self.rules
    }

    /// The score required to win the match.
    pub fn target_score(&self) -> u8 {
        self.target_score
    }

    /// The cards currently held by `seat`.
    pub fn hand(&self, seat: Seat) -> &[Card] {
        &self.hands[self.player_at(seat)]
    }

    /// Whether the match has been decided.
    pub fn is_over(&self) -> bool {
        self.winner.is_some()
    }

    /// The winning team (index 0 = North/South), once the match is
    /// [over](Game::is_over).
    pub fn winner(&self) -> Option<TeamId> {
        self.winner
    }

    // --- Bidding -------------------------------------------------------------

    fn apply_upcard(&mut self, turn: Seat, bid: UpcardBid) -> Result<(), ApplyError> {
        match bid {
            UpcardBid::Pass => {
                self.phase = if turn == Seat::Dealer {
                    // Everyone passed the up-card; turn it down and open round two.
                    Phase::BidRound2 { turn: Seat::First }
                } else {
                    Phase::BidRound1 { turn: turn.next() }
                };
                Ok(())
            }
            UpcardBid::OrderUp { alone } => {
                self.contract = Some(Contract {
                    trump: self.up_card.suit,
                    maker: turn,
                    alone,
                });
                self.begin_after_order_up();
                Ok(())
            }
        }
    }

    /// After the up-card is ordered up, the dealer takes it into hand and must
    /// discard — unless the dealer is sitting out a loner, in which case the
    /// pickup is moot and we go straight to the play.
    fn begin_after_order_up(&mut self) {
        if self.sitting_out() == Some(Seat::Dealer) {
            self.start_play();
        } else {
            self.hands[self.dealer].push(self.up_card);
            self.phase = Phase::Discard;
        }
    }

    fn apply_call(&mut self, turn: Seat, bid: CallBid) -> Result<(), ApplyError> {
        match bid {
            CallBid::Pass => {
                if turn == Seat::Dealer {
                    if self.rules.stick_the_dealer {
                        return Err(ApplyError::MustNotPass);
                    }
                    // Every seat passed both rounds: throw the hand in.
                    self.last_result = Some(HandRecord::PassedOut);
                    self.phase = Phase::HandComplete;
                } else {
                    self.phase = Phase::BidRound2 { turn: turn.next() };
                }
                Ok(())
            }
            CallBid::Call { suit, alone } => {
                if suit == self.up_card.suit {
                    return Err(ApplyError::IllegalCall(suit));
                }
                self.contract = Some(Contract {
                    trump: suit,
                    maker: turn,
                    alone,
                });
                self.start_play();
                Ok(())
            }
        }
    }

    fn apply_discard(&mut self, card: Card) -> Result<(), ApplyError> {
        let hand = &mut self.hands[self.dealer];
        let pos = hand
            .iter()
            .position(|&c| c == card)
            .ok_or(ApplyError::NotInHand(card))?;
        let discarded = hand.remove(pos);
        self.discarded = Some(discarded);
        self.kitty.push(discarded);
        self.start_play();
        Ok(())
    }

    // --- Play ----------------------------------------------------------------

    fn start_play(&mut self) {
        self.phase = Phase::Play {
            turn: self.first_leader(),
        };
    }

    /// The first seat to lead: the dealer's left, skipping a seat that is
    /// sitting out because its partner went alone.
    fn first_leader(&self) -> Seat {
        let candidate = Seat::First;
        if self.sitting_out() == Some(candidate) {
            candidate.next()
        } else {
            candidate
        }
    }

    fn apply_play(&mut self, turn: Seat, card: Card) -> Result<(), ApplyError> {
        if !self.hands[self.player_at(turn)].contains(&card) {
            return Err(ApplyError::NotInHand(card));
        }
        if !self.legal_plays(turn).contains(&card) {
            return Err(ApplyError::MustFollowSuit(card));
        }

        let hand = &mut self.hands[self.player_at(turn)];
        let pos = hand.iter().position(|&c| c == card).expect("card is held");
        hand.remove(pos);
        self.current_trick.push(Play { seat: turn, card });

        if self.current_trick.len() == self.active_count() {
            self.resolve_trick();
        } else {
            self.phase = Phase::Play {
                turn: self.next_active(turn),
            };
        }
        Ok(())
    }

    /// The cards `seat` may legally play to the current trick: those following
    /// the led suit if it holds any, otherwise its whole hand.
    fn legal_plays(&self, seat: Seat) -> Vec<Card> {
        let trump = self.contract.expect("contract set during play").trump;
        let hand = &self.hands[self.player_at(seat)];
        match self.current_trick.led_suit(trump) {
            None => hand.clone(),
            Some(led) => {
                let following: Vec<Card> = hand
                    .iter()
                    .copied()
                    .filter(|c| c.effective_suit(trump) == led)
                    .collect();
                if following.is_empty() {
                    hand.clone()
                } else {
                    following
                }
            }
        }
    }

    fn resolve_trick(&mut self) {
        let trump = self.contract.expect("contract set during play").trump;
        let winner = self
            .current_trick
            .winner(trump)
            .expect("a completed trick has a winner");
        self.team_tricks[self.team_of(winner)] += 1;
        let finished = std::mem::replace(&mut self.current_trick, Trick::new());
        self.completed_tricks.push((finished, winner));

        if self.completed_tricks.len() == TRICKS_PER_HAND {
            self.score_hand();
        } else {
            // The trick winner leads the next trick.
            self.phase = Phase::Play { turn: winner };
        }
    }

    fn score_hand(&mut self) {
        let contract = self.contract.expect("a played hand has a contract");
        let makers = self.team_of(contract.maker);
        let maker_tricks = self.team_tricks[makers];
        let alone = contract.alone;
        let euchred = maker_tricks < 3;
        let march = maker_tricks == TRICKS_PER_HAND as u8;

        let (team, points) = if euchred {
            (1 - makers, 2)
        } else if march {
            (makers, if alone { 4 } else { 2 })
        } else {
            (makers, 1)
        };

        self.scores[team] += points;

        self.last_result = Some(HandRecord::Played {
            maker_tricks,
            scoring_team: team,
            points,
        });

        if self.scores[team] >= self.target_score {
            self.winner = Some(team);
        }
        self.phase = Phase::HandComplete;
    }

    // --- Seat helpers --------------------------------------------------------

    /// The fixed team [`seat`](Seat) belongs to this hand (index 0 = North/South).
    fn team_of(&self, seat: Seat) -> TeamId {
        team_of_player(self.player_at(seat))
    }

    /// The seat sitting out the current hand, if a loner was declared.
    fn sitting_out(&self) -> Option<Seat> {
        self.contract.and_then(|c| c.sitting_out())
    }

    /// The number of seats actually playing this hand (three under a loner).
    fn active_count(&self) -> usize {
        if self.sitting_out().is_some() { 3 } else { 4 }
    }

    /// The next seat clockwise that is not sitting out.
    fn next_active(&self, seat: Seat) -> Seat {
        let candidate = seat.next();
        if self.sitting_out() == Some(candidate) {
            candidate.next()
        } else {
            candidate
        }
    }
}

/// The number of clockwise steps from the dealer to `seat`: the dealer's left
/// (`First`) is 1, around to the dealer itself (`Dealer`) at 0.
fn seat_offset(seat: Seat) -> usize {
    match seat {
        Seat::First => 1,
        Seat::Second => 2,
        Seat::Third => 3,
        Seat::Dealer => 0,
    }
}

/// The inverse of [`seat_offset`]: the seat that many steps clockwise from the
/// dealer.
fn seat_from_offset(offset: usize) -> Seat {
    match offset % 4 {
        0 => Seat::Dealer,
        1 => Seat::First,
        2 => Seat::Second,
        _ => Seat::Third,
    }
}

/// Whether `deck` is exactly the 24 distinct Euchre cards (debug check only).
fn is_full_deck(deck: &[Card; 24]) -> bool {
    let mut seen = deck.to_vec();
    seen.sort_by_key(|c| (c.suit, c.rank));
    seen.dedup();
    seen.len() == 24
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::Rank;

    /// The deck in canonical (unshuffled) order. With `first_dealer = 0` (North),
    /// North is the dealer (`Seat::Dealer`) and the deal is deterministic:
    ///   Dealer/North:  9♣ 10♣ J♣ Q♣ K♣   First/East:   A♣ 9♦ 10♦ J♦ Q♦
    ///   Second/South:  K♦ A♦ 9♥ 10♥ J♥   Third/West:   Q♥ K♥ A♥ 9♠ 10♠
    ///   up-card: J♠   kitty: Q♠ K♠ A♠
    fn ordered_deck() -> [Card; 24] {
        Card::deck().try_into().expect("24 cards")
    }

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    /// Drives a game from the current point with a "first legal" policy until the
    /// hand completes. Bids are always passes (so a hand only gets played if
    /// someone is forced or chooses to call before this is invoked); discards and
    /// plays take the first available card.
    fn play_out_first_legal(game: &mut Game) {
        loop {
            match game.next_action() {
                Action::BidUpcard { .. } => game.apply(Decision::Upcard(UpcardBid::Pass)).unwrap(),
                Action::BidCall { .. } => game.apply(Decision::Call(CallBid::Pass)).unwrap(),
                Action::Discard { seat, .. } => {
                    let c = game.hand(seat)[0];
                    game.apply(Decision::Discard(c)).unwrap();
                }
                Action::Play { legal, .. } => {
                    game.apply(Decision::Play(legal[0])).unwrap();
                }
                Action::HandComplete { .. } => return,
            }
        }
    }

    #[test]
    fn deal_distributes_24_cards() {
        let game = Game::new(GameConfig::default(), ordered_deck());
        for seat in Seat::ALL {
            assert_eq!(game.hand(seat).len(), HAND_SIZE);
        }
        assert_eq!(game.up_card(), card(Rank::Jack, Suit::Spades));
        // First action asks the seat to the dealer's left.
        match game.next_action() {
            Action::BidUpcard { seat, up_card } => {
                assert_eq!(seat, Seat::First);
                assert_eq!(up_card, card(Rank::Jack, Suit::Spades));
            }
            other => panic!("expected BidUpcard, got {other:?}"),
        }
    }

    #[test]
    fn passing_round_one_opens_round_two_with_turned_down_suit() {
        let mut game = Game::new(GameConfig::default(), ordered_deck());
        // All four seats pass the up-card.
        for _ in 0..4 {
            game.apply(Decision::Upcard(UpcardBid::Pass)).unwrap();
        }
        match game.next_action() {
            Action::BidCall {
                seat,
                turned_down,
                may_pass,
            } => {
                assert_eq!(seat, Seat::First);
                assert_eq!(turned_down, Suit::Spades);
                assert!(may_pass);
            }
            other => panic!("expected BidCall, got {other:?}"),
        }
    }

    #[test]
    fn all_passes_throws_the_hand_in() {
        let mut game = Game::new(GameConfig::default(), ordered_deck());
        for _ in 0..4 {
            game.apply(Decision::Upcard(UpcardBid::Pass)).unwrap();
        }
        for _ in 0..4 {
            game.apply(Decision::Call(CallBid::Pass)).unwrap();
        }
        match game.next_action() {
            Action::HandComplete { dealer } => {
                assert_eq!(dealer, 0); // North dealt
            }
            other => panic!("expected HandComplete, got {other:?}"),
        }
        assert_eq!(game.hand_result(Seat::First), HandResult::PassedOut);
        assert_eq!(game.scores(), [0, 0]);
        assert!(!game.is_over());

        // The deal rotates to the next player on a redeal.
        game.start_next_hand(ordered_deck()).unwrap();
        assert_eq!(game.dealer(), 1); // East deals next
    }

    #[test]
    fn stick_the_dealer_forbids_a_final_pass() {
        let config = GameConfig {
            rules: GameRules {
                stick_the_dealer: true,
            },
            ..GameConfig::default()
        };
        let mut game = Game::new(config, ordered_deck());
        for _ in 0..4 {
            game.apply(Decision::Upcard(UpcardBid::Pass)).unwrap();
        }
        // First, Second, Third pass the second round.
        for _ in 0..3 {
            game.apply(Decision::Call(CallBid::Pass)).unwrap();
        }
        // The dealer is now stuck.
        match game.next_action() {
            Action::BidCall { seat, may_pass, .. } => {
                assert_eq!(seat, Seat::Dealer);
                assert!(!may_pass);
            }
            other => panic!("expected BidCall, got {other:?}"),
        }
        assert_eq!(
            game.apply(Decision::Call(CallBid::Pass)),
            Err(ApplyError::MustNotPass)
        );
        // Naming a legal suit succeeds and moves to play.
        game.apply(Decision::Call(CallBid::Call {
            suit: Suit::Hearts,
            alone: false,
        }))
        .unwrap();
        assert_eq!(game.contract().unwrap().trump, Suit::Hearts);
        assert!(matches!(game.next_action(), Action::Play { .. }));
    }

    #[test]
    fn ordering_up_makes_the_dealer_pick_up_and_discard() {
        let mut game = Game::new(GameConfig::default(), ordered_deck());
        // First (East) orders up the J♠.
        game.apply(Decision::Upcard(UpcardBid::OrderUp { alone: false }))
            .unwrap();
        let contract = game.contract().expect("contract set");
        assert_eq!(contract.trump, Suit::Spades);
        assert_eq!(contract.maker, Seat::First);

        // The dealer now holds six cards and must discard.
        assert_eq!(game.hand(Seat::Dealer).len(), HAND_SIZE + 1);
        let discard = match game.next_action() {
            Action::Discard { seat, .. } => {
                assert_eq!(seat, Seat::Dealer);
                game.hand(Seat::Dealer)[0]
            }
            other => panic!("expected Discard, got {other:?}"),
        };
        game.apply(Decision::Discard(discard)).unwrap();
        assert_eq!(game.hand(Seat::Dealer).len(), HAND_SIZE);
        // The dealer's own view remembers the buried card.
        assert_eq!(game.view(Seat::Dealer).discarded, Some(discard));
        assert_eq!(game.view(Seat::First).discarded, None);

        // Play opens with the seat to the dealer's left.
        match game.next_action() {
            Action::Play { seat, legal } => {
                assert_eq!(seat, Seat::First);
                assert_eq!(legal.len(), HAND_SIZE); // leading: whole hand is legal
            }
            other => panic!("expected Play, got {other:?}"),
        }
    }

    #[test]
    fn loner_seats_out_the_partner_and_skips_them_in_play() {
        let mut game = Game::new(GameConfig::default(), ordered_deck());
        // First orders up alone; First's partner (Third) sits out.
        game.apply(Decision::Upcard(UpcardBid::OrderUp { alone: true }))
            .unwrap();
        // Dealer still discards (the dealer is not the one sitting out).
        let discard = game.hand(Seat::Dealer)[0];
        game.apply(Decision::Discard(discard)).unwrap();

        let seats_played: std::collections::HashSet<Seat> = {
            let mut seen = std::collections::HashSet::new();
            // Play the first trick and record who acts.
            for _ in 0..3 {
                match game.next_action() {
                    Action::Play { seat, legal } => {
                        seen.insert(seat);
                        game.apply(Decision::Play(legal[0])).unwrap();
                    }
                    other => panic!("expected Play, got {other:?}"),
                }
            }
            seen
        };
        // Exactly three seats play, and Third (the loner's partner) is not one.
        assert_eq!(seats_played.len(), 3);
        assert!(!seats_played.contains(&Seat::Third));
    }

    #[test]
    fn a_played_hand_awards_points_and_records_tricks() {
        let mut game = Game::new(GameConfig::default(), ordered_deck());
        // First (East, team East/West) orders up, then play out first-legal.
        game.apply(Decision::Upcard(UpcardBid::OrderUp { alone: false }))
            .unwrap();
        play_out_first_legal(&mut game);

        // Seen from the maker's seat: positive points if the makers made it,
        // negative if they were euchred.
        let HandResult::Played(score) = game.hand_result(Seat::First) else {
            panic!("expected a played hand");
        };
        let defender_tricks = TRICKS_PER_HAND as u8 - score.maker_tricks;
        if score.euchred() {
            assert!(score.maker_tricks < 3);
            assert_eq!(score.points_awarded, -2);
        } else if score.march() {
            assert_eq!(score.points_awarded, 2);
        } else {
            assert_eq!(score.points_awarded, 1);
        }
        // Only one hand was played, so the whole table score is this hand's
        // points, handed to exactly one team.
        let scores = game.scores();
        assert_eq!(scores[0] + scores[1], score.points_awarded.unsigned_abs());
        assert_eq!(score.maker_tricks + defender_tricks, TRICKS_PER_HAND as u8);
        assert_eq!(game.completed_tricks().len(), TRICKS_PER_HAND);
    }

    #[test]
    fn player_and_seat_mappings_are_inverse() {
        let mut game = Game::new(GameConfig::default(), ordered_deck());
        assert_eq!(game.dealer(), 0);
        for player in 0..4 {
            assert_eq!(game.player_at(game.seat_of(player)), player);
        }
        // The dealer is always `Seat::Dealer`; First sits to its left.
        assert_eq!(game.player_at(Seat::Dealer), 0);
        assert_eq!(game.player_at(Seat::First), 1);

        // After a redeal the mapping rotates with the deal.
        game.apply(Decision::Upcard(UpcardBid::Pass)).unwrap();
        for _ in 0..3 {
            game.apply(Decision::Upcard(UpcardBid::Pass)).unwrap();
        }
        for _ in 0..4 {
            game.apply(Decision::Call(CallBid::Pass)).unwrap();
        }
        game.start_next_hand(ordered_deck()).unwrap();
        assert_eq!(game.dealer(), 1);
        assert_eq!(game.player_at(Seat::Dealer), 1);
        assert_eq!(game.player_at(Seat::First), 2);
    }

    #[test]
    fn wrong_decision_for_phase_is_rejected() {
        let mut game = Game::new(GameConfig::default(), ordered_deck());
        // We are in bidding, not play.
        let some_card = game.hand(Seat::First)[0];
        assert_eq!(
            game.apply(Decision::Play(some_card)),
            Err(ApplyError::WrongPhase)
        );
    }

    #[test]
    fn calling_the_turned_down_suit_is_illegal() {
        let mut game = Game::new(GameConfig::default(), ordered_deck());
        for _ in 0..4 {
            game.apply(Decision::Upcard(UpcardBid::Pass)).unwrap();
        }
        // Spades was turned down; it cannot be named.
        assert_eq!(
            game.apply(Decision::Call(CallBid::Call {
                suit: Suit::Spades,
                alone: false,
            })),
            Err(ApplyError::IllegalCall(Suit::Spades))
        );
    }
}
