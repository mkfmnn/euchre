//! The terminal **driver**: a game loop that ties the [`Game`] core to players.
//!
//! A [`Driver`] owns a [`Game`] and four [`Player`]s — each either an AI
//! [`Agent`] or a human at the terminal — and runs a full match to completion.
//! At every decision point it asks the core [what is needed](Game::next_action),
//! routes the request to the appropriate player, and feeds the answer back in.
//!
//! ## Players
//!
//! Drivers support the two configurations the project calls for:
//!
//! * **One human, three bots** — put a [`Player::Human`] in one seat and a
//!   [`Player::Bot`] in the other three.
//! * **Four bots** — put a [`Player::Bot`] in every seat (typically with
//!   [`Driver::headless`], which needs no input stream).
//!
//! This crate intentionally ships no agents; concrete bots live in a separate
//! crate. The driver only needs `&mut dyn Agent`, so any implementation works.
//!
//! ## Output
//!
//! [`Verbosity`] selects how much the driver narrates:
//!
//! * [`Verbosity::Full`] — every bid, play and trick, enough for a human to
//!   follow what the other seats are doing.
//! * [`Verbosity::Hand`] — one summary line per completed hand, by who scored.
//! * [`Verbosity::Silent`] — nothing; callers read the result from the returned
//!   [`Outcome`].
//!
//! Prompts to a human are always written, regardless of verbosity, since the
//! human cannot act without them.

use std::io::{self, BufRead, BufReader, Empty, Write};

use euchre_interface::{Agent, CallBid, Card, HandResult, Seat, Suit, Trick, UpcardBid};
use rand::SeedableRng;
use rand::rngs::ChaCha12Rng;

use crate::game::{Action, Decision, Game, GameConfig};

/// How much the driver narrates a match to its output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// Narrate every bid, play, and trick. Required for a human to follow play.
    Full,
    /// Print one summary line per completed hand.
    Hand,
    /// Print nothing.
    Silent,
}

/// Who occupies a seat: an AI agent, or a human typing at the terminal.
pub enum Player<'a> {
    /// A human player driven through the input/output streams.
    Human,
    /// An AI agent. The driver borrows it mutably for the match.
    Bot(&'a mut dyn Agent),
}

/// The result of running a match to completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    /// The team that reached the target score (index 0 = North/South).
    pub winner: usize,
    /// The final cumulative score by team (index 0 = North/South).
    pub scores: [u8; 2],
    /// How many hands were dealt, including any thrown in.
    pub hands_played: u32,
}

/// Runs a full Euchre match, mediating between the [`Game`] core and the
/// players. See the [module documentation](self) for an overview.
pub struct Driver<'a, R: BufRead, W: Write> {
    game: Game,
    players: [Player<'a>; 4],
    verbosity: Verbosity,
    input: R,
    output: W,
    rng: ChaCha12Rng,
}

impl<'a, W: Write> Driver<'a, BufReader<Empty>, W> {
    /// Builds a driver for four autonomous agents, with no input stream.
    ///
    /// Convenient for simulations and four-bot matches. If a [`Player::Human`]
    /// is supplied here it will only ever read end-of-input and fall back to
    /// safe defaults; use [`Driver::new`] with a real input stream for humans.
    pub fn headless(
        config: GameConfig,
        players: [Player<'a>; 4],
        verbosity: Verbosity,
        output: W,
    ) -> Self {
        Driver::new(
            config,
            players,
            verbosity,
            BufReader::new(io::empty()),
            output,
        )
    }
}

impl<'a, R: BufRead, W: Write> Driver<'a, R, W> {
    /// Builds a driver over explicit input/output streams, seeding the shuffler
    /// from system entropy.
    ///
    /// For a human at a real terminal, pass `std::io::stdin().lock()` and
    /// `std::io::stdout()`.
    pub fn new(
        config: GameConfig,
        players: [Player<'a>; 4],
        verbosity: Verbosity,
        input: R,
        output: W,
    ) -> Self {
        Driver::with_rng(
            config,
            players,
            verbosity,
            input,
            output,
            ChaCha12Rng::from_rng(&mut rand::rng()),
        )
    }

    /// Like [`Driver::new`] but with a fixed shuffler seed, for reproducible
    /// matches (used in tests and for replaying a deal).
    pub fn with_seed(
        config: GameConfig,
        players: [Player<'a>; 4],
        verbosity: Verbosity,
        input: R,
        output: W,
        seed: u64,
    ) -> Self {
        Driver::with_rng(
            config,
            players,
            verbosity,
            input,
            output,
            ChaCha12Rng::seed_from_u64(seed),
        )
    }

    /// Shared constructor: deals the first hand from `rng` and assembles the
    /// driver. The ChaCha12 shuffler is for game variety, not cryptographic
    /// security; seeding it explicitly (see [`Driver::with_seed`]) makes a whole
    /// match reproducible.
    fn with_rng(
        config: GameConfig,
        players: [Player<'a>; 4],
        verbosity: Verbosity,
        input: R,
        output: W,
        mut rng: ChaCha12Rng,
    ) -> Self {
        let game = Game::new(config, deal(&mut rng));
        Driver {
            game,
            players,
            verbosity,
            input,
            output,
            rng,
        }
    }

    /// Runs the match to completion and returns its [`Outcome`].
    ///
    /// Any error is an I/O failure writing narration or reading a human's input.
    pub fn run(&mut self) -> io::Result<Outcome> {
        let mut hands_played: u32 = 1;
        self.announce_hand()?;

        loop {
            match self.game.next_action() {
                Action::BidUpcard { seat, up_card } => {
                    let decision = self.decide_upcard(seat, up_card)?;
                    self.game.apply(decision).expect("legal up-card bid");
                    if let Decision::Upcard(bid) = decision {
                        self.narrate_upcard(seat, up_card, bid)?;
                    }
                }
                Action::BidCall {
                    seat,
                    turned_down,
                    may_pass,
                } => {
                    let decision = self.decide_call(seat, turned_down, may_pass)?;
                    self.game.apply(decision).expect("legal call bid");
                    if let Decision::Call(bid) = decision {
                        self.narrate_call(seat, bid)?;
                    }
                }
                Action::Discard { seat, .. } => {
                    let decision = self.decide_discard(seat)?;
                    self.game.apply(decision).expect("legal discard");
                    if self.verbosity == Verbosity::Full {
                        writeln!(
                            self.output,
                            "{} discards.",
                            player_name(self.game.player_at(seat))
                        )?;
                    }
                }
                Action::Play { seat, legal } => {
                    let decision = self.decide_play(seat, &legal)?;
                    let Decision::Play(card) = decision else {
                        unreachable!("decide_play yields a play");
                    };
                    let before = self.game.completed_tricks().len();
                    self.game
                        .apply(decision)
                        .expect("agent returned a legal card");
                    self.narrate_play(seat, card, before)?;
                }
                Action::HandComplete { .. } => {
                    self.notify_hand_end();
                    self.narrate_hand_result()?;
                    if self.game.is_over() {
                        let winner = self.game.winner().expect("decided match has a winner");
                        self.narrate_game_over(winner)?;
                        return Ok(Outcome {
                            winner,
                            scores: self.game.scores(),
                            hands_played,
                        });
                    }
                    let deck = deal(&mut self.rng);
                    self.game
                        .start_next_hand(deck)
                        .expect("ready for next hand");
                    hands_played += 1;
                    self.announce_hand()?;
                }
            }
        }
    }

    // --- Decision routing ----------------------------------------------------
    //
    // Each `decide_*` borrows the player slot mutably and the game immutably (to
    // build the view) — disjoint fields, so no conflict. Human branches delegate
    // to free `prompt_*` functions that take the needed streams explicitly, to
    // keep from borrowing all of `self` while the view is alive.

    fn decide_upcard(&mut self, seat: Seat, up_card: Card) -> io::Result<Decision> {
        let player = self.game.player_at(seat);
        match &mut self.players[player] {
            Player::Bot(agent) => {
                let view = self.game.view(seat);
                Ok(Decision::Upcard(agent.bid_upcard(&view)))
            }
            Player::Human => {
                prompt_upcard(&self.game, seat, up_card, &mut self.input, &mut self.output)
            }
        }
    }

    fn decide_call(
        &mut self,
        seat: Seat,
        turned_down: Suit,
        may_pass: bool,
    ) -> io::Result<Decision> {
        let player = self.game.player_at(seat);
        match &mut self.players[player] {
            Player::Bot(agent) => {
                let view = self.game.view(seat);
                Ok(Decision::Call(agent.bid_call(&view)))
            }
            Player::Human => prompt_call(
                &self.game,
                seat,
                turned_down,
                may_pass,
                &mut self.input,
                &mut self.output,
            ),
        }
    }

    fn decide_discard(&mut self, seat: Seat) -> io::Result<Decision> {
        let player = self.game.player_at(seat);
        match &mut self.players[player] {
            Player::Bot(agent) => {
                let view = self.game.view(seat);
                Ok(Decision::Discard(agent.discard(&view)))
            }
            Player::Human => prompt_discard(&self.game, seat, &mut self.input, &mut self.output),
        }
    }

    fn decide_play(&mut self, seat: Seat, legal: &[Card]) -> io::Result<Decision> {
        let player = self.game.player_at(seat);
        match &mut self.players[player] {
            Player::Bot(agent) => {
                let view = self.game.view(seat);
                Ok(Decision::Play(agent.play_card(&view, legal)))
            }
            Player::Human => {
                prompt_play(&self.game, seat, legal, &mut self.input, &mut self.output)
            }
        }
    }

    /// Lets every agent observe the completed hand, from its own seat's point of
    /// view, so stateful bots can learn.
    fn notify_hand_end(&mut self) {
        for seat in Seat::ALL {
            let player = self.game.player_at(seat);
            if let Player::Bot(agent) = &mut self.players[player] {
                let view = self.game.view(seat);
                let result = self.game.hand_result(seat);
                agent.observe_hand_end(&view, &result);
            }
        }
    }

    // --- Narration -----------------------------------------------------------

    fn announce_hand(&mut self) -> io::Result<()> {
        if self.verbosity != Verbosity::Full {
            return Ok(());
        }
        let s = self.game.scores();
        writeln!(self.output)?;
        writeln!(
            self.output,
            "--- New hand. Dealer: {}. Up card: {}. Score: N/S {}, E/W {}.",
            player_name(self.game.dealer()),
            self.game.up_card(),
            s[0],
            s[1],
        )
    }

    fn narrate_upcard(&mut self, seat: Seat, up_card: Card, bid: UpcardBid) -> io::Result<()> {
        if self.verbosity != Verbosity::Full {
            return Ok(());
        }
        let actor = player_name(self.game.player_at(seat));
        match bid {
            UpcardBid::Pass => writeln!(self.output, "{actor} passes.")?,
            UpcardBid::OrderUp { alone } => {
                writeln!(
                    self.output,
                    "{} orders up the {}{}. {} is trump.",
                    actor,
                    up_card,
                    alone_suffix(alone),
                    up_card.suit,
                )?;
                let dealer_sits_out =
                    self.game.contract().and_then(|c| c.sitting_out()) == Some(Seat::Dealer);
                if !dealer_sits_out {
                    writeln!(
                        self.output,
                        "{} picks up the {}.",
                        player_name(self.game.dealer()),
                        up_card
                    )?;
                }
            }
        }
        Ok(())
    }

    fn narrate_call(&mut self, seat: Seat, bid: CallBid) -> io::Result<()> {
        if self.verbosity != Verbosity::Full {
            return Ok(());
        }
        let actor = player_name(self.game.player_at(seat));
        match bid {
            CallBid::Pass => writeln!(self.output, "{actor} passes.")?,
            CallBid::Call { suit, alone } => writeln!(
                self.output,
                "{} names {} as trump{}.",
                actor,
                suit,
                alone_suffix(alone),
            )?,
        }
        Ok(())
    }

    fn narrate_play(&mut self, seat: Seat, card: Card, tricks_before: usize) -> io::Result<()> {
        if self.verbosity != Verbosity::Full {
            return Ok(());
        }
        writeln!(
            self.output,
            "  {} plays {}",
            player_name(self.game.player_at(seat)),
            card
        )?;
        let tricks = self.game.completed_tricks();
        if tricks.len() > tricks_before {
            let (_, winner) = tricks[tricks.len() - 1];
            writeln!(
                self.output,
                "  -> {} wins trick {}.",
                player_name(self.game.player_at(winner)),
                tricks.len(),
            )?;
        }
        Ok(())
    }

    fn narrate_hand_result(&mut self) -> io::Result<()> {
        if self.verbosity == Verbosity::Silent {
            return Ok(());
        }
        writeln!(self.output, "{}", self.format_hand_result())
    }

    /// One human-readable line summarizing the just-completed hand, framed by who
    /// scored. Read from the god's-eye game state rather than any one seat's view.
    fn format_hand_result(&self) -> String {
        let s = self.game.scores();
        let running = format!("[N/S {}, E/W {}]", s[0], s[1]);
        // North's seat anchors the relative result to the North/South team.
        let HandResult::Played(score) = self.game.hand_result(self.game.seat_of(0)) else {
            return format!("Hand thrown in — no one scored. {running}");
        };
        let points = score.points_awarded.unsigned_abs();
        let scoring_team = if score.points_awarded >= 0 { 0 } else { 1 };
        let contract = self.game.contract().expect("a played hand has a contract");
        let maker_team = self.game.player_at(contract.maker) % 2;
        let detail = if score.euchred() {
            format!(
                "{} euchre {}",
                team_name(scoring_team),
                team_name(maker_team)
            )
        } else if score.march() {
            if contract.alone {
                format!("{} march alone", team_name(scoring_team))
            } else {
                format!("{} march", team_name(scoring_team))
            }
        } else {
            format!(
                "{} make it ({} tricks)",
                team_name(scoring_team),
                score.maker_tricks
            )
        };
        format!("{detail}: +{points}. {running}")
    }

    fn narrate_game_over(&mut self, winner: usize) -> io::Result<()> {
        if self.verbosity == Verbosity::Silent {
            return Ok(());
        }
        let s = self.game.scores();
        writeln!(
            self.output,
            "{} wins the match! Final score: N/S {}, E/W {}.",
            team_name(winner),
            s[0],
            s[1],
        )
    }
}

// --- Human prompts -----------------------------------------------------------
//
// Free functions so callers can hand them just the streams and a `&Game`,
// sidestepping a whole-`self` borrow while a `GameView` is alive.

fn prompt_upcard<R: BufRead, W: Write>(
    game: &Game,
    seat: Seat,
    up_card: Card,
    input: &mut R,
    output: &mut W,
) -> io::Result<Decision> {
    write_hand(output, "Your hand", game.hand(seat))?;
    if !ask_yes_no(input, output, &format!("Order up the {up_card}? [y/N]: "))? {
        return Ok(Decision::Upcard(UpcardBid::Pass));
    }
    let alone = ask_alone(input, output)?;
    Ok(Decision::Upcard(UpcardBid::OrderUp { alone }))
}

fn prompt_call<R: BufRead, W: Write>(
    game: &Game,
    seat: Seat,
    turned_down: Suit,
    may_pass: bool,
    input: &mut R,
    output: &mut W,
) -> io::Result<Decision> {
    write_hand(output, "Your hand", game.hand(seat))?;
    let choices: Vec<Suit> = Suit::ALL
        .into_iter()
        .filter(|&s| s != turned_down)
        .collect();
    writeln!(output, "{turned_down} was turned down. Name trump:")?;
    for (i, suit) in choices.iter().enumerate() {
        writeln!(output, "  {i}) {suit}")?;
    }
    if may_pass {
        writeln!(output, "  p) pass")?;
    }
    loop {
        write!(output, "Choice: ")?;
        output.flush()?;
        let Some(line) = read_line(input) else {
            // End of input: pass if allowed, else take the first legal suit.
            return Ok(Decision::Call(if may_pass {
                CallBid::Pass
            } else {
                CallBid::Call {
                    suit: choices[0],
                    alone: false,
                }
            }));
        };
        let lower = line.to_lowercase();
        if may_pass && (lower == "p" || lower == "pass") {
            return Ok(Decision::Call(CallBid::Pass));
        }
        if let Ok(idx) = lower.parse::<usize>()
            && let Some(&suit) = choices.get(idx)
        {
            let alone = ask_alone(input, output)?;
            return Ok(Decision::Call(CallBid::Call { suit, alone }));
        }
        writeln!(output, "Invalid choice.")?;
    }
}

fn prompt_discard<R: BufRead, W: Write>(
    game: &Game,
    seat: Seat,
    input: &mut R,
    output: &mut W,
) -> io::Result<Decision> {
    let hand = game.hand(seat);
    writeln!(output, "You took the up card. Choose a card to discard:")?;
    write_numbered(output, hand)?;
    let idx = read_index(input, output, hand.len())?;
    Ok(Decision::Discard(hand[idx]))
}

fn prompt_play<R: BufRead, W: Write>(
    game: &Game,
    seat: Seat,
    legal: &[Card],
    input: &mut R,
    output: &mut W,
) -> io::Result<Decision> {
    let trick = game.current_trick();
    if !trick.is_empty() {
        write_trick(output, game, trick)?;
    }
    write_hand(output, "Your hand", game.hand(seat))?;
    writeln!(output, "Legal plays:")?;
    write_numbered(output, legal)?;
    let idx = read_index(input, output, legal.len())?;
    Ok(Decision::Play(legal[idx]))
}

/// Asks the maker whether to play alone.
fn ask_alone<R: BufRead, W: Write>(input: &mut R, output: &mut W) -> io::Result<bool> {
    ask_yes_no(input, output, "Go alone? [y/N]: ")
}

/// Prompts for a yes/no answer, re-asking on garbage. End-of-input and a bare
/// Enter both mean "no".
fn ask_yes_no<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    prompt: &str,
) -> io::Result<bool> {
    loop {
        write!(output, "{prompt}")?;
        output.flush()?;
        match read_line(input) {
            None => return Ok(false),
            Some(line) => match line.to_lowercase().as_str() {
                "y" | "yes" => return Ok(true),
                "" | "n" | "no" => return Ok(false),
                _ => writeln!(output, "Please answer y or n.")?,
            },
        }
    }
}

/// Prompts for an index in `0..len`, re-asking on garbage. End-of-input picks 0.
fn read_index<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    len: usize,
) -> io::Result<usize> {
    loop {
        write!(output, "Choice [0-{}]: ", len - 1)?;
        output.flush()?;
        match read_line(input) {
            None => return Ok(0),
            Some(line) => match line.parse::<usize>() {
                Ok(idx) if idx < len => return Ok(idx),
                _ => writeln!(output, "Invalid choice.")?,
            },
        }
    }
}

/// Reads and trims one line, returning `None` at end of input.
fn read_line<R: BufRead>(input: &mut R) -> Option<String> {
    let mut buf = String::new();
    match input.read_line(&mut buf) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(buf.trim().to_string()),
    }
}

fn write_hand<W: Write>(output: &mut W, label: &str, cards: &[Card]) -> io::Result<()> {
    write!(output, "{label}:")?;
    for card in cards {
        write!(output, " {card}")?;
    }
    writeln!(output)
}

fn write_numbered<W: Write>(output: &mut W, cards: &[Card]) -> io::Result<()> {
    for (i, card) in cards.iter().enumerate() {
        writeln!(output, "  {i}) {card}")?;
    }
    Ok(())
}

fn write_trick<W: Write>(output: &mut W, game: &Game, trick: &Trick) -> io::Result<()> {
    write!(output, "Current trick:")?;
    for play in trick.plays() {
        write!(
            output,
            " {}={}",
            player_name(game.player_at(play.seat)),
            play.card
        )?;
    }
    writeln!(output)
}

// --- Formatting --------------------------------------------------------------

fn alone_suffix(alone: bool) -> &'static str {
    if alone { " and goes alone" } else { "" }
}

/// The name of a fixed player (`0` = North … `3` = West).
fn player_name(player: usize) -> &'static str {
    ["North", "East", "South", "West"][player]
}

/// The name of a fixed team (`0` = North/South, `1` = East/West).
fn team_name(team: usize) -> &'static str {
    ["North/South", "East/West"][team]
}

// --- Shuffling ---------------------------------------------------------------

/// A freshly shuffled 24-card deck, drawn from the driver's ChaCha12 generator.
///
/// Thin wrapper over [`crate::shuffle::deal`] so the driver and a server deal
/// the same way.
fn deal(rng: &mut ChaCha12Rng) -> [Card; 24] {
    crate::shuffle::deal(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::GameView;

    /// A minimal scaffold agent for exercising the driver. It is *not* a real
    /// strategy — production bots live in their own crate — but it always makes
    /// a legal move, and always calls trump in round two so hands get played.
    struct FirstLegal;

    impl Agent for FirstLegal {
        fn bid_upcard(&mut self, _view: &GameView<'_>) -> UpcardBid {
            UpcardBid::Pass
        }

        fn bid_call(&mut self, view: &GameView<'_>) -> CallBid {
            let turned_down = view.up_card.suit;
            let suit = Suit::ALL.into_iter().find(|&s| s != turned_down).unwrap();
            CallBid::Call { suit, alone: false }
        }

        fn discard(&mut self, view: &GameView<'_>) -> Card {
            view.hand[0]
        }

        fn play_card(&mut self, _view: &GameView<'_>, legal: &[Card]) -> Card {
            legal[0]
        }
    }

    fn four_bots<'a>(bots: &'a mut [FirstLegal; 4]) -> [Player<'a>; 4] {
        let [a, b, c, d] = bots;
        [
            Player::Bot(a),
            Player::Bot(b),
            Player::Bot(c),
            Player::Bot(d),
        ]
    }

    #[test]
    fn four_bots_play_a_full_match() {
        let mut bots = [FirstLegal, FirstLegal, FirstLegal, FirstLegal];
        let mut out = Vec::new();
        let outcome = Driver::with_seed(
            GameConfig::default(),
            four_bots(&mut bots),
            Verbosity::Silent,
            BufReader::new(io::empty()),
            &mut out,
            42,
        )
        .run()
        .unwrap();

        assert!(out.is_empty(), "silent mode prints nothing");
        assert!(outcome.scores[outcome.winner] >= GameConfig::default().target_score);
        assert!(outcome.hands_played >= 1);
    }

    #[test]
    fn hand_verbosity_prints_one_line_per_hand() {
        let mut bots = [FirstLegal, FirstLegal, FirstLegal, FirstLegal];
        let mut out = Vec::new();
        let outcome = Driver::with_seed(
            GameConfig::default(),
            four_bots(&mut bots),
            Verbosity::Hand,
            BufReader::new(io::empty()),
            &mut out,
            7,
        )
        .run()
        .unwrap();

        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        // One line per hand played, plus the final game-over line.
        assert_eq!(lines.len() as u32, outcome.hands_played + 1);
        assert!(lines.last().unwrap().contains("wins the match!"));
    }

    #[test]
    fn full_verbosity_narrates_and_announces_the_winner() {
        let mut bots = [FirstLegal, FirstLegal, FirstLegal, FirstLegal];
        let mut out = Vec::new();
        Driver::with_seed(
            GameConfig::default(),
            four_bots(&mut bots),
            Verbosity::Full,
            BufReader::new(io::empty()),
            &mut out,
            123,
        )
        .run()
        .unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("New hand"));
        assert!(text.contains("is trump") || text.contains("names"));
        assert!(text.contains("wins the match!"));
    }

    #[test]
    fn a_human_seat_can_play_from_scripted_input() {
        // Three bots and one human (North). Feed enough "0\n" lines that the
        // human always orders up nothing / plays the first legal card; the
        // y/n and index prompts all accept these or fall back safely.
        let mut bots = [FirstLegal, FirstLegal, FirstLegal];
        let [b1, b2, b3] = &mut bots;
        let players = [
            Player::Human,
            Player::Bot(b1),
            Player::Bot(b2),
            Player::Bot(b3),
        ];
        // A long script of newlines/zeros: "no" to ordering up & going alone,
        // and index 0 for any discard/play prompt.
        let script = "n\n".repeat(50) + &"0\n".repeat(400);
        let outcome = Driver::with_seed(
            GameConfig::default(),
            players,
            Verbosity::Silent,
            BufReader::new(io::Cursor::new(script)),
            io::sink(),
            5,
        )
        .run()
        .unwrap();
        assert!(outcome.scores[outcome.winner] >= GameConfig::default().target_score);
    }

    #[test]
    fn shuffle_is_deterministic_for_a_fixed_seed() {
        let mut a = ChaCha12Rng::seed_from_u64(99);
        let mut b = ChaCha12Rng::seed_from_u64(99);
        assert_eq!(deal(&mut a), deal(&mut b));
        let mut c = ChaCha12Rng::seed_from_u64(100);
        assert_ne!(deal(&mut a), deal(&mut c));
    }
}
