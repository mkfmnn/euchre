//! An agent driven by a person through a text interface.
//!
//! [`HumanAgent`] turns each engine callback into a prompt and a menu of legal
//! choices, delegating the actual input/output to a [`Prompter`]. The default
//! [`TerminalPrompter`] reads from standard input and writes to standard
//! output, but any `Prompter` works — which keeps the agent testable with a
//! scripted prompter and reusable behind a GUI or network front end.

use std::io::{self, BufRead, Write};

use euchre_interface::{Agent, Bid, CallBid, Card, GameView, HandResult, Play, Suit, UpcardBid};

/// Presents choices to a human and returns their selection.
///
/// The engine never blocks on a `Prompter`; it is the agent's job to keep
/// asking until it gets a legal answer. Implementations must return an index
/// **in range** for [`choose`](Prompter::choose) — the agent trusts the value
/// and uses it to index the option list.
pub trait Prompter {
    /// Displays informational text (board state, results) with no input.
    fn show(&mut self, message: &str);

    /// Presents `options` under `prompt` and returns the chosen index, which
    /// must satisfy `0 <= index < options.len()`.
    fn choose(&mut self, prompt: &str, options: &[String]) -> usize;
}

/// A [`Prompter`] backed by standard input and output.
///
/// It prints a numbered menu and reads a line, re-prompting until it parses a
/// valid selection. On end-of-input it falls back to the first option so an
/// automated or piped session terminates instead of looping forever.
pub struct TerminalPrompter {
    input: Box<dyn BufRead>,
    output: Box<dyn Write>,
}

impl TerminalPrompter {
    /// A prompter wired to the process's real stdin and stdout.
    pub fn stdio() -> Self {
        TerminalPrompter {
            input: Box::new(io::BufReader::new(io::stdin())),
            output: Box::new(io::stdout()),
        }
    }

    /// A prompter reading from and writing to arbitrary streams, handy for
    /// scripting or testing.
    pub fn with_streams(input: Box<dyn BufRead>, output: Box<dyn Write>) -> Self {
        TerminalPrompter { input, output }
    }
}

impl Default for TerminalPrompter {
    fn default() -> Self {
        TerminalPrompter::stdio()
    }
}

impl Prompter for TerminalPrompter {
    fn show(&mut self, message: &str) {
        let _ = writeln!(self.output, "{message}");
        let _ = self.output.flush();
    }

    fn choose(&mut self, prompt: &str, options: &[String]) -> usize {
        loop {
            let _ = writeln!(self.output, "\n{prompt}");
            for (i, opt) in options.iter().enumerate() {
                let _ = writeln!(self.output, "  {}) {}", i + 1, opt);
            }
            let _ = write!(self.output, "> ");
            let _ = self.output.flush();

            let mut line = String::new();
            match self.input.read_line(&mut line) {
                Ok(0) => return 0, // EOF: pick the first (safe) option.
                Ok(_) => {}
                Err(_) => return 0,
            }
            match line.trim().parse::<usize>() {
                Ok(n) if (1..=options.len()).contains(&n) => return n - 1,
                _ => {
                    let _ = writeln!(
                        self.output,
                        "Please enter a number between 1 and {}.",
                        options.len()
                    );
                }
            }
        }
    }
}

/// An [`Agent`] that asks a human for every decision through a [`Prompter`].
pub struct HumanAgent<P: Prompter> {
    prompter: P,
}

impl HumanAgent<TerminalPrompter> {
    /// A human agent wired to the terminal (stdin/stdout).
    pub fn terminal() -> Self {
        HumanAgent {
            prompter: TerminalPrompter::stdio(),
        }
    }
}

impl<P: Prompter> HumanAgent<P> {
    /// Wraps an arbitrary prompter.
    pub fn new(prompter: P) -> Self {
        HumanAgent { prompter }
    }

    /// Borrows the underlying prompter.
    pub fn prompter(&mut self) -> &mut P {
        &mut self.prompter
    }
}

/// Renders the public board state plus the agent's own hand as a banner shown
/// before a decision.
fn render_context(view: &GameView<'_>) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "── You are {:?} ({:?}).  Dealer: {:?}.  Score  N/S {} – E/W {}\n",
        view.seat,
        view.seat.team(),
        view.dealer,
        view.scores.north_south,
        view.scores.east_west,
    ));
    if let Some(contract) = view.contract {
        s.push_str(&format!(
            "   Trump: {}{}  (made by {:?}{})\n",
            contract.trump,
            contract.trump.symbol(),
            contract.maker,
            if contract.alone { ", alone" } else { "" },
        ));
    }
    if !view.current_trick.is_empty() {
        s.push_str("   Trick so far: ");
        s.push_str(&render_plays(view.current_trick.plays()));
        s.push('\n');
    }
    s.push_str(&format!("   Your hand: {}", render_cards(view.hand)));
    s
}

fn render_cards(cards: &[Card]) -> String {
    cards
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("  ")
}

fn render_plays(plays: &[Play]) -> String {
    plays
        .iter()
        .map(|p| format!("{:?}:{}", p.seat, p.card))
        .collect::<Vec<_>>()
        .join("  ")
}

impl<P: Prompter> Agent for HumanAgent<P> {
    fn bid_upcard(&mut self, view: &GameView<'_>, up_card: Card) -> UpcardBid {
        self.prompter.show(&render_context(view));
        let prompt = format!(
            "Up-card is {} ({}{}). Order it up as trump?",
            up_card,
            up_card.suit,
            up_card.suit.symbol(),
        );
        let options = vec![
            "Pass".to_string(),
            format!("Order up {}", up_card.suit),
            format!("Order up {} — go alone", up_card.suit),
        ];
        match self.prompter.choose(&prompt, &options) {
            1 => UpcardBid::OrderUp(Bid::WithPartner),
            2 => UpcardBid::OrderUp(Bid::Alone),
            _ => UpcardBid::Pass,
        }
    }

    fn bid_call(&mut self, view: &GameView<'_>, turned_down: Suit) -> CallBid {
        self.prompter.show(&render_context(view));
        let callable: Vec<Suit> = Suit::ALL
            .into_iter()
            .filter(|&s| s != turned_down)
            .collect();

        let mut options = vec!["Pass".to_string()];
        for &suit in &callable {
            options.push(format!("Call {suit}"));
            options.push(format!("Call {suit} — go alone"));
        }
        let prompt = format!("Up-card ({turned_down}) was turned down. Name a trump suit?",);
        let choice = self.prompter.choose(&prompt, &options);
        if choice == 0 {
            return CallBid::Pass;
        }
        // Options after "Pass" come in (call, alone) pairs per callable suit.
        let idx = choice - 1;
        let suit = callable[idx / 2];
        let bid = if idx % 2 == 1 {
            Bid::Alone
        } else {
            Bid::WithPartner
        };
        CallBid::Call { suit, bid }
    }

    fn discard(&mut self, view: &GameView<'_>) -> Card {
        self.prompter.show(&render_context(view));
        let options: Vec<String> = view.hand.iter().map(|c| c.to_string()).collect();
        let choice = self
            .prompter
            .choose("You picked up the up-card. Discard one:", &options);
        view.hand[choice]
    }

    fn play_card(&mut self, view: &GameView<'_>, legal: &[Card]) -> Card {
        self.prompter.show(&render_context(view));
        let options: Vec<String> = legal.iter().map(|c| c.to_string()).collect();
        let choice = self.prompter.choose("Your turn — play a card:", &options);
        legal[choice]
    }

    fn observe_hand_end(&mut self, view: &GameView<'_>, result: &HandResult) {
        let (team, points) = result.points_awarded;
        let summary = if result.euchred {
            format!("{:?} were euchred!", result.makers)
        } else if result.march {
            format!(
                "{:?} swept all five{}!",
                result.makers,
                if result.alone { " (alone)" } else { "" }
            )
        } else {
            format!("{:?} took {} tricks.", result.makers, result.maker_tricks)
        };
        self.prompter.show(&format!(
            "── Hand over. {summary}  {team:?} +{points}.  \
             Score now N/S {} – E/W {}.",
            view.scores.north_south, view.scores.east_west,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euchre_interface::{Rank, Seat, Trick};

    /// A prompter that replays a fixed script of choices.
    struct ScriptedPrompter {
        answers: std::collections::VecDeque<usize>,
        shown: Vec<String>,
    }

    impl ScriptedPrompter {
        fn new(answers: impl IntoIterator<Item = usize>) -> Self {
            ScriptedPrompter {
                answers: answers.into_iter().collect(),
                shown: Vec::new(),
            }
        }
    }

    impl Prompter for ScriptedPrompter {
        fn show(&mut self, message: &str) {
            self.shown.push(message.to_string());
        }
        fn choose(&mut self, _prompt: &str, options: &[String]) -> usize {
            let idx = self
                .answers
                .pop_front()
                .expect("ran out of scripted answers");
            assert!(idx < options.len(), "scripted answer {idx} out of range");
            idx
        }
    }

    fn view_with<'a>(hand: &'a [Card], trick: &'a Trick) -> GameView<'a> {
        GameView {
            seat: Seat::North,
            dealer: Seat::North,
            hand,
            contract: None,
            current_trick: trick,
            completed_tricks: &[],
            scores: Default::default(),
        }
    }

    #[test]
    fn order_up_alone_maps_correctly() {
        let mut agent = HumanAgent::new(ScriptedPrompter::new([2]));
        let hand = [Card::new(Rank::Ace, Suit::Hearts)];
        let trick = Trick::new();
        let view = view_with(&hand, &trick);
        let up = Card::new(Rank::King, Suit::Spades);
        assert_eq!(agent.bid_upcard(&view, up), UpcardBid::OrderUp(Bid::Alone));
    }

    #[test]
    fn call_choice_maps_to_suit_and_alone() {
        // turned_down = Clubs, so callable = [Diamonds, Hearts, Spades].
        // Options: 0 Pass, 1 Call D, 2 Call D alone, 3 Call H, 4 Call H alone,
        //          5 Call S, 6 Call S alone. Choose index 4 → Hearts, alone.
        let mut agent = HumanAgent::new(ScriptedPrompter::new([4]));
        let hand = [Card::new(Rank::Ace, Suit::Hearts)];
        let trick = Trick::new();
        let view = view_with(&hand, &trick);
        match agent.bid_call(&view, Suit::Clubs) {
            CallBid::Call { suit, bid } => {
                assert_eq!(suit, Suit::Hearts);
                assert_eq!(bid, Bid::Alone);
            }
            CallBid::Pass => panic!("expected a call"),
        }
    }

    #[test]
    fn play_returns_chosen_legal_card() {
        let mut agent = HumanAgent::new(ScriptedPrompter::new([1]));
        let hand = [
            Card::new(Rank::Nine, Suit::Hearts),
            Card::new(Rank::Ace, Suit::Hearts),
        ];
        let trick = Trick::new();
        let mut view = view_with(&hand, &trick);
        view.contract = Some(euchre_interface::Contract {
            trump: Suit::Spades,
            maker: Seat::North,
            alone: false,
        });
        let chosen = agent.play_card(&view, &hand);
        assert_eq!(chosen, Card::new(Rank::Ace, Suit::Hearts));
    }

    #[test]
    fn terminal_prompter_parses_input() {
        let input = b"2\n".to_vec();
        let output: Vec<u8> = Vec::new();
        let mut p = TerminalPrompter::with_streams(
            Box::new(io::Cursor::new(input)),
            Box::new(io::Cursor::new(output)),
        );
        let opts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(p.choose("pick", &opts), 1);
    }
}
