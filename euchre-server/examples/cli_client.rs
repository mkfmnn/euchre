//! A minimal terminal websocket client — the network analogue of the engine's
//! terminal driver prompts. Connects to a running `euchre-server`, joins the
//! table, prints the event stream, and lets you act on your turn from stdin.
//!
//! ```text
//! cargo run -p euchre-server                 # in one terminal
//! cargo run -p euchre-server --example cli_client   # in another
//! ```
//!
//! Set `EUCHRE_URL` (default `ws://127.0.0.1:8080/ws`) and `EUCHRE_NAME`
//! (default `human`) to override the target and display name.

use std::io::Write;

use euchre_server::protocol::{ClientMsg, PublicAction, ServerMsg, TurnHint};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

#[tokio::main]
async fn main() {
    let url = std::env::var("EUCHRE_URL").unwrap_or_else(|_| "ws://127.0.0.1:8080/ws".into());
    let name = std::env::var("EUCHRE_NAME").unwrap_or_else(|_| "human".into());

    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("connect to server");
    let (mut tx, mut rx) = ws.split();

    send(&mut tx, &ClientMsg::Hello { name, seat: None }).await;

    while let Some(item) = rx.next().await {
        let Ok(Message::Text(text)) = item else {
            if matches!(item, Ok(Message::Close(_)) | Err(_)) {
                break;
            }
            continue;
        };
        let msg: ServerMsg = match serde_json::from_str(&text) {
            Ok(msg) => msg,
            Err(e) => {
                eprintln!("(unparsable server message: {e})");
                continue;
            }
        };
        match msg {
            ServerMsg::Joined {
                players, your_seat, ..
            } => {
                println!("Joined as {your_seat:?}. Table:");
                for p in players {
                    let kind = if p.bot { "bot" } else { "human" };
                    println!("  {:?}: {} ({kind})", p.seat, p.name);
                }
            }
            ServerMsg::Sync { view } => {
                println!("Your hand: {}", cards(&view.hand));
            }
            ServerMsg::Deal {
                dealer,
                hand,
                up_card,
            } => {
                println!("\n--- New hand. Dealer {dealer:?}, up card {up_card}. ---");
                println!("Your hand: {}", cards(&hand));
            }
            ServerMsg::Update { player, action } => match action {
                PublicAction::Bid { suit, alone } => {
                    let solo = if alone { " (alone)" } else { "" };
                    println!("{player:?} named {suit} trump{solo}.");
                }
                PublicAction::Pass => println!("{player:?} passed."),
                PublicAction::Discard => println!("{player:?} discarded."),
                PublicAction::Play { card } => println!("{player:?} played {card}."),
            },
            ServerMsg::TrickWon { player } => println!("  -> {player:?} won the trick."),
            ServerMsg::HandComplete { result } => println!("Hand complete: {result:?}"),
            ServerMsg::GameOver { winner, scores } => {
                println!(
                    "GAME OVER — {winner:?} wins ({}-{}).",
                    scores.north_south, scores.east_west
                );
            }
            ServerMsg::Error { message } => eprintln!("(error: {message})"),
            ServerMsg::Awaiting {
                player,
                hint,
                legal,
            } => {
                // Only act when it is our turn; the active player's Awaiting is
                // the only one carrying `legal` for a play.
                if let Some(reply) = prompt(&hint, legal.as_deref()) {
                    send(&mut tx, &reply).await;
                } else {
                    println!("Waiting for {player:?}...");
                }
            }
        }
    }
}

/// Prompts for this seat's move, or returns `None` if it is not our turn.
fn prompt(hint: &TurnHint, legal: Option<&[euchre_interface::Card]>) -> Option<ClientMsg> {
    match hint {
        TurnHint::Bid { up, may_pass } => {
            let round = if *up {
                "order up the up-card"
            } else {
                "name a suit"
            };
            let opts = if *may_pass {
                "[pass/<suit>/alone]"
            } else {
                "[<suit>/alone]"
            };
            let line = ask(&format!("Your bid ({round}) {opts}: "))?;
            parse_bid(&line, *up, *may_pass)
        }
        TurnHint::Discard => {
            let line = ask("Discard which card (e.g. 9C)? ")?;
            euchre_interface::Card::from_code(line.trim()).map(|card| ClientMsg::Discard { card })
        }
        TurnHint::Play { lead } => {
            // Only the active seat receives `legal`; absence means it isn't our turn.
            let legal = legal?;
            println!("Lead suit: {lead:?}. Legal plays: {}", cards(legal));
            let line = ask("Play which card? ")?;
            euchre_interface::Card::from_code(line.trim()).map(|card| ClientMsg::Play { card })
        }
    }
}

fn parse_bid(line: &str, up: bool, may_pass: bool) -> Option<ClientMsg> {
    let line = line.trim().to_lowercase();
    if line.is_empty() || line == "pass" || line == "p" {
        return may_pass.then_some(ClientMsg::Pass);
    }
    let alone = line.contains("alone");
    let suit = if up {
        // Round one orders up the up-card; the suit is implied, any token works.
        first_suit(&line).unwrap_or(euchre_interface::Suit::Spades)
    } else {
        first_suit(&line)?
    };
    Some(ClientMsg::Bid { suit, alone })
}

fn first_suit(s: &str) -> Option<euchre_interface::Suit> {
    s.chars().find_map(|c| match c {
        'c' => Some(euchre_interface::Suit::Clubs),
        'd' => Some(euchre_interface::Suit::Diamonds),
        'h' => Some(euchre_interface::Suit::Hearts),
        's' => Some(euchre_interface::Suit::Spades),
        _ => None,
    })
}

fn ask(prompt: &str) -> Option<String> {
    print!("{prompt}");
    std::io::stdout().flush().ok()?;
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line),
    }
}

fn cards(cards: &[euchre_interface::Card]) -> String {
    cards
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

async fn send<S>(tx: &mut S, msg: &ClientMsg)
where
    S: SinkExt<Message> + Unpin,
{
    let json = serde_json::to_string(msg).expect("serialize client message");
    let _ = tx.send(Message::Text(json)).await;
}
