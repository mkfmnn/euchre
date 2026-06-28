//! A minimal terminal websocket client — the network analogue of the engine's
//! terminal driver prompts. Connects to a running `euchre-server`, joins the
//! table, prints the event stream, and lets you act on your turn from stdin.
//!
//! ```text
//! cargo run -p euchre-server                 # in one terminal
//! cargo run -p euchre-server --example cli_client   # in another
//! ```
//!
//! Set `EUCHRE_URL` (default `ws://127.0.0.1:8080/ws`), `EUCHRE_NAME`
//! (default `human`), and `EUCHRE_TABLE` (a code to join; omit to create a new
//! table) to override the target, display name, and table.
//!
//! Seating is automatic: the client takes the first open seat, then fills the
//! rest of the table with bots so a match starts on its own.

use std::io::Write;

use euchre_server::protocol::{
    ClientMsg, PublicAction, SeatInfo, SeatRequest, ServerMsg, TurnHint,
};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

#[tokio::main]
async fn main() {
    let url = std::env::var("EUCHRE_URL").unwrap_or_else(|_| "ws://127.0.0.1:8080/ws".into());
    let name = std::env::var("EUCHRE_NAME").unwrap_or_else(|_| "human".into());
    let table = std::env::var("EUCHRE_TABLE").ok();

    let (ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("connect to server");
    let (mut tx, mut rx) = ws.split();

    send(&mut tx, &ClientMsg::Hello { name, table }).await;

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
            ServerMsg::TableState {
                table,
                your_seat,
                seats,
            } => {
                println!("Table {table}. Seats:");
                for (i, s) in seats.iter().enumerate() {
                    let who = match s {
                        SeatInfo::Empty => "(empty)".to_string(),
                        SeatInfo::Bot { name } => format!("{name} (bot)"),
                        SeatInfo::Human { name } => format!("{name} (human)"),
                    };
                    println!("  {i}: {who}");
                }
                // Take a seat if we have none, then fill the rest with bots.
                match your_seat {
                    None => {
                        if let Some(seat) = first_empty(&seats) {
                            send(
                                &mut tx,
                                &ClientMsg::Seat {
                                    seat,
                                    player: SeatRequest::Me,
                                },
                            )
                            .await;
                        }
                    }
                    Some(_) => {
                        for seat in empties(&seats) {
                            send(
                                &mut tx,
                                &ClientMsg::Seat {
                                    seat,
                                    player: SeatRequest::Bot,
                                },
                            )
                            .await;
                        }
                    }
                }
            }
            ServerMsg::StartGame { first_dealer } => {
                println!("Game starting — first dealer {first_dealer:?}.");
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
                let team = if winner == 0 {
                    "North/South"
                } else {
                    "East/West"
                };
                println!("GAME OVER — {team} wins ({}-{}).", scores[0], scores[1]);
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

/// The seat positions that are currently empty.
fn empties(seats: &[SeatInfo; 4]) -> Vec<u8> {
    (0..4)
        .filter(|&i| matches!(seats[i as usize], SeatInfo::Empty))
        .collect()
}

/// The first empty seat position, if any.
fn first_empty(seats: &[SeatInfo; 4]) -> Option<u8> {
    empties(seats).first().copied()
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
