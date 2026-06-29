//! End-to-end test for assist mode: a human seat playing against three bots on
//! an assist-enabled server receives a well-formed `SUGGEST` before each of its
//! own turns, and an assist-disabled server sends none.

use std::time::Duration;

use euchre_engine::GameConfig;
use euchre_interface::Card;
use euchre_server::protocol::{
    ClientMsg, Player, ScoredAction, SeatInfo, SeatRequest, ServerMsg, SuggestedAction, TurnHint,
};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

#[tokio::test]
async fn assist_on_sends_a_suggestion_before_each_turn() {
    let counts = run_match(true).await;
    assert!(
        counts.suggests > 0,
        "assist mode produced no SUGGEST messages"
    );
    // Every one of our own decisions should have been preceded by a suggestion.
    assert_eq!(
        counts.suggests, counts.our_turns,
        "expected one SUGGEST per turn ({} turns, {} suggests)",
        counts.our_turns, counts.suggests
    );
}

#[tokio::test]
async fn assist_off_sends_no_suggestions() {
    let counts = run_match(false).await;
    assert_eq!(
        counts.suggests, 0,
        "assist disabled but {} SUGGEST messages arrived",
        counts.suggests
    );
    assert!(counts.our_turns > 0, "the human seat never acted");
}

#[derive(Default)]
struct Counts {
    /// SUGGEST messages received for our seat.
    suggests: usize,
    /// AWAITING messages for our seat (decisions we had to make).
    our_turns: usize,
}

/// Plays one match as a human seat against three bots and returns how many
/// `SUGGEST` messages and own-turn `AWAITING`s were seen. Asserts each
/// suggestion is well-formed (its recommended move is the top-scored option).
async fn run_match(assist: bool) -> Counts {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        euchre_server::serve(listener, GameConfig::default(), assist)
            .await
            .unwrap();
    });

    tokio::time::timeout(Duration::from_secs(60), play(addr))
        .await
        .expect("a match completes within the timeout")
}

async fn play(addr: std::net::SocketAddr) -> Counts {
    let url = format!("ws://{addr}/ws");
    let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut tx, mut rx) = ws.split();

    send(
        &mut tx,
        &ClientMsg::Hello {
            name: "tester".into(),
            table: None,
        },
    )
    .await;

    let mut my_seat: Option<Player> = None;
    let mut hand: Vec<Card> = Vec::new();
    let mut counts = Counts::default();

    while let Some(item) = rx.next().await {
        let Message::Text(text) = item.unwrap() else {
            continue;
        };
        match serde_json::from_str::<ServerMsg>(&text).unwrap() {
            ServerMsg::TableState {
                your_seat, seats, ..
            } => {
                my_seat = your_seat;
                match your_seat {
                    None => {
                        let seat = (0..4)
                            .find(|&i| matches!(seats[i as usize], SeatInfo::Empty))
                            .expect("an open seat");
                        send(
                            &mut tx,
                            &ClientMsg::Seat {
                                seat,
                                player: SeatRequest::Me,
                            },
                        )
                        .await;
                    }
                    Some(_) => {
                        for seat in (0..4).filter(|&i| matches!(seats[i as usize], SeatInfo::Empty))
                        {
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
            ServerMsg::Deal { hand: h, .. } => hand = h,
            ServerMsg::GameOver { .. } => return counts,
            ServerMsg::Suggest {
                player,
                recommended,
                scores,
            } => {
                assert_eq!(Some(player), my_seat, "suggestion for someone else's seat");
                assert_well_formed(&recommended, &scores);
                counts.suggests += 1;
            }
            ServerMsg::Awaiting {
                player,
                hint,
                legal,
            } => {
                if Some(player) != my_seat {
                    continue;
                }
                counts.our_turns += 1;
                let reply = match hint {
                    TurnHint::Bid { .. } => ClientMsg::Pass,
                    TurnHint::Discard => ClientMsg::Discard { card: hand[0] },
                    TurnHint::Play { .. } => ClientMsg::Play {
                        card: legal.expect("active player gets legal plays")[0],
                    },
                };
                send(&mut tx, &reply).await;
            }
            _ => {}
        }
    }

    panic!("connection closed before the match ended");
}

/// The recommended move must be one of the scored options and carry the highest
/// score — the contract the assist UI relies on to outline the right control.
fn assert_well_formed(recommended: &SuggestedAction, scores: &[ScoredAction]) {
    assert!(!scores.is_empty(), "a suggestion with no scored options");
    let best = scores
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap();
    assert_eq!(
        &best.action, recommended,
        "recommended move is not the top-scored option"
    );
}

async fn send<S>(tx: &mut S, msg: &ClientMsg)
where
    S: SinkExt<Message> + Unpin,
{
    let json = serde_json::to_string(msg).unwrap();
    let _ = tx.send(Message::Text(json)).await;
}
