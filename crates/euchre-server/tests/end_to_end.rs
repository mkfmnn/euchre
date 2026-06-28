//! End-to-end test: a real websocket client plays one seat against three
//! server-side bots and a full match runs to completion over the wire.
//!
//! This exercises the whole path — serialize → connection → room → engine →
//! broadcast — that the unit tests in the individual crates cannot.

use std::time::Duration;

use euchre_engine::GameConfig;
use euchre_interface::Card;
use euchre_server::protocol::{ClientMsg, Player, ServerMsg, TurnHint};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

#[tokio::test]
async fn one_human_and_three_bots_play_a_full_match() {
    // Bind first so the address is live before we connect, then serve.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        euchre_server::serve(listener, GameConfig::default())
            .await
            .unwrap();
    });

    let outcome = tokio::time::timeout(Duration::from_secs(60), play_a_match(addr))
        .await
        .expect("a match completes within the timeout");

    let (winner, ns, ew) = outcome;
    // The winner must actually have reached the target score.
    let winning_score = if winner == 0 { ns } else { ew };
    assert!(
        winning_score >= GameConfig::default().target_score,
        "winning team {winner} should have reached target (scores: N/S {ns}, E/W {ew})"
    );
}

/// Connects, plays a seat by always passing on bids and playing the first legal
/// card, and returns `(winning team index, ns_score, ew_score)`.
async fn play_a_match(addr: std::net::SocketAddr) -> (u8, u8, u8) {
    let url = format!("ws://{addr}/ws");
    let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut tx, mut rx) = ws.split();

    send(
        &mut tx,
        &ClientMsg::Hello {
            name: "tester".into(),
            seat: None,
        },
    )
    .await;

    let mut my_seat: Option<Player> = None;
    let mut hand: Vec<Card> = Vec::new();

    while let Some(item) = rx.next().await {
        let Message::Text(text) = item.unwrap() else {
            continue;
        };
        match serde_json::from_str::<ServerMsg>(&text).unwrap() {
            ServerMsg::Joined { your_seat, .. } => my_seat = Some(your_seat),
            ServerMsg::Deal { hand: h, .. } => hand = h,
            ServerMsg::GameOver { winner, scores } => {
                // winner is a fixed team index (0 = North/South, 1 = East/West).
                return (winner, scores[0], scores[1]);
            }
            ServerMsg::Awaiting {
                player,
                hint,
                legal,
            } => {
                if Some(player) != my_seat {
                    continue; // not our turn
                }
                let reply = match hint {
                    // Always pass; the heuristic bots make trump and drive play.
                    TurnHint::Bid { .. } => ClientMsg::Pass,
                    // Bury any card we hold (we still have our dealt five here).
                    TurnHint::Discard => ClientMsg::Discard { card: hand[0] },
                    // Play the first legal card the engine offered.
                    TurnHint::Play { .. } => ClientMsg::Play {
                        card: legal.expect("active player gets legal plays")[0],
                    },
                };
                send(&mut tx, &reply).await;
            }
            _ => {} // Sync, Update, TrickWon, HandComplete, Error: ignore
        }
    }

    panic!("connection closed before the match ended");
}

async fn send<S>(tx: &mut S, msg: &ClientMsg)
where
    S: SinkExt<Message> + Unpin,
{
    let json = serde_json::to_string(msg).unwrap();
    let _ = tx.send(Message::Text(json)).await;
}
