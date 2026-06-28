//! The websocket connection handler: bridges one socket to the room.
//!
//! Per connection there are two cheap tasks: a **writer** that drains this
//! connection's outbound [`ServerMsg`] channel to the socket, and the **reader**
//! (this function) that deserializes incoming [`ClientMsg`]s and forwards them
//! to the room as [`RoomMsg`]s. The first client message must be
//! [`ClientMsg::Hello`]; once the room assigns a table position, every later
//! message is tagged with it.

use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, oneshot};

use crate::AppState;
use crate::protocol::{ClientMsg, ServerMsg};
use crate::room::RoomMsg;

/// Hands each connection a small unique id so its log lines can be correlated
/// (and, after the handshake, tied to a table position).
static NEXT_CONN_ID: AtomicU64 = AtomicU64::new(1);

/// Axum handler for `GET /ws`: upgrades to a websocket and serves it.
pub async fn ws_handler(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state.room_tx))
}

async fn handle_socket(socket: WebSocket, room_tx: mpsc::UnboundedSender<RoomMsg>) {
    let conn_id = NEXT_CONN_ID.fetch_add(1, Ordering::Relaxed);
    tracing::info!(conn_id, "client connected");
    let (mut sink, mut stream) = socket.split();

    // Outbound: the room pushes ServerMsgs here; this task writes them out.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<ServerMsg>();
    let writer = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            let json = match serde_json::to_string(&msg) {
                Ok(json) => json,
                Err(e) => {
                    tracing::error!(conn_id, error = %e, "failed to serialize outgoing message");
                    continue;
                }
            };
            tracing::debug!(conn_id, msg = %json, "sent");
            if sink.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    });

    // Handshake: wait for HELLO and a table-position assignment.
    let player = match handshake(&mut stream, &room_tx, &out_tx, conn_id).await {
        Some(player) => player,
        None => {
            tracing::info!(conn_id, "client disconnected before seating");
            writer.abort();
            return;
        }
    };
    tracing::info!(conn_id, player, "client seated");

    // Main read loop: forward each action to the room.
    while let Some(item) = stream.next().await {
        match item {
            Ok(Message::Text(text)) => {
                tracing::debug!(conn_id, player, msg = %text, "received");
                match serde_json::from_str::<ClientMsg>(&text) {
                    Ok(msg) => {
                        if room_tx.send(RoomMsg::Action { player, msg }).is_err() {
                            break; // room is gone
                        }
                    }
                    Err(e) => {
                        tracing::warn!(conn_id, player, error = %e, "could not parse client message");
                        let _ = out_tx.send(ServerMsg::Error {
                            message: format!("could not parse message: {e}"),
                        });
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(_) => {} // ping/pong/binary: ignore
            Err(e) => {
                tracing::debug!(conn_id, player, error = %e, "websocket error");
                break;
            }
        }
    }

    tracing::info!(conn_id, player, "client disconnected");
    let _ = room_tx.send(RoomMsg::Disconnect { player });
    writer.abort();
}

/// Reads messages until a valid `HELLO` yields a table position, or the socket
/// closes / the table is full (returns `None`).
async fn handshake(
    stream: &mut futures_util::stream::SplitStream<WebSocket>,
    room_tx: &mpsc::UnboundedSender<RoomMsg>,
    out_tx: &mpsc::UnboundedSender<ServerMsg>,
    conn_id: u64,
) -> Option<crate::protocol::Player> {
    while let Some(item) = stream.next().await {
        let text = match item {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) | Err(_) => return None,
            Ok(_) => continue,
        };
        tracing::debug!(conn_id, msg = %text, "received");
        match serde_json::from_str::<ClientMsg>(&text) {
            Ok(ClientMsg::Hello { name, seat }) => {
                let (ack_tx, ack_rx) = oneshot::channel();
                room_tx
                    .send(RoomMsg::Hello {
                        name,
                        seat,
                        out: out_tx.clone(),
                        ack: ack_tx,
                    })
                    .ok()?;
                match ack_rx.await {
                    Ok(Some(seat)) => return Some(seat),
                    _ => {
                        let _ = out_tx.send(ServerMsg::Error {
                            message: "the table is full".into(),
                        });
                        return None;
                    }
                }
            }
            Ok(_) => {
                let _ = out_tx.send(ServerMsg::Error {
                    message: "expected HELLO as the first message".into(),
                });
            }
            Err(e) => {
                tracing::warn!(conn_id, error = %e, "could not parse client message");
                let _ = out_tx.send(ServerMsg::Error {
                    message: format!("could not parse message: {e}"),
                });
            }
        }
    }
    None
}
