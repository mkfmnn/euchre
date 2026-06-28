//! The websocket connection handler: bridges one socket to a table's room.
//!
//! Per connection there are two cheap tasks: a **writer** that drains this
//! connection's outbound [`ServerMsg`] channel to the socket, and the **reader**
//! (this function) that deserializes incoming [`ClientMsg`]s and forwards them
//! to the room as [`RoomMsg`]s. The first client message must be
//! [`ClientMsg::Hello`], which names the table to join (or, omitted, asks for a
//! fresh one); once the connection is registered, every later message is tagged
//! with its connection id and the room decides what it means.

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
/// and the room can tell its connections apart.
static NEXT_CONN_ID: AtomicU64 = AtomicU64::new(1);

/// Axum handler for `GET /ws`: upgrades to a websocket and serves it.
pub async fn ws_handler(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
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

    // Handshake: wait for HELLO, then join (or create) a table.
    let room_tx = match handshake(&mut stream, &state, &out_tx, conn_id).await {
        Some(room_tx) => room_tx,
        None => {
            tracing::info!(conn_id, "client disconnected before joining a table");
            writer.abort();
            return;
        }
    };
    tracing::info!(conn_id, "client joined a table");

    // Main read loop: forward each message to the room, tagged by connection.
    while let Some(item) = stream.next().await {
        match item {
            Ok(Message::Text(text)) => {
                tracing::debug!(conn_id, msg = %text, "received");
                match serde_json::from_str::<ClientMsg>(&text) {
                    Ok(msg) => {
                        if room_tx.send(RoomMsg::Msg { conn_id, msg }).is_err() {
                            break; // room is gone
                        }
                    }
                    Err(e) => {
                        tracing::warn!(conn_id, error = %e, "could not parse client message");
                        let _ = out_tx.send(ServerMsg::Error {
                            message: format!("could not parse message: {e}"),
                        });
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(_) => {} // ping/pong/binary: ignore
            Err(e) => {
                tracing::debug!(conn_id, error = %e, "websocket error");
                break;
            }
        }
    }

    tracing::info!(conn_id, "client disconnected");
    let _ = room_tx.send(RoomMsg::Disconnect { conn_id });
    writer.abort();
}

/// Reads messages until a valid `HELLO` resolves a table and the room registers
/// this connection, returning a sender into that room. Returns `None` if the
/// socket closes, the named table does not exist, or the room is gone.
async fn handshake(
    stream: &mut futures_util::stream::SplitStream<WebSocket>,
    state: &AppState,
    out_tx: &mpsc::UnboundedSender<ServerMsg>,
    conn_id: u64,
) -> Option<mpsc::UnboundedSender<RoomMsg>> {
    while let Some(item) = stream.next().await {
        let text = match item {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) | Err(_) => return None,
            Ok(_) => continue,
        };
        tracing::debug!(conn_id, msg = %text, "received");
        match serde_json::from_str::<ClientMsg>(&text) {
            Ok(ClientMsg::Hello { name, table }) => {
                let room_tx = match table {
                    Some(code) => match state.table(&code) {
                        Some(tx) => tx,
                        None => {
                            let _ = out_tx.send(ServerMsg::Error {
                                message: format!("no table with code {code}"),
                            });
                            return None;
                        }
                    },
                    None => {
                        let (code, tx) = state.create_table();
                        tracing::info!(conn_id, code, "created table");
                        tx
                    }
                };
                let (ack_tx, ack_rx) = oneshot::channel();
                room_tx
                    .send(RoomMsg::Join {
                        conn_id,
                        name,
                        out: out_tx.clone(),
                        ack: ack_tx,
                    })
                    .ok()?;
                // The room acks once it has registered us (and sent TableState).
                ack_rx.await.ok()?;
                return Some(room_tx);
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
