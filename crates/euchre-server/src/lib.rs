//! # euchre-server
//!
//! A websocket server that runs a Euchre match end-to-end over the wire — the
//! walking skeleton of a multiplayer backend. One table seats four players;
//! each seat is either a connected human (a websocket client) or a server-side
//! bot, and the two are interchangeable from the engine's point of view.
//!
//! The pieces:
//!
//! * [`protocol`] — the JSON wire types ([`ClientMsg`](protocol::ClientMsg) /
//!   [`ServerMsg`](protocol::ServerMsg)).
//! * [`view`] — translation between those and the engine's types.
//! * [`room`] — the actor that owns the [`Game`](euchre_engine::Game) and drives
//!   a match.
//! * [`conn`] — the per-socket bridge to the room.
//!
//! [`router`] wires a single room to an Axum app with one `/ws` route; [`serve`]
//! runs it on a listener. The single shared room means this is one table, not a
//! lobby — multiple tables, matchmaking, and reconnection are future work.

pub mod conn;
pub mod protocol;
pub mod room;
pub mod view;

use axum::Router;
use axum::routing::any;
use euchre_engine::GameConfig;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use room::{Room, RoomMsg};

/// Shared state handed to the websocket handler: a sender into the room.
#[derive(Clone)]
pub struct AppState {
    pub room_tx: mpsc::UnboundedSender<RoomMsg>,
}

/// Builds the Axum app, spawning the single room task that backs it.
pub fn router(config: GameConfig) -> Router {
    let (room_tx, room_rx) = mpsc::unbounded_channel::<RoomMsg>();
    tokio::spawn(Room::new(config, room_rx).run());
    Router::new()
        .route("/ws", any(conn::ws_handler))
        .with_state(AppState { room_tx })
}

/// Serves the app on an already-bound listener until the process ends.
pub async fn serve(listener: TcpListener, config: GameConfig) -> std::io::Result<()> {
    axum::serve(listener, router(config)).await
}
