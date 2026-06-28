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
//! [`router`] wires a [`Registry`] of tables to an Axum app with one `/ws`
//! route; [`serve`] runs it on a listener. Each table is an independent [`Room`]
//! actor keyed by a short code; a client picks one (or makes a new one) in its
//! `HELLO`. Reconnection is still future work.

pub mod conn;
pub mod protocol;
pub mod room;
pub mod view;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::routing::any;
use euchre_engine::GameConfig;
use rand::RngExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use room::{Room, RoomMsg};

/// The set of live tables, keyed by code. Each value is a sender into that
/// table's [`Room`] actor. The mutex is held only for brief lookups/inserts,
/// never across an `.await`.
pub type Registry = Arc<Mutex<HashMap<String, mpsc::UnboundedSender<RoomMsg>>>>;

/// Shared state handed to the websocket handler: the table registry and the
/// game config to stamp on any newly created table.
#[derive(Clone)]
pub struct AppState {
    pub registry: Registry,
    pub config: GameConfig,
}

impl AppState {
    /// Looks up a live table's sender by code.
    pub fn table(&self, code: &str) -> Option<mpsc::UnboundedSender<RoomMsg>> {
        self.registry.lock().unwrap().get(code).cloned()
    }

    /// Creates a new table: picks a free code, spawns its [`Room`] actor, and
    /// registers it. Returns the code and a sender into the new room.
    pub fn create_table(&self) -> (String, mpsc::UnboundedSender<RoomMsg>) {
        let (room_tx, room_rx) = mpsc::unbounded_channel::<RoomMsg>();
        let mut tables = self.registry.lock().unwrap();
        let code = loop {
            let candidate = format!("{:04}", rand::rng().random_range(0..10_000));
            if !tables.contains_key(&candidate) {
                break candidate;
            }
        };
        tables.insert(code.clone(), room_tx.clone());
        tokio::spawn(Room::new(self.config, code.clone(), self.registry.clone(), room_rx).run());
        (code, room_tx)
    }
}

/// Builds the Axum app with an empty table registry.
pub fn router(config: GameConfig) -> Router {
    let state = AppState {
        registry: Arc::new(Mutex::new(HashMap::new())),
        config,
    };
    Router::new()
        .route("/ws", any(conn::ws_handler))
        .with_state(state)
}

/// Serves the app on an already-bound listener until the process ends.
pub async fn serve(listener: TcpListener, config: GameConfig) -> std::io::Result<()> {
    axum::serve(listener, router(config)).await
}
