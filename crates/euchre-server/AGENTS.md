## Commands

```bash
cargo run -p euchre-server                       # serve one table on EUCHRE_ADDR (default 127.0.0.1:8080), route /ws
cargo run -p euchre-server --example cli_client  # connect a terminal client to it
```

## Architecture

- **`room.rs` (`Room`)** — an **actor**: a single task owning the `Game`. This
  is deliberate — one owning task preserves the engine's "sequential decisions"
  guarantee for free. It's the async analogue of `Driver`: ask the core, route
  to a bot (call directly) or human (send `Awaiting`, await reply), `apply`,
  broadcast. A `TURN_TIMEOUT` substitutes a `HeuristicAgent` fallback move so a
  vanished player can't wedge the table.
- **`conn.rs`** — per-socket bridge: a writer task draining an outbound channel
  and a reader task forwarding `ClientMsg`s to the room as `RoomMsg`s. First
  client message must be `Hello`.
- **`protocol.rs`** — JSON wire types (`ClientMsg`/`ServerMsg`), tagged by a
  `"type"` field in `SCREAMING_SNAKE_CASE`. The protocol is **event-sourced**:
  clients learn their hand from `Deal`, then derive state from `Update` /
  `TrickWon` events; `Sync` is only for join/reconnect. Hidden info is preserved
  on the wire — a `Discard` rebroadcast carries no card.
- **`view.rs`** — translation between protocol types and engine types.
- **`lib.rs`** — `router`/`serve` wire one room into an Axum app at `/ws`.
