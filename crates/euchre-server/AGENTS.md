## Commands

```bash
cargo run -p euchre-server                       # serve on EUCHRE_ADDR (default 127.0.0.1:8080), route /ws
EUCHRE_ASSIST=1 cargo run -p euchre-server       # ...with assist mode on (neural SUGGEST hints to humans)
cargo run -p euchre-server --example cli_client  # connect a terminal client (EUCHRE_TABLE to join a code; omit to create)
```

## Architecture

Many concurrent **tables**, each an independent `Room` actor keyed by a 4-digit
code in a shared `Registry`. A client picks a table (or creates one) in its
`Hello`; a connection is identified by a `conn_id`, decoupled from any seat.

- **`room.rs` (`Room`)** — an **actor**: a single task owning a table. One
  owning task preserves the engine's "sequential decisions" guarantee for free.
  It alternates two phases: a **lobby**, where `Seat` requests arrange the four
  seats and a match auto-starts once all four stay occupied for `AUTOSTART` (5s);
  and a **match**, the async analogue of `Driver` (ask the core, route to a bot
  or human, `apply`, broadcast) that returns to the lobby when it ends. The
  `Game` is `None` in the lobby. Bot seats play the `StrongAgent`; a
  `TURN_TIMEOUT` substitutes a `HeuristicAgent` fallback so a vanished player
  can't wedge the table; a disconnected human mid-match is replaced by a bot,
  and a room removes itself from the registry when its last connection leaves.
- **`conn.rs`** — per-socket bridge: a writer task draining an outbound channel
  and a reader task forwarding `ClientMsg`s to the room as `RoomMsg { conn_id }`.
  First client message must be `Hello`, which resolves/creates the table.
- **`protocol.rs`** — JSON wire types (`ClientMsg`/`ServerMsg`), tagged by a
  `"type"` field in `SCREAMING_SNAKE_CASE`. Lobby: `Hello {table?}` /
  `Seat {seat, player}` ↔ `TableState` / `StartGame`. The play is
  **event-sourced**: clients learn their hand from `Deal`, then derive state
  from `Update` / `TrickWon`; `Sync` is only for join/reconnect. Hidden info is
  preserved — a `Discard` rebroadcast carries no card.
- **Assist mode** — an operator toggle (`EUCHRE_ASSIST=1`/`true`, off by
  default), threaded from `main`/`serve`/`router` into `AppState.assist` and
  each `Room`. When on, a room holds a shared `StrongAgent` (the same agent the
  bots play) and, right after every `Awaiting`, privately sends the active human
  a `Suggest` — the agent's recommended move plus, per option, its raw network
  score and the probability it is the best move (a softmax of the scores over
  the legal options). `view::suggestion` builds it; `StrongAgent::score_*` supply
  the logits. The recommended move is the top-scored option, so it matches what
  the bots would play. Off → no `Suggest` is ever sent.
- **`view.rs`** — translation between protocol types and engine types, plus
  `suggestion()` for assist hints.
- **`lib.rs`** — `router`/`serve` wire the `Registry` into an Axum app at `/ws`;
  `AppState::create_table`/`table` spawn and look up rooms.
