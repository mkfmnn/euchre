# CLAUDE.md

## Overview

A Euchre engine and multiplayer server in Rust, organized as a Cargo workspace
(edition 2024). Euchre is a 4-player, 2v2 trick-taking game on a 24-card deck
(Nine–Ace). The central domain quirk is the trump ordering: the Jack of the
trump suit (**right bower**) and the same-color Jack (**left bower**) outrank
all other cards, and the left bower counts as trump, not its printed suit.

## Commands

```bash
cargo build --workspace          # build everything
cargo test --workspace           # run all tests
cargo test -p euchre-engine      # test one crate
cargo test -p euchre-agents --test vs_random  # one integration test target (by file name)
cargo test --workspace test_name        # run a single test by name (substring filter)
cargo clippy --workspace --all-targets  # lint
cargo fmt                        # format

cargo run -p euchre-server                       # serve one table on EUCHRE_ADDR (default 127.0.0.1:8080), route /ws
cargo run -p euchre-server --example cli_client  # connect a terminal client to it
```

There is no CI config; run `cargo test`, `cargo clippy`, and `cargo fmt`
locally before considering work done.

## Architecture

Four crates form a dependency chain `interface → engine → agents → server`.
Keep concerns in their layer — e.g. don't add randomness to the core, don't add
strategies to the engine.

### `euchre-interface` — shared vocabulary, no logic

Engine-agnostic types and the `Agent` trait. Defines *what an agent is asked and
answers*, not how the game runs. Has no dependency on the other crates.

- **`card.rs`**: `Card`, `Suit`, `Rank`, `Color`. The trump-aware helpers live
  here and encode the bower rules: `effective_suit`, `is_left/right_bower`,
  `is_trump`, and `trump_strength(trump, led)` — the single sort key for "which
  card wins a trick." When reasoning about card comparison, always go through
  these, never raw `Rank`/`Suit`. Cards serialize as 2-letter codes (`"JS"`).
- **`game.rs`**: `Seat`/`Team` topology (N/S vs E/W, partners across),
  `Trick`, `Contract`, `GameRules`, and `GameView` — the read-only,
  hidden-information-respecting snapshot handed to an agent at each decision.
- **`agent.rs`**: the `Agent` trait with four decision points
  (`bid_upcard`, `bid_call`, `discard`, `play_card`) plus `observe_hand_end`.
  All methods take `&mut self`; the engine guarantees sequential, never
  concurrent, calls per agent.

`serde` is an optional feature here (the server enables it).

### `euchre-engine` — rules, scoring, orchestration. Ships no agents.

Two layers:

- **`game.rs` (`Game`)** — the **core** state machine. Deterministic and pure:
  it never shuffles, reads input, or prints. The caller loop is
  `next_action()` → `view(seat)` → `apply(Decision)`, repeated until
  `Action::HandComplete`, then `is_over()` / `start_next_hand(deck)`. `Game::new`
  takes an already-shuffled 24-card deck, which is what makes it reproducible
  and trivial to test. This same core backs both the terminal driver and the
  server.
- **`driver.rs` (`Driver`)** — a synchronous terminal game loop wiring the core
  to four `Player`s (`Player::Bot(&mut dyn Agent)` or `Player::Human`).
  `Driver::headless` runs four bots with no input; `Verbosity` controls
  narration. `with_seed` gives reproducible matches.
- **`shuffle.rs`** — `deal(rng)`, the one place randomness becomes a deck.

### `euchre-agents` — concrete `Agent` strategies

- `RandomAgent` — uniform legal choice; baseline opponent and fuzz source.
- `HeuristicAgent` — rule-of-thumb bidding/play, no search; reliably beats
  random. The `tests/vs_random.rs` integration test asserts this.

### `euchre-server` — websocket multiplayer (walking skeleton)

Async (tokio + axum). One shared `Room` = one table, not a lobby. Seats are a
mix of connected humans and server-side bots, interchangeable to the engine.

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

## Conventions

- Public items carry doc comments; the crate-level docs in each `lib.rs` are the
  best starting point and include runnable usage examples. Match this density
  when adding public API.
- The core engine must stay deterministic and I/O-free — randomness and
  printing belong in the driver/server layers.
- Trick logic and card comparison must route through `Card::trump_strength` /
  `effective_suit` so the bower rules stay correct in one place.
