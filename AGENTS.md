## Overview

A Euchre engine and multiplayer server in Rust.
Euchre is a 4-player, 2v2 trick-taking game on a 24-card deck (Nine–Ace).
The main quirk is the trump ordering: the Jack of the trump suit (**right bower**)
and the same-color Jack (**left bower**) outrank all other cards, and the
left bower counts as trump, not its printed suit.

## Commands

```bash
cargo build --workspace          # build everything
cargo test --workspace           # run all tests
cargo test -p euchre-engine      # test one crate
cargo test -p euchre-agents --test vs_random  # one integration test target (by file name)
cargo test --workspace test_name        # run a single test by name (substring filter)
cargo clippy --workspace --all-targets  # lint
cargo fmt                        # format
```

There is no CI config; run `cargo test`, `cargo clippy`, and `cargo fmt`
locally before considering work done.

## Architecture

Five crates, under the `crates/` directory:

### `euchre-interface` — common types and agent interface

Has no dependency on the other crates.

- **`card.rs`**: `Card`, `Suit`, `Rank`, `Color`. The trump-aware helpers live
  here and encode the bower rules.
- **`game.rs`**: `Seat`, `Trick`, `Contract`, `GameRules`, and `GameView` — the
  read-only, hidden-information-respecting snapshot handed to an agent at each
  decision point.
- **`agent.rs`**: the `Agent` trait with four decision points
  (`bid_upcard`, `bid_call`, `discard`, `play_card`) plus `observe_hand_end`.

`serde` is an optional feature here (the server enables it).

### `euchre-engine` — rules, scoring, orchestration

Depends only on `euchre-interface`.

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

### `euchre-agents` — concrete `Agent` implementations

Depends only on `euchre-interface`, but takes a dev-dependency on
`euchre-engine` for testing agent play.

- `RandomAgent` — uniform legal choice; baseline opponent and fuzz source.
- `HeuristicAgent` — basic rule-of-thumb bidding/play
- `AdvancedAgent` — a stronger heuristic player
- `MonteCarloAgent` — a searching agent (Perfect-Information Monte Carlo)
- `NeuralAgent` — a *learned*, search-free agent
- `StrongAgent` — a *learned*, search-free agent tuned to beat `NeuralAgent`

### `euchre-server` — websocket multiplayer

Async (tokio + axum). One shared `Room` = one table, not a lobby. Seats are a
mix of connected humans and server-side bots, interchangeable to the engine.

### `euchre-eval` — agent evaluation harness

Measures whether one `Agent` wins more matches than another. Includes a
tournament binary for multi-agent matchups.
