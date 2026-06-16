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

cargo run -p euchre-eval -- heuristic random              # score one agent against another
cargo run -p euchre-eval -- heuristic random --sprt       # stop early once the result is decided
cargo run -p euchre-eval -- neural advanced               # the learned agent vs the teacher it was cloned from

cargo run -p euchre-eval --bin euchre-tournament -- --all          # round-robin every built-in agent, ranked by Elo
cargo run -p euchre-eval --bin euchre-tournament -- random heuristic advanced --format csv  # named pool, CSV output

# Retrain the neural agent's embedded weights (run in --release; the teacher is slow):
cargo run --release -p euchre-agents --example train_neural -- --teacher advanced --eval
```

There is no CI config; run `cargo test`, `cargo clippy`, and `cargo fmt`
locally before considering work done.

## Architecture

Five crates build on the chain `interface → engine → agents`, with `server` and
`eval` as two consumers of `agents`. Keep concerns in their layer — e.g. don't
add randomness to the core, don't add strategies to the engine.

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
- `AdvancedAgent` — a stronger heuristic player (still no search or learning):
  trick-counting hand evaluation, position-aware bidding with the "next"/"green"
  calling conventions, score-aware aggression, and card counting in the play
  (tracking played cards and revealed voids to spot masters, draw trump, and
  slough dead cards). The `tests/advanced.rs` integration test asserts it beats
  both random and the plain heuristic.
- `MonteCarloAgent` — the first *searching* agent (Perfect-Information Monte
  Carlo). For each play it samples full deals of the hidden cards consistent with
  what it has seen (respecting revealed voids), solves each sampled world exactly
  with a small double-dummy alpha-beta search (`solver.rs`), and plays the card
  with the best average match-point outcome. It anchors to `AdvancedAgent`'s card,
  overriding only when the search is confident, so it is robustly at least as
  strong as the advanced agent at any search width. Its **bidding** is likewise
  anchored PIMC — `AdvancedAgent` picks the suit and default bid, and the search
  retunes alone/partner, vetoes losing makes, and orders up profitable passes
  (`discard` stays delegated; `play_only()` disables the bidding search). Tunable
  via `with_determinizations`; the `tests/montecarlo.rs` integration test asserts
  it beats both random and the advanced agent.
- `NeuralAgent` — a *learned*, search-free agent. Four small policy MLPs (one per
  decision) are trained by **behavioural cloning** of a strong teacher, so every
  move is a single forward pass — no search, by design. The `neural` module is
  self-contained: `net.rs` is a hand-written, gradient-checked MLP + Adam (no ML
  dependency, matching the project's verifiable-numerics bent); `features.rs`
  encodes the `GameView` in a **trump-relative** frame (cards numbered by their
  role relative to trump) so suit symmetry is learned once; `train.rs` is the
  model bundle + supervised loop. The trained weights ship embedded
  (`assets/euchre-net.bin`, distilled from `AdvancedAgent`); `examples/train_neural.rs`
  regenerates them (it, not the library, depends on the engine to generate games),
  and the `tests/neural.rs` integration test asserts the agent beats random and the
  heuristic and stays competitive with its teacher. The module docs hold the
  rationale (cloning over RL, hand-rolled MLP over a framework, the encoding).

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

### `euchre-eval` — agent evaluation harness

Measures whether one `Agent` wins more matches than another. The objective is
**match-win probability, not points**: skilled Euchre is score-aware (sandbag a
lead, gamble from behind), so a points metric would punish good play. Matches
therefore run to the real target score and are scored purely on who won.

- **`runner.rs`** — match execution and variance reduction. Agents are supplied
  as factories (`AgentFactory`), not instances, so every match gets a fresh,
  cleanly-seeded agent (needed for stateful learners and stochastic agents). The
  core technique is **duplicate dealing** (`run_pair`): each deck is played twice
  with the sides swapped so deal luck cancels. The deck seed (fixed across the
  mirror) and the agent-randomness seed (independent per game) are decoupled, so
  two identically-seeded stochastic agents don't play in lockstep.
- **`stats.rs`** — `wilson_interval` (CI on a win rate), `mcnemar` (the paired
  test for the duplicate design), and `Sprt` (sequential test for early stopping,
  specified in Elo). No external stats dependency; keep formulas verifiable.
- **`tournament.rs`** — round-robin over a pool of `Contestant`s. `run_round_robin`
  plays every pair via the same `run_pair` machinery (disjoint deck-seed bands per
  pairing, so results stay independent) and aggregates into a `wins_matrix` for
  rating.
- **`elo.rs`** — batch **BayesElo** ratings: a Bradley-Terry maximum-likelihood
  `fit` over the whole win matrix (Hunter's MM algorithm), regularised by a small
  prior of virtual games against a rating-0 anchor (keeps sweepers finite, pins
  the scale). Reports per-agent standard errors (relative to the pool mean) from
  the inverse Fisher information. Same no-external-deps, verifiable-formulas rule.
- **`lib.rs`** — `builtin(name)` registry mapping agent names to factories; add
  new agents here so the CLI and tournament can name them.
- **`main.rs`** — the `euchre-eval` binary (clap), with fixed-games and `--sprt`
  modes.
- **`bin/tournament.rs`** — the `euchre-tournament` binary (clap): round-robin a
  named pool (or `--all`), print the Elo leaderboard plus a pairwise win-rate grid,
  with `--format table|csv|json`.

Not yet built: rayon parallelism (matches are independent — add it once
search-based agents make each one slow); TrueSkill as an alternative to BayesElo
(handles the 2v2 team structure natively); and CSV/JSON regression tracking over
time built on the tournament's machine-readable output.

## Conventions

- Public items carry doc comments; the crate-level docs in each `lib.rs` are the
  best starting point and include runnable usage examples. Match this density
  when adding public API.
- The core engine must stay deterministic and I/O-free — randomness and
  printing belong in the driver/server layers.
- Trick logic and card comparison must route through `Card::trump_strength` /
  `effective_suit` so the bower rules stay correct in one place.
