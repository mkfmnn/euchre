Measures whether one `Agent` wins more matches than another. The objective is
**match-win probability, not points**: skilled Euchre is score-aware (sandbag a
lead, gamble from behind), so a pure points metric could punish good play.

## Commands

```bash
cargo run -p euchre-eval -- heuristic random              # score one agent against another
cargo run -p euchre-eval -- heuristic random --sprt       # stop early once the result is decided
cargo run -p euchre-eval -- neural advanced               # the learned agent vs the teacher it was cloned from

cargo run -p euchre-eval --bin euchre-tournament -- --all          # round-robin every built-in agent, ranked by Elo
cargo run -p euchre-eval --bin euchre-tournament -- random heuristic advanced --format csv  # named pool, CSV output
```

## Architecture

- **`runner.rs`** — match execution and variance reduction. Agents are supplied
  as factories (`AgentFactory`), not instances, so every match gets a fresh,
  cleanly-seeded agent (needed for stateful learners and stochastic agents). The
  core technique is **duplicate dealing** (`run_pair`): each deck is played twice
  with the sides swapped so deal luck cancels. The deck seed (fixed across the
  mirror) and the agent-randomness seed (independent per game) are decoupled, so
  two identically-seeded stochastic agents don't play in lockstep.
- **`stats.rs`** — `wilson_interval` (CI on a win rate), `mcnemar` (the paired
  test for the duplicate design), and `Sprt` (sequential test for early stopping,
  specified in Elo).
- **`tournament.rs`** — round-robin over a pool of `Contestant`s. `run_round_robin`
  plays every pair via the same `run_pair` machinery (disjoint deck-seed bands per
  pairing, so results stay independent) and aggregates into a `wins_matrix` for
  rating.
- **`elo.rs`** — batch **BayesElo** ratings: a Bradley-Terry maximum-likelihood
  `fit` over the whole win matrix (Hunter's MM algorithm), regularised by a small
  prior of virtual games against a rating-0 anchor (keeps sweepers finite, pins
  the scale). Reports per-agent standard errors (relative to the pool mean) from
  the inverse Fisher information.
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