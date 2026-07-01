## Commands

```bash
# Retrain the neural agent's embedded weights (run in --release). Two stages:
cargo run --release -p euchre-agents --example train_neural -- --teacher advanced --eval  # 1. behavioural-cloning warm start
cargo run --release -p euchre-agents --example train_rl                                    # 2. self-play RL fine-tune (defaults reproduce the shipped weights; writes the same asset)

# Retrain the strong agent's embedded weights (run in --release). Self-play RL that
# spars against and is checkpoint-selected to beat the neural champion; iterate by
# warm-starting from the previous round's output to compound the edge.
cargo run --release -p euchre-agents --example train_strong                                                       # round 1, from the champion
cargo run --release -p euchre-agents --example train_strong -- --warm-start euchre-agents/assets/euchre-strong.bin  # round 2+, iterating (writes the same asset)
```

## Agent implementations

- `RandomAgent` — uniform legal choice; baseline opponent and fuzz source.
- `HeuristicAgent` — rule-of-thumb bidding/play, no search; reliably beats
  random.
- `AdvancedAgent` — a stronger heuristic player (still no search or learning):
  trick-counting hand evaluation, position-aware bidding with the "next"/"green"
  calling conventions, score-aware aggression, and card counting in the play
  (tracking played cards and revealed voids).
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
  decision) are trained in two stages — **behavioural cloning** of a strong teacher
  for a competent warm start, then **self-play reinforcement learning** (REINFORCE
  with a whitened-return baseline and an entropy bonus) that pushes the policy
  *past* its teacher — so every move is still a single forward pass with no search,
  yet the agent now beats the `AdvancedAgent` it was cloned from. The `neural`
  module is self-contained: `net.rs` is a hand-written, gradient-checked MLP + Adam
  (the policy gradient reuses the same `softmax - onehot` gradient, so it inherits
  that check); `features.rs` encodes the `GameView` in a **trump-relative** frame (cards numbered
  by their role relative to trump) so suit symmetry is learned once; `train.rs` is
  the model bundle, the supervised loop, and the `PolicyTrainer` used for RL. The
  trained weights ship embedded (`assets/euchre-net.bin`); `examples/train_neural.rs`
  produces the cloned warm start and `examples/train_rl.rs` fine-tunes it by
  self-play (both depend on the engine to generate games; the library does not).
  The `tests/neural.rs` integration test asserts the agent beats random, the
  heuristic, and the advanced teacher. The module docs hold the design rationale.
- `StrongAgent` — the strongest search-free agent, built to **beat the neural
  champion** while keeping its single-forward-pass speed. It shares the
  `NeuralAgent` architecture *exactly* (same-sized nets) and reuses its inference
  path verbatim; only the weights differ. They are produced by **self-play RL aimed
  at the champion** (`examples/train_strong.rs`): warm-started from the champion's
  own weights, the policy spars against the *frozen neural champion* with a hotter
  sampling temperature and keeps the checkpoint that beats it by the most on a
  fixed, training-disjoint deck band, iterated to compound the edge. Where
  `train_rl` selected checkpoints on win-rate vs the `AdvancedAgent` (a proxy),
  `train_strong` optimises the real objective directly, so the result wins the
  head-to-head (~56% over 3000 games, McNemar p<0.0001; `cargo run -p euchre-eval --
  strong neural`). Weights ship embedded (`assets/euchre-strong.bin`);
  `tests/strong.rs` asserts it beats both random and the neural champion.
- `OpenAiAdvancedAgent` (`openai::advanced`) — stronger bounded-compute
  strategy with richer hand evaluation and optional late-hand rollouts.
