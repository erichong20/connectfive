# Bot roadmap

This roadmap is deliberately progressive. Each stage should produce a named,
playable bot, an evaluation against earlier bots, and a written account of what
we learned. Details may change when experiments provide better evidence.

## Current status

- [x] Standard 15x15 exact-five JAX/PGX environment
- [x] CLI and browser game with a random opponent
- [x] AlphaZero training sketch for later study
- [x] Shared Python agent interface
- [x] Deterministic game recorder and paired-color evaluation harness
- [x] First one-ply tactical bot and unit tests
- [x] Compact eight-position tactical regression suite
- [x] Baseline random-vs-random and tactical-vs-random reports
- [x] Tactical bot in the browser
- [x] First budgeted two-ply look-ahead bot
- [x] First iterative-deepening negamax with alpha-beta and a node budget
- [x] Paired randomized-opening evaluation and replayable failure capture
- [x] Fifty-game Negamax-versus-Tactical diagnostic
- [x] First fork-aware threat-extension experiment
- [x] Compact residual policy/value network and tiny-batch overfit check
- [x] Perspective encoding, legal masking, D4 augmentation, and checkpoint tests
- [x] Generate a game-level supervised dataset with a held-out validation split
- [x] Add a checkpoint-backed raw network agent and paired evaluation
- [x] Fast incremental pattern board and threat-aware `pattern` search with VCF
- [x] Parallel teacher self-play with soft policy and search-value labels
- [x] Supervised v2: raw policy beats Tactical 75-20-5 over 100 games;
      held-out value MAE 0.61 (constant baseline 0.89)
- [x] One DAgger round (no measurable gain; documented)
- [x] Use the learned network inside search: guided MCTS beats `pattern` 63-37
      over 100 games at 0.2 s/move (`experiments/2026-09-30-guided-search/`)
- [x] AlphaZero-lite round 1: self-play pipeline works; candidate scored 54.5%
      (45-64%) vs v2, not promoted (`experiments/2026-09-30-alphazero-lite-r1/`)
- [x] AlphaZero-lite round 2 (400 sims): promoted, 62% (52-71%) vs v2 in
      guided MCTS over 100 games
- [x] Browser: az-r2 guided MCTS in TypeScript + WASM SIMD, parity-tested
      against Python (`experiments/2026-10-01-browser-neural-bot/`)
- [ ] Continue rounds from the az-r2 champion

No purchased compute is planned. The current work is CPU-friendly.

## 0. Trustworthy game foundation

Verify exact-five wins, overlines, legal and illegal moves, draws, rewards,
player-relative observations, JIT, batching, recorded-game replay, and agreement
between Python and browser rules.

Learning: immutable state, JAX transformations, zero-sum rewards, deterministic
reproduction, and why environment bugs poison every later result.

Gate: rule tests pass and saved games replay identically.

## 1. Random baseline

Provide a seeded random agent through the same interface used by every later
bot. Measure throughput, game length, color balance, and illegal moves over a
large CPU sample.

Learning: baselines, randomness, agent interfaces, match accounting, and noisy
estimates.

Gate: at least 1,000 games without illegal moves or replay errors.

## 2. Tactical rule bot

Build an inspectable one-ply bot that wins an immediate exact five, blocks an
opponent's immediate win, creates or blocks contiguous threats, prefers moves
near stones, and uses seeded random tie-breaking. Add positions for wins,
blocks, open fours, forks, edge play, and overline traps.

Learning: feature design, tactical patterns, priority conflicts, defensive
versus aggressive play, and the limits of one-ply reasoning.

Gate: solve the fixed tactical suite and convincingly beat random play with both
colors. Document representative failures for the next milestone.

## 3. Pattern evaluator and shallow search — explored, not promoted

Add candidate generation, a documented pattern evaluator, negamax with
alpha-beta pruning, move ordering, a transposition table, and strict node/time
budgets.

The first step is a deliberately small two-ply bot: it keeps the strongest 16
candidate moves, examines the opponent's strongest 12 replies, and evaluates
the resulting tactical threats. This makes the minimax idea visible before we
add recursion, alpha-beta pruning, or a transposition table.

The second step adds iterative-deepening negamax, alpha-beta pruning, a small
transposition table, an explicit node budget, search diagnostics, and an
eight-position JSON tactical suite. It is available as `negamax` in the CLI and
evaluation scripts, but is not yet the browser default: the first ten-game
sample is encouraging but too small for promotion, and its per-move latency
needs a browser-specific budget.

The larger randomized-opening result did not support promotion: 250-node
Negamax scored 16-21-13 against Tactical over 50 games, for a 45% match score
and an approximate 95% interval of 32%-59%. It averaged 331 ms per move. The
shortest saved losses ended in opponent double threats, indicating that fork
prevention and the leaf evaluator matter more than increasing the node budget.

A first threat-extension implementation followed wins, forced blocks, and fork
moves for up to two extra plies and added broken-pattern scoring. On the same
ten paired openings used for the budget comparison, it scored only 3-4-3 while
averaging 558 ms per move and completed depth 1.56. Every move in its four
losses exhausted the 100-node budget. Preserve it as a failed experimental
baseline; do not run a 50-game match or port it to the browser.

Learning: branching factor, minimax, pruning, horizon effects, evaluation
functions, and fair time controls.

Gate: outperform the tactical bot under equal per-move resources.

The implemented search agents did not pass this gate. A cheaper fork-feature
redesign remains a possible later experiment, but it is deferred while the
project tests whether a compact learned representation is more productive.

## 4. Non-neural MCTS

Implement MCTS/UCT with random and tactical rollout policies. Expose tree
statistics and traces for selected positions.

Learning: selection, expansion, simulation, backup, exploration, and why random
rollouts struggle in tactical games.

Gate: measure where MCTS beats shallow search and where it does not.

## 5. Supervised policy/value warm start — current

Generate positions with the strongest classical agents and train a small
residual network to predict moves and outcomes. Split validation data by whole
games.

The first foundation is complete: a 4-block, 32-channel policy/value network
has 133,100 parameters; uses mover-relative stone planes plus an absolute
black-to-move plane; masks illegal policy logits; applies all eight square
symmetries consistently; and round-trips a versioned parameter checkpoint. On
the eight tactical fixtures, a 200-step CPU sanity run reduced total loss from
5.8728 to 0.3472 in about 5.2 seconds. The remaining policy loss is the expected
entropy of fixtures with two equally acceptable target moves, not a failure to
fit them. The value labels in this plumbing check were synthetic and are not
evidence of value accuracy or playing strength.

Next, generate complete games from Tactical and Negamax, store provenance and
final outcomes, split by game, and measure held-out policy accuracy, value error,
and calibration. This pipeline is now implemented and has produced two local
runs. Random opening moves are replayed but excluded as imitation labels.

The 32-game v1 corpus contains 3,024 labeled positions. Its network reached
51.6% training and 39.9% held-out policy accuracy, but held-out value MAE was
1.12 on targets in `[-1, 1]`. The greedy raw policy went 20-0 against Random and
0-20 against Tactical with both colors. This meets the basic Random-play gate
but not the intended teacher-strength or value-generalization standard. Do not
scale the model yet: first improve data coverage, diagnose value splits, and
consider richer or softened teacher policy targets.

A review then found that the v1 teacher rarely completed its advertised depth
and that half the value labels came from board-filling draws. The new
incremental `pattern` search (VCF, threat pruning, bound-flagged transposition
table) beats Tactical 40-0 and Negamax 19-1. Its 1,200-game, 38,896-position
corpus trained supervised v2 with the same 133k-parameter architecture:
47.5% held-out move accuracy, value MAE 0.61, and 75-20-5 against Tactical over
100 paired games (v1: 4-94-2). One DAgger round did not measurably help. See
`experiments/2026-09-30-pattern-teacher-v2/`.

Learning: datasets, leakage, policy loss, value targets, calibration, symmetry
augmentation, and checkpoint reproduction.

Gate: the raw network beats random play and generalizes to held-out games.

## 6. AlphaZero-lite

Use one local accelerator if one is already available; otherwise keep runs
small and educational. Start with a 4-6 block, 64-channel network, roughly 16
Gumbel MuZero simulations, synchronous self-play/training, bounded replay,
eight symmetries, and a deliberate mix of empty and randomized openings.

Before a serious run, fix the current prototype's incomplete checkpoints,
cross-iteration trajectory loss, weak evaluation, missing structured metrics,
and lack of time limits.

Learning: policy iteration, search targets, replay, self-play non-stationarity,
value backup, and batching.

Gate: beat classical baselines in a search-based arena; loss alone is not a gate.

## 7. Reliable research runner

Separate model, self-play, replay, training, evaluation, checkpoints, and
configuration. Save model/optimizer/RNG/progress state atomically; add graceful
resume, JSONL metrics, time limits, paired arenas, and best-model promotion.

Learning: research engineering, fault tolerance, confidence intervals, and the
difference between a demo and a trustworthy system.

Gate: interrupt and resume a run, reproduce a control closely, and generate a
complete report automatically.

## 8. Cost-aware scaling, only if later approved

Compare simulations, network width, replay size, opening mixture, learning rate,
and update ratio through small ablations. Select by strength gained per unit of
compute, not final loss. Renting compute requires a new proposal and explicit
approval; it is not part of the current plan.

## 9. Curriculum, league, and distribution

Possible later research includes smaller-board curriculum, historical opponent
leagues, optimized search kernels, separate collection/training, multiple
workers, and AutoGo-inspired automated experiments. Distribution is justified
only after profiling identifies collection as the dominant bottleneck.
