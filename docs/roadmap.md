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
- [ ] Larger tactical fixture suite
- [x] Baseline random-vs-random and tactical-vs-random reports
- [x] Tactical bot in the browser

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

## 2. Tactical rule bot — current

Build an inspectable one-ply bot that wins an immediate exact five, blocks an
opponent's immediate win, creates or blocks contiguous threats, prefers moves
near stones, and uses seeded random tie-breaking. Add positions for wins,
blocks, open fours, forks, edge play, and overline traps.

Learning: feature design, tactical patterns, priority conflicts, defensive
versus aggressive play, and the limits of one-ply reasoning.

Gate: solve the fixed tactical suite and convincingly beat random play with both
colors. Document representative failures for the next milestone.

## 3. Pattern evaluator and shallow search

Add candidate generation, a documented pattern evaluator, negamax with
alpha-beta pruning, move ordering, a transposition table, and strict node/time
budgets.

Learning: branching factor, minimax, pruning, horizon effects, evaluation
functions, and fair time controls.

Gate: outperform the tactical bot under equal per-move resources.

## 4. Non-neural MCTS

Implement MCTS/UCT with random and tactical rollout policies. Expose tree
statistics and traces for selected positions.

Learning: selection, expansion, simulation, backup, exploration, and why random
rollouts struggle in tactical games.

Gate: measure where MCTS beats shallow search and where it does not.

## 5. Optional supervised policy/value warm start

Generate positions with the strongest classical agents and train a small
residual network to predict moves and outcomes. Split validation data by whole
games. This stage is optional if evidence shows direct self-play is simpler and
equally efficient.

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
