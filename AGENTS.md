# Connect Five Agent Guide

This repository is both a Gomoku project and a learning project. Make the bot
stronger while leaving behind a reproducible explanation of what was tried,
why it was tried, what it cost, and what was learned.

## Rules and product

- Play standard Gomoku on a 15x15 board.
- Exactly five contiguous stones wins. An overline is legal but is not a win.
- There are no Renju forbidden-move rules.
- Keep the JAX environment, agents, CLI, tests, browser, and documentation
  consistent with these rules.
- The eventual product is a browser-playable bot, but correctness and honest
  strength measurement come before presentation.

## Working principles

- Follow the staged plan in `docs/roadmap.md`; work on one milestone at a time.
- Preserve every useful baseline so later agents can play earlier agents.
- Prefer the simplest implementation that answers the current research question.
- Never infer playing strength from training loss. Use head-to-head games and
  fixed tactical positions.
- Keep deterministic seeds available and record experiment seeds.
- Keep game rules, agents, search, training, evaluation, and UI separated.
- Put reusable code in `src/connectfive/` and thin entry points in `scripts/`.
- Test rule edge cases, player perspective, reward signs, search, recorded-game
  replay, and training data before trusting results.
- Preserve unrelated user changes in a dirty worktree.

## Learning and documentation

Documentation is part of the implementation. Maintain:

- `docs/roadmap.md` for milestones, gates, and current status.
- `docs/architecture.md` when system boundaries need explanation.
- `docs/lessons.md` for durable explanations and discoveries.
- `docs/journal/YYYY-MM-DD-topic.md` for chronological work notes.
- `experiments/<run-id>/README.md` for each meaningful experiment report.

For each new concept, explain it without code, connect it to the previous bot,
show the essential equation or pseudocode, point to code and tests, demonstrate
one understandable position, measure the effect, and record limitations.

Every meaningful experiment must record its hypothesis, code commit and dirty
state, configuration, seeds, hardware, software versions, runtime, compute cost,
raw evaluation protocol, results, anomalies, interpretation, and next decision.
Generated checkpoints and raw logs belong in ignored `runs/`; concise reports
and small fixtures belong in Git.

## Evaluation rules

- Give compared agents the same rules and search or time budget.
- Play both colors. Pair non-empty openings with colors reversed.
- Record wins, losses, draws, illegal moves, move time, and game length.
- State the number of games and uncertainty; do not overstate a small sample.
- Keep a stable tactical suite and historical champion checkpoints.
- A new champion must pass an explicit promotion gate.

## Compute policy

- Use local resources and existing agent-plan allowances only. Do not purchase,
  rent, or provision additional compute without explicit user approval.
- Agent subscriptions help write, inspect, and reason about code; do not assume
  they provide a sustained accelerator for model training.
- Short CPU smoke tests and small educational experiments are encouraged.
- Before any future long run, benchmark throughput and add resumable checkpoints,
  structured metrics, a wall-clock limit, and automatic failure stops.
- Stop on NaNs, illegal moves, corrupt targets, no completed games, or severe
  throughput regressions.

## Current milestone

The current milestone is a CPU-budgeted shallow-search bot. Preserve the random
and one-ply tactical baselines, compare the new bot against both, document
representative wins and failures, and do not begin neural training yet.
