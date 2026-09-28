# Evaluation harness and tactical bot

Date: 2026-09-28

## Question

Can we create a reproducible CPU-only baseline that is visibly stronger than
random play and teaches the tactical structure of Gomoku before using search or
machine learning?

## Implementation

- Introduced a shared `Agent` protocol.
- Moved uniform random play into `RandomAgent` while retaining the original CLI
  compatibility wrapper.
- Added `TacticalAgent`, a deliberately one-ply policy with these priorities:
  immediate exact-five win, immediate block, contiguous threat scoring, local
  stone density, and centrality.
- Added deterministic tie-breaking through JAX keys.
- Added a game recorder and an evaluation harness that alternates colors and
  reports wins, losses, draws, illegal moves, game length, and elapsed time.
- Added a command-line evaluation entry point.
- Added Tactical and Random opponent selection to the browser. The browser
  tactical policy mirrors the Python bot's priorities and defaults to Tactical.
- Added tests for deterministic random play, immediate wins and blocks, choosing
  a win over a block, and not treating an overline as an exact-five win.

The bot intentionally does not search ahead. Its failures will define the test
positions and requirements for the shallow-search milestone.

## Compute

No purchased compute is used. Development and evaluation run locally on CPU.

Environment:

- Source base: commit `6983860`, with the training prototype and this milestone
  uncommitted in the working tree.
- macOS 26.6.2 on ARM.
- Python 3.14.7, JAX 0.11.2, PGX 2.6.0.
- JAX device: CPU only.

## Initial results

With seed 0:

| Match | Games | Result for first agent | Average length | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| random vs random | 1,000 | 474-526-0 | 113.3 moves | 50.4 s |
| tactical vs random | 100 | 100-0-0 | 10.6 moves | 2.7 s |
| tactical vs tactical | 100 | 15-13-72 | 183.8 moves | 35.0 s |

Both evaluations alternated colors. There were no illegal moves. The
random-vs-random result is close enough to an even split to support the basic
accounting path. The tactical result establishes that immediate wins, blocks,
and simple threat preferences are overwhelmingly stronger than unstructured
play, but it does not yet measure strength against an adversary that recognizes
the same threats. Tactical self-play produced 72 draws and very long games,
which exposes the one-ply bot's main weakness: both sides can repeatedly answer
local threats without forming a multi-move plan. That is useful motivation for
the later shallow-search milestone.

The complete test suite passed: 35 tests in 41.09 seconds. Ruff also passed.

## Next measurements

1. Add tactical fixtures for open fours, double threats, board edges, and known
   one-ply failure cases.
2. Collect human-play examples where the tactical bot makes a poor move and
   turn them into fixtures for the next iteration.
