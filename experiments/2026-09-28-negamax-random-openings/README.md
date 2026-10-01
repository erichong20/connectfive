# Negamax randomized-opening evaluation

## Hypothesis

The encouraging empty-board result will persist across paired randomized
openings, and a 250-node budget will preserve tactical accuracy while remaining
cheap enough for interactive use.

## Code and configuration

- Base commit: `91eb99fd4513ab570b8e0e6dcf6c702f42c81c40` with uncommitted search and
  evaluation changes.
- Agent: Negamax, maximum depth 3, candidate width 10.
- Opponent: one-ply Tactical.
- Main run: 50 games, seed 200, four opening plies, 250 nodes per move.
- Each generated opening was played twice with agent colors reversed.
- Rules: 15x15 exact-five Gomoku; overlines legal but non-winning.
- Hardware/software: Darwin 25.6.0 arm64, Python 3.14.7, JAX 0.11.2,
  NumPy 2.5.3.
- Compute: local CPU only; $0 purchased compute.

## Tactical budget benchmark

All budgets from 250 through 1,500 solved 8/8 essential fixtures. On that small
suite, 250 nodes averaged 117 ms and had a 269 ms maximum observed position.
These easy fixtures did not represent full-game cost.

## Main result

`python scripts/evaluate.py negamax tactical --games 50 --seed 200 --opening-plies 4 --node-budget 250`

- 16 wins, 21 losses, 13 draws.
- Match score: 45%; approximate 95% interval: 32%-59%.
- As Black: 10 wins, 8 losses, 7 draws.
- As White: 6 wins, 13 losses, 6 draws.
- 95.8 average total moves per game.
- 331.4 ms average Negamax move time.
- 227.2 average counted nodes and depth 2.32.
- 767.5 seconds total runtime.
- No illegal moves.

## Budget comparison

On the same ten games from seed 100:

- 250 nodes: 5-2-3, 343 ms/move, 228 nodes, depth 2.29.
- 1,500 nodes: 6-3-1, 461 ms/move, 293 nodes, depth 2.96.

Both earned 6.5/10 match points. More nodes changed individual outcomes but did
not improve aggregate score on this controlled sample.

## Failure review

The main run saved 21 replayable losses: 8 as Black and 13 as White. The node
cap was reached on 387 of 609 Negamax moves in those losses. In each of the four
shortest losses, the final defensive turn faced two opponent winning moves at
once. Negamax was recognizing an already-forced loss, but had permitted the fork
to form earlier.

## Interpretation and decision

The hypothesis was not supported. Negamax is not demonstrated stronger than
Tactical, and its measured latency is not attractive for the browser. Do not
promote or port this version. The next experiment should add explicit fork and
forcing-line awareness to candidate evaluation or a tactical extension at leaf
nodes, then rerun the same saved openings before generating new games.

All 69 repository tests pass after the search and evaluation-harness changes.
