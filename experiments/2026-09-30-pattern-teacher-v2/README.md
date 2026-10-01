# Pattern-search teacher, supervised v2, and one DAgger round

## Hypothesis

The v1 network (39.9% held-out move accuracy, 0-20 against Tactical) was limited
by its data, not its architecture:

1. the 100-node Negamax teacher was no stronger than Tactical (11-14-7) and
   finished depth 3 on only 58 of 1,511 moves;
2. 51% of value labels were zeros from 7 board-filling draws, and 25% of
   positions came after ply 120;
3. 32 games gave only 3,024 positions.

A faster, stronger classical teacher, labelling every position with a soft
policy and a search value, should give a raw policy that beats Tactical
without changing the model size.

## Changes

- `src/connectfive/patterns.py`: incremental board. Each empty cell caches a
  per-direction "line level" (five, open four, four, open three, closed three,
  open two) for both players, from a memoized table of 10-cell neighbourhoods.
  A move updates at most 4 x 11 cells (about 90 microseconds per move/undo,
  versus about 1.4 ms per node and 5 ms per leaf in the old search).
- `src/connectfive/pattern_search.py`: alpha-beta with:
  - a transposition table that stores exact/lower/upper bound flags;
  - threat-based move generation: wins, must-blocks, open-four wins, and
    defences against open threes come before quiet moves;
  - forced replies that do not reduce the remaining depth;
  - a VCF (victory by continuous fours) search at the root and at quiet leaves;
  - root moves searched against a lowered window, so near-best root scores are
    exact. Those scores provide soft targets and unbiased tie-breaking.
- `src/connectfive/teacher.py` and `scripts/generate_teacher_data.py`: parallel
  self-play. The first six post-opening moves are sampled from the teacher's
  soft target for diversity; the label is always the teacher's recommendation.
  Every game is replayed through the JAX environment before saving.
- Dataset format 2: optional `policy_targets` and `search_values`.
- Network: optional constant fourth input plane (board-edge visibility for
  White to move), a jitted raw-policy agent, and new training flags: soft
  policy target, outcome/search/blend value target, value weight, AdamW, and
  cosine learning-rate decay.
- `scripts/dagger_round.py`: the student plays itself; the teacher labels every
  position the student reached.

The old `negamax` and `threatsearch` agents are unchanged baselines. Their
transposition-table and root-tie defects are documented in `docs/lessons.md`.

## Configuration

| Item | Value |
| --- | --- |
| Teacher | `pattern-n2000-d6-w12`: 2,000 nodes, depth 6, width 12, VCF depth 8 |
| Openings | four-ply random local openings (`generate_opening`) |
| Seeds | games 10000-11199; DAgger 50000-50399; training seed 0 |
| Evaluation seeds | 900 (20 games); 2000 (100 games); paired colours |
| Network | 4 blocks, 32 channels, 4 input planes, 133,388 parameters |
| Training | 3,000 updates x 256, AdamW lr 2e-3 cosine to 5%, wd 1e-4, soft policy, value = mean of outcome and search value, value weight 0.5 |
| Split | whole games, 10% validation (120 games / 3,955 positions) |
| Hardware | Apple M1 Pro, 32 GB, CPU only (9 worker processes) |
| Software | Python 3.14.7, JAX 0.11.2, Flax 0.12.10, Optax 0.2.8, NumPy 2.5.3 |
| Code | base commit `91eb99f`, dirty worktree with these uncommitted changes |
| Purchased compute | $0 |

## Results

### Teacher

| Match | Result | 95% interval | Teacher ms/move |
| --- | --- | --- | --- |
| Pattern vs Tactical, 40 games, seed 2000 | 40-0-0 | 91%-100% | 188 |
| Pattern vs Negamax (1,500 nodes), 20 games | 19-1-0 | 76%-99% | 178 |

All 8 tactical-suite positions are solved at a 1,000-node budget.

### Corpus

| | v1 (old) | v2 |
| --- | ---: | ---: |
| Games | 32 | 1,200 |
| Labelled positions | 3,024 | 38,896 |
| Generation time | 220 s | 493 s (9 workers) |
| Draws / value-zero share | 7 games / 51% | 13 games / 7% |
| Mean game length | ~94 | 36.4 |
| Black-White-Draw | 15-10-7 | 905-282-13 |

### Network (held-out games)

| Metric | v1 | v2 | v3 (v2 data + DAgger) |
| --- | ---: | ---: | ---: |
| Exact teacher-move accuracy | 39.9% | 47.5% | 47.1% |
| Prediction inside teacher's near-best set | - | 57.3% | 57.2% |
| Value MAE vs outcome | 1.12 | 0.61 | 0.62 |
| Constant-prediction MAE baseline | - | 0.89 | 0.89 |
| Value sign accuracy, decisive games | - | 74.9% | 75.8% |
| Train move accuracy | 51.6% | 50.0% | - |
| Training time | 62 s | 358 s | 340 s |

### Play (raw greedy policy, no search)

| Opponent | v1 | v2 | v3 |
| --- | --- | --- | --- |
| Random, 20 games | 20-0-0 | 20-0-0 | 20-0-0 |
| Tactical, 20 games, seed 900 | 0-20 | 12-8-0 | 11-8-1 |
| Negamax 1,500 nodes, 20 games | - | 12-8-0 | 14-6-0 |
| Pattern teacher, 20 games | - | 5-15-0 | 4-16-0 |
| **Tactical, 100 games, seed 2000** | **4-94-2 (5%; 2-11%)** | **75-20-5 (77.5%; 68-85%)** | **72-26-2 (73%; 64-81%)** |

The raw network takes about 1 ms per move. Before the jitted agent it took
about 21 ms. No illegal moves occurred.

### DAgger round 1

400 v2 self-play games produced 9,430 labelled positions (12 s of play, 115 s
of labelling). The student picked a teacher near-best move 56% of the time
in its own games. Adding these positions to training did not measurably
change validation metrics or play: 73% versus 77.5% against Tactical, with
overlapping intervals.

## Interpretation

The hypothesis is supported. With the same network size, the raw policy went
from 5% to 77.5% against Tactical. Value error fell below the constant baseline
for the first time. Train and validation accuracy are now within 3 points, so
the model is no longer overfitting. It is now likely capacity-limited or
limited by teacher-target ambiguity (one in three positions has more than one
near-best move).

One DAgger round of this size did not help. Possible reasons: v2 already
reaches positions similar to the teacher corpus; 9k positions is small next to
35k; and blended value labels from student games differ in kind from outcome
labels.

## Anomalies and limitations

- Black wins 76% of teacher self-play. Freestyle Gomoku has a strong
  first-move advantage, and the value head learns this colour bias.
- 56% of positions have a saturated search value (a proven win or loss). Many
  are short forced endings, so the corpus over-represents tactics near the
  end of the game.
- The 20-game rows have wide intervals (about ±20 points); only the 100-game
  Tactical rows are suitable for comparison.
- The Pattern teacher is still a shallow search. It is a strong baseline for
  this project, not an expert Gomoku engine.

## Decision

- Promote `pattern` as the classical champion and teacher. Keep `negamax` and
  `threatsearch` as historical baselines.
- Keep supervised v2 (`runs/teacher-v2/model`) as the learned champion. It
  passes the "beat Tactical" part of the milestone gate.
- Do not run more DAgger rounds at this size. The next useful experiments are
  a wider network (now justified by the small train/validation gap) and using
  the v2 policy/value as move ordering and leaf evaluation inside search or
  MCTS.

## Reproduce

```bash
./.venv/bin/python scripts/generate_teacher_data.py --games 1200 --seed 10000 --workers 9 --node-budget 2000 --out runs/teacher-v2/games.npz
./.venv/bin/python scripts/train_supervised.py runs/teacher-v2/games.npz --steps 3000 --batch-size 256 --learning-rate 0.002 --weight-decay 1e-4 --validation-fraction 0.1 --seed 0 --policy-target soft --value-target blend --value-weight 0.5 --input-planes 4 --out runs/teacher-v2/model
./.venv/bin/python scripts/evaluate_network.py runs/teacher-v2/model --opponent tactical --games 100 --seed 2000 --opening-plies 4
./.venv/bin/python scripts/dagger_round.py runs/teacher-v2/model --games 400 --seed 50000 --workers 9 --out runs/teacher-v2/dagger1.npz
./.venv/bin/python scripts/evaluate.py pattern tactical --games 40 --seed 2000 --opening-plies 4
```
