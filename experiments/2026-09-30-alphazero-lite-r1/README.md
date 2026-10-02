# AlphaZero-lite round 1: guided-MCTS self-play

## Hypothesis

Training on guided-MCTS self-play (visit-count policy targets, outcome plus
root-value labels) improves the v2 network beyond what imitation of the
`pattern` teacher gave.

## Protocol

- Self-play: `scripts/selfplay_round.py`, v2 network, 200 simulations/move,
  c_puct 1.5, root Dirichlet noise 0.25 (alpha 0.3), the first 8
  post-opening moves sampled by visit count, four-ply random openings,
  seeds 70000-70599, 9 single-thread XLA workers.
- Every game replay-verified through the JAX environment.
- Training: fine-tune v2 for 2,000 x 256 updates, AdamW lr 5e-4 cosine, wd
  1e-4, soft (visit) policy target, value = mean(outcome, root Q), value
  weight 0.5. Self-play split 90/10 by game (seed 0); all 1,200 teacher-v2
  games are added to training as replay.
- Gate: guided MCTS (new network) vs guided MCTS (v2), 0.2 s/move,
  100 paired games, seeds 6000-6049; promote only if the 95% interval
  excludes 50%.
- Apple M1 Pro CPU, $0. Base commit `6a8709d` plus this change.

## Data anomaly and repair

Positions where MCTS returns immediately (proven win, forced block, single
move) had zero visits, so the first run stored **empty policy targets** for
5,201 of 18,423 positions. The code now stores a one-hot target on the forced
move. The saved corpus was repaired by applying that same rule
(`runs/az-r1/games-fixed.npz`) instead of regenerating it.

## Results

| Measurement | Value |
| --- | --- |
| Self-play games / positions | 600 / 18,423 (34.7 plies mean) |
| Black-White-Draw | 476-113-11 |
| Generation | 558 s, 2.64M simulations |
| Training | 264 s |
| Held-out self-play policy accuracy | 66.1% (soft-target loss 1.42) |
| Held-out value MAE vs outcome | 0.66 (constant baseline 0.88) |
| **Gate: vs v2, guided MCTS, 100 games** | **53-44-3, 54.5%, 95% CI 45-64%** |
| Raw policy vs Tactical, 100 games, seed 2000 | 80-18-2, 81% (72-87%); v2: 77.5% (68-85%) |

## Interpretation

Both measurements point the same direction, but neither is statistically
significant. The new network does **not** pass the promotion gate. One round
of 600 games at 200 simulations produces relatively little new information:
the MCTS targets are close to what v2 already predicts (66% held-out
agreement), and black wins 79% of games, so the value signal is skewed.

## Decision

Keep v2 as champion; keep `runs/az-r1/model` as a candidate. Before more
rounds, consider larger visit budgets (sharper targets), a few hundred more
games per round, and playing the candidate against the champion during
generation. Each round costs about 15 minutes of local CPU.

## Round 2 (2026-10-01): promoted

- Self-play: round-1 candidate (`runs/az-r1/model`), **400 simulations/move**,
  500 games, seeds 80000-80499, otherwise as round 1 (with the forced-move
  target fix in code). 15,275 positions; Black-White-Draw 410-83-7; 1,574 s.
- Training: fine-tune az-r1 for 2,000 x 256 updates on round-2 data plus
  round-1 and teacher-v2 replay; same optimizer and targets; 385 s.
- Held-out round-2 games: policy accuracy 63.2%, value MAE 0.59 (constant
  baseline 1.00), decisive-game sign accuracy 78%.
- **Gate vs v2 (guided MCTS, 0.2 s/move, seeds 6000-6049): 61-37-2, 62.0%,
  95% CI 52-71%.** The interval excludes 50%, so `runs/az-r2/model` is
  promoted to champion.

Doubling simulations sharpened the targets, and a second round compounded the
small round-1 gain. Next rounds should start from az-r2 and gate against it.
Script: `runs/az-r2/run.sh` (self-play, train, gate).

## Round 3 (2026-10-01): not promoted

- Hypothesis: lopsided openings skew value targets (Black won 82% of round-2
  games); sharper targets (600 sims) plus balanced openings give another gain.
- Balanced openings (new, `choose_opening` in `src/connectfive/selfplay.py`,
  test in `tests/test_guided_search.py`): az-r2 rates every random 4-ply
  opening about +0.61 for Black (300 seeds, quartiles 0.54-0.69). Up to 64
  candidate openings (seeds `seed*64+k`) are tried; the first with |value|
  <= 0.3 is used, else the most balanced. Mean opening value drops to ~0.35.
- Self-play: az-r2, **600 simulations**, balance threshold 0.3, 64 attempts,
  500 games, seeds 90000-90499. 22,444 positions (48.9 plies mean; round 1:
  34.7); Black-White-Draw **350-128-22** (70% Black, was 82%); 2,382 s on 9
  workers, concurrent with a website build.
- Training: fine-tune az-r2, 2,000 x 256 updates on round-3 data plus round-2,
  round-1 and teacher-v2 replay; otherwise as round 2; 230 s.
- Held-out round-3 games: policy accuracy 56.4%, value MAE 0.63 (constant
  baseline 0.78), decisive-game sign accuracy 69%.
- **Gate vs az-r2 (guided MCTS, 0.2 s/move, seeds 7000-7049, 100 paired
  games): 53-45-2, 54.0%, 95% CI 44-63%.** Not promoted; az-r2 stays champion.
- Provenance: base commit and dirty diff in `runs/az-r3/`; script
  `runs/az-r3/run.sh`. Apple M1 Pro CPU, $0.

Interpretation: balancing worked as a data fix (more White wins, longer and
less one-sided games, lower constant baseline), but one round of it did not
produce a measurable strength gain over the champion. The result is the same
size as round 1's non-significant 54.5%. Held-out metrics are not comparable
to round 2's because the validation games themselves changed. Two variables
moved at once (simulations, openings), so neither effect is isolated.

Next: keep az-r3 as a candidate; run round 4 from az-r3 with the same
settings so gains can compound as in round 1 to round 2, then gate against
az-r2. If that also fails, ablate openings vs simulations.
