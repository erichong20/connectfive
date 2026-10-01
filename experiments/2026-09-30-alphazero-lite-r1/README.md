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
