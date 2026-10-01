# Supervised warm-start v0 and v1

## Hypothesis

A compact residual network trained on complete Tactical-versus-Negamax games
will generalize to held-out games and produce a legal raw policy stronger than
Random without purchased compute.

## Protocol

- Exact-five 15x15 Gomoku; four-ply paired randomized openings.
- Tactical versus 100-node, depth-3, width-10 Negamax teachers.
- Openings excluded from imitation labels.
- Deterministic whole-game 75/25 train/validation split, seed 0.
- Eight D4 symmetries sampled during training.
- 4-block, 32-channel, 133,100-parameter network.
- Greedy raw-policy evaluation with legal masking and paired colors.
- Generation seeds 300 (v0) and 700 (v1); training/split seed 0.
- Gameplay seeds 500/600 (v0 Random/Tactical) and 800/900 (v1).
- Base commit `91eb99f`; dirty worktree containing documented classical and
  neural changes.
- Local Apple arm64 CPU on macOS 26.6.2; purchased compute $0.
- Python 3.14.7, JAX 0.11.2, NumPy 2.5.3, Flax 0.12.10, Optax 0.2.8.

## Runs

| Measurement | v0 smoke | v1 data scale-up |
| --- | ---: | ---: |
| Games | 8 | 32 |
| Examples | 562 | 3,024 |
| Teacher W-L-D (Negamax) | 3-4-1 | 11-14-7 |
| Generation time | 43.4 s | 219.9 s |
| Updates / batch | 500 / 64 | 1,000 / 128 |
| Training time | 20.4 s | 61.9 s |
| Train policy accuracy | 81.2% | 51.6% |
| Validation policy accuracy | 15.5% | 39.9% |
| Train value MAE | 0.08 | 0.17 |
| Validation value MAE | 0.82 | 1.12 |
| Versus Random | 20-0 | 20-0 |
| Versus Tactical | 0-10 | 0-20 |

The v1 gameplay evaluations used seeds 800 and 900 respectively, 20 games each,
and four-ply paired openings. The network averaged 21.5 ms per move versus
Random and 20.0 ms per move versus Tactical, with no illegal actions.

## Interpretation

The hypothesis was only partly supported. The raw policy decisively beat Random,
and more games improved held-out policy accuracy. It did not approach Tactical
in direct play, and the value head did not generalize. Position count overstates
independent value evidence because every position in one game shares its final
outcome.

## Decision

Preserve v1 as the first learned gameplay baseline. Before MCTS, inspect where
its Tactical games first diverge, improve data diversity and teacher target
ambiguity, and test a better-balanced value dataset. Scaling model width is not
justified by this evidence.
