# Compact network overfit check

## Hypothesis

The tested policy/value training path can memorize eight fixed tactical
positions, including both policy distributions and distinct scalar values, on
local CPU in seconds.

## Reproduction

```bash
python scripts/overfit_network.py --steps 200 --learning-rate 0.003 --seed 0
```

The checkpoint is written to `runs/network-overfit/model.{json,msgpack}` and is
intentionally ignored by Git.

## Configuration

- 15x15 exact-five Gomoku.
- Four residual blocks, 32 trunk channels, 64-unit value hidden layer.
- 133,100 trainable parameters.
- Adam, learning rate 0.003, 200 full-batch updates.
- Eight tactical-suite positions; uniform mass over acceptable moves.
- Synthetic value targets evenly spaced from -1 to 1. These are plumbing labels,
  not game outcomes.
- Seed 0.
- Base commit `91eb99f`; the experiment ran in a dirty worktree containing the
  documented search and neural changes.

## Environment and cost

- Apple arm64 CPU, Darwin 25.6.0.
- Python 3.14.7, JAX 0.11.2, NumPy 2.5.3.
- Flax 0.12.10, Optax 0.2.8.
- Runtime approximately 5.2 seconds after startup/compilation.
- Purchased compute: $0.

## Results

| Step | Total loss | Policy loss | Value loss |
| ---: | ---: | ---: | ---: |
| 1 | 5.8728 | 5.3100 | 0.5628 |
| 50 | 0.4774 | 0.3709 | 0.1065 |
| 100 | 0.3616 | 0.3601 | 0.0015 |
| 150 | 0.3474 | 0.3470 | 0.0004 |
| 200 | 0.3472 | 0.3470 | 0.0002 |

The parameter checkpoint reloaded with the recorded architecture, step, metrics,
and all 133,100 parameters.

## Interpretation

The value head and optimizer can fit distinct examples, and the policy reached
its target entropy. Four positions have two equally acceptable actions, so the
minimum possible mean policy cross-entropy is approximately
`4 * ln(2) / 8 = 0.3466`.

This result validates the training plumbing only. The tiny positions are not an
independent validation set, their values are artificial, and no games were
played. The next experiment must use real complete-game outcomes and hold out
entire games.
