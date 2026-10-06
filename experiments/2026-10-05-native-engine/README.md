# Native search core

## Question

Python self-play spends most of its time outside the network (board updates,
VCF, tree walking). How much faster is a native core, and what limits it?

## Implementation

- `src/connectfive/native/engine.c` (about 600 lines, C11, no dependencies):
  the exact-five pattern board with an undo log, `generate` and `vcf` with
  failure/win caches, and the batched PUCT tree of
  `GuidedMCTS(batch_size=B)` (virtual loss, collision stop, tactical leaves
  backed up immediately). Compiled on first use with the system compiler
  (`-O3`) into an ignored `_build/` directory, loaded with `ctypes`.
- `src/connectfive/native/__init__.py`: `NativeCore` (one C search state, its
  VCF cache kept across moves), `BatchEvaluator` (JAX network on a fixed batch
  size, results cached by position hash, as `NetworkEvaluator` does), and
  `NativeMCTS`, a drop-in for `GuidedMCTS` with batched semantics, including
  root Dirichlet noise drawn from the caller's NumPy generator.
- Wired in as `SelfPlayConfig.native_batch` / `selfplay_round.py
  --native-batch`, `GuidedAgent.native_batch`, and `--native-batch` on the
  gate scripts (applies to both sides). Default off.

## Verification

- 24 positions from round-8 games, az-r7, 200 simulations: identical root
  visit counts to Python for B = 1 and B = 4 (24/24 each).
- `tests/test_guided_search.py::test_native_core_matches_python_batched_search`
  checks the same on a small random network in CI.
- Native self-play with and without the evaluation cache produced identical
  games (same seeds, same results), as expected from an exact cache.

## Results (Apple M1 Pro)

Single search, az-r7, 600 simulations over 16 positions, single-threaded XLA
(measured while other jobs ran, so absolute numbers are noisy):

| Search | Simulations/s |
| --- | --- |
| Python serial | 963 |
| Native, B = 1 | 1,556 |
| Native, B = 4 | 2,395 |
| Native, B = 8 | 2,663 |
| Native, B = 16 | 2,519 |
| Native, B = 32 | 2,166 |

The network is 97% of native search time.

Self-play (`selfplay_round.py`, az-r7, 90 games, 9 workers, 600/100
simulations with a 25% playout cap, balanced openings, draw cap 150):

| Search | Games/hour | Simulations/s |
| --- | --- | --- |
| Python | 1,768 | 3,806 |
| Native B = 8, no evaluation cache | 2,083 | 3,579 |
| Native B = 4, cache | 2,767 | 5,570 |
| Native B = 8, cache | 2,964 | 5,093 |
| Native B = 16, cache | 3,057 | 4,341 |

## Interpretation

- Moving board, tactics and tree to C makes them almost free; the search is
  now network-bound. The single-search gain (2.7x at B = 8) shrinks to 1.7x
  in self-play because Python self-play already reuses network results across
  moves through its hash cache, and short 100-simulation searches batch less
  well.
- Batched search changes the search slightly (virtual loss). At B = 4 in the
  browser this cost no measurable strength (63.2% vs about 61% unbatched
  against az-r2); B = 8 is used for self-play and has not been separately
  gated.
- Further speed needs a faster network evaluation: larger cross-game batches
  in each worker, multithreaded XLA with fewer workers, or a smaller network.

## Next decision

Use `--native-batch 8` for self-play rounds (round 9 is the first).
