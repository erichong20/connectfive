# Self-play throughput

## Question

Round 5 spent 2.7 h generating 600 self-play games. Where does the time go,
and how much faster can self-play be made without changing what is learned?

## Profile (before)

cProfile and wall-clock splits of `play_selfplay_game` (600 simulations,
single-threaded XLA as in the workers), base commit `e96be90`:

| Model | Time per simulation | Leaf VCF | Network | Other (tree walk, generate) |
| --- | --- | --- | --- | --- |
| az-r2 (32 channels) | 4.6 ms | 88% | 5% | 7% |
| az-r5 (64 channels) | 1.3 ms | ~50% | ~25% | ~25% |

The plan's premise was wrong: the network is not the bottleneck. The depth-4
VCF proof attempted at every new leaf was (about 200 VCF nodes per leaf, each
a `PatternBoard.play` with pattern refresh plus a full candidate scan).

## Changes

Exact (search output and self-play games unchanged):

1. `PatternBoard.undo` restores the pattern entries its move changed instead
   of recomputing them; the refresh loop is inlined
   (`src/connectfive/patterns.py`). Boards built with `from_array` fall back
   to recomputation when undone past their loaded stones.
2. VCF looks for the opponent's forced replies only on the four's own lines
   (there were no winning cells before the four, and levels change only on
   its lines) instead of scanning every candidate
   (`src/connectfive/pattern_search.py`).
3. VCF successes are cached by (position, depth) next to the existing failure
   cache, and both caches persist across searches via `VcfCache`
   (`src/connectfive/guided_search.py`): per self-play worker and per
   `GuidedAgent`. Cached entries are facts about a position, so answers do
   not change; only the reported `vcf_nodes` counts do.

Not exact (changes the data):

4. Playout-cap randomisation (KataGo): with `--full-search-fraction f`, each
   move gets the full search (with root noise) with probability f and becomes
   a training position; other moves use `--fast-simulations` without noise
   and are not recorded. Default f = 1.0 keeps the previous behaviour and does
   not consume random numbers.

Tests: `tests/test_pattern_search.py` (undo restores exactly what a rebuild
computes, including undo past `from_array` stones) and
`tests/test_guided_search.py` (playout cap records only full searches).

## Verification

- 8 positions, 800 fixed simulations, az-r5: root visit counts, actions and
  scores byte-identical before and after changes 1-2.
- 3 complete self-play games each with az-r2 and az-r5: moves and policy
  targets byte-identical before and after changes 1-3.

## Results

Single-process, 3 games (seeds 110010-110012):

| Model | Before | After 1-3 | Speed-up |
| --- | --- | --- | --- |
| az-r2 | 4,611 us/sim | 1,130 us/sim | 4.1x |
| az-r5 | 1,320 us/sim | 622 us/sim | 2.1x |

Full pipeline, az-r5, 9 workers, 45 games (seeds 110000-110044), 600
simulations, balanced openings, draw cap 150 (`runs/speed-bench/`):

| Configuration | Seconds | Games/h | Recorded positions/h | Black-White-Draw |
| --- | --- | --- | --- | --- |
| Round 5 actual (600 games) | 9,725 | 222 | 10,000 | 379-179-42 |
| Changes 1-3, all moves full | 276 | 587 | 28,100 | 32-8-5 |
| Changes 1-4, f = 0.25, fast 100 | 84 | 1,930 | 17,300 | 22-22-1 |

Network share after changes 1-3 (az-r5) is about 40%; batching 16 positions
measured 0.37 ms vs 1.08 ms per position single-threaded, so batched
evaluation could save roughly a further 25%. Not implemented: it needs a
restructured search for a smaller gain than the changes above.

## Limitations

- The 45-game samples are small; game lengths and colour results differ
  between them (mean 51.9 vs 41.2 plies), so per-hour rates are approximate.
- Playout-cap games are played mostly with 100-simulation moves, so their
  trajectories are weaker and more varied than full-search games. KataGo found
  this a net gain for value learning (many more independent games), but here
  it is unmeasured until a round is gated.
- Benchmarks ran on an otherwise idle Apple M1 Pro, CPU only, $0.

## Next decision

Run round 6 from `runs/az-r5/model` with changes 1-4 (f = 0.25, 100 fast
simulations), enough games for about 25k recorded positions, `--hold-out-extra`,
per-game value weighting, and the 200-game gate against az-r2. Expected
self-play time: under 1.5 h instead of about 5 h.
