# Browser benchmark: az-r7 vs az-r2 at the website's budget

## Question

az-r7 beat az-r2 in Python at 0.2 s/move (62.7% and 60.0% over two 200-game
gates). The website gives the bot 1.5 s or 600 simulations per move in a
TypeScript/WASM engine, where the 64-channel network is slower. Is az-r7
stronger *on the website*?

## Method

1. Export: `scripts/export_web_model.py runs/az-r7/model --name az-r7 --out
   web/public/models/az-r7 --games runs/az-r7/games.json --fixtures
   web/tests/fixtures/engine-parity-az-r7.json` (356,492 float32 parameters,
   1.43 MB vs az-r2's 0.53 MB).
2. Parity: `web/tests/engine.test.mjs` now runs the network, MCTS and SIMD
   checks for every shipped model (az-r2 and az-r7).
3. Speed: `web/scripts/bench-engine.mjs 1500 12` (Node, TypeScript + WASM
   SIMD, Apple M1 Pro): simulations reached in 1.5 s on the same 12 quiet
   positions (az-r2 parity fixtures).
4. Strength at those budgets: `scripts/promotion_gate.py` with fixed
   simulations, az-r7 at 300 vs az-r2 at 600, 200 paired games, seeds
   14000-14099 (`--simulations` / `--opponent-simulations`, added for this).

## Results

Parity (all pass): az-r7 network logits and value within 1e-3 of Python,
fixed-size MCTS agrees with Python on at least 28/30 positions, SIMD matches
plain TypeScript.

| Model | Mean | Min | Median | Max simulations in 1.5 s |
| --- | --- | --- | --- | --- |
| az-r2 | 1,995 | 599 | 820 | 9,007 |
| az-r7 | 722 | 239 | 312 | 3,115 |

The site caps at 600, so az-r2 always plays at the cap; az-r7 typically gets
about 300.

| Match | W-L-D | Score | 95% CI |
| --- | --- | --- | --- |
| az-r7 @ 300 sims vs az-r2 @ 600 sims | 94-101-5 | 48.2% | 41.4-55.1% |

az-r7 won 80/100 as Black and 14/100 as White.

## Interpretation

At equal wall-clock in Python, az-r7 is clearly stronger; at the website's
budget it is not, because it reaches half the simulations. Keep az-r2 on the
website. Either more simulations for az-r7 (faster engine or longer think
time) or a smaller strong network is needed before switching.

## Limitations

- Speeds are from Node on an M1 Pro; browsers and slower devices vary, and
  phones would reach fewer simulations for both models.
- Simulation budgets are fixed medians; real play varies by position.
- The TypeScript engine does not yet have the Python self-play speed-ups
  (undo restore, line-local VCF replies, shared VCF cache), which removed
  most of the non-network cost in Python.

## Next decision

Port the exact Python speed-ups to the TypeScript engine, re-measure az-r7's
simulations at 1.5 s, and repeat this match at the new budgets. Separately,
consider a 3 s think time for az-r7 as a "stronger" setting.
