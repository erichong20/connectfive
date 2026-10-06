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

## Follow-up: Python speed-ups ported to TypeScript

Ported the exact changes from `experiments/2026-10-03-selfplay-speed/`:
an undo log in `web/lib/engine/patterns.ts` (with recomputation for stones
loaded by `fromArray`), line-local VCF replies and a shared `VcfCache` in
`tactics.ts`, used by `mcts.ts` and kept across moves in `bot.worker.ts`.
Parity still passes (12 tests, including a new undo test on loaded boards).

Effect at 1.5 s, same 12 positions: az-r2 median 820 -> 845 simulations,
az-r7 median 312 -> 320 (mean 722 -> 770). Small, because in the browser
engine the network is 95% (az-r2, 2.35 ms per evaluation) to 98% (az-r7,
6.30 ms) of search time, unlike Python where VCF dominated. The remaining
levers are network speed (a faster WASM GEMM or WebGPU), parallel search
across several Web Workers, or a longer think time.

## Follow-up: parallel network workers (batched search)

Because the network dominates browser search time, the search now evaluates
several leaves at once on a pool of network workers.

- Algorithm (`guidedMctsBatched` in `web/lib/engine/mcts.ts`, and
  `GuidedMCTS(batch_size=...)` in `src/connectfive/guided_search.py`): pick up
  to B leaves before expanding any of them. Each picked path gets a temporary
  extra visit and every non-root node on it a temporary +1 value for its own
  side (a loss for the parent choosing it), so later picks spread out.
  Tactical and terminal leaves are backed up at once; picking stops at a leaf
  that is already pending. `batch_size=1` is the old search, unchanged
  (byte-identical on the reference positions).
- Browser plumbing: `web/lib/engine/net.worker.ts` (one network per worker),
  `pool.ts` (splits each batch across workers, caches by position hash,
  fails fast on worker errors or a 10 s timeout), `bot.worker.ts` (runs the
  batched search; on any pool failure switches permanently to a serial search
  with the fallback model).
- Parity: Python batched search (B=4) results added to both fixture files;
  the TypeScript batched search picks the same top move on at least 28/30
  positions per model (14 engine tests pass).

In-browser speed (Chromium, Apple M1 Pro, 10 cores; median simulations in
1.5 s over 8 quiet positions, `/engine/` bench page served statically):

| Model | Network workers | Median | Min |
| --- | --- | --- | --- |
| az-r2 | 1 | 827 | 65 |
| az-r7 | 1 | 353 | 270 |
| az-r7 | 2 | 528 | 488 |
| az-r7 | 3 | 782 | 732 |
| az-r7 | 4 | 952 | 856 |
| az-r7 | 6 | 1,399 | 1,182 |

Strength at the site's settings (Python, fixed simulations, 200 paired games,
seeds 16000-16099; `gate-batch4-600v600.json`):

| Match | W-L-D | Score | 95% CI |
| --- | --- | --- | --- |
| az-r7, batched B=4, 600 sims vs az-r2, 600 sims | 124-71-5 | 63.2% | 56.4-69.6% |

Batching costs little: this matches az-r7's unbatched results against az-r2
(62.7% and 60.0%).

**Site change** (`web/app/page.tsx`): with at least 4 logical cores the
neural bot uses az-r7 with min(4, cores - 1) network workers; otherwise, or
if workers fail, az-r2 serially as before. Verified in a local production
build (`npm run build` + Wrangler on port 8788): 600 simulations in about
1.0 s on az-r7, and page moves answered in about 1.6 s. On the Vite dev
server, nested module workers fail to start in this browser; the bot then
falls back to az-r2 within about 1 s, which is how the fallback was tested.

Limitations: speeds are from one fast laptop. Phones report 6-8 cores but
mix fast and slow cores, so they will reach fewer simulations; the time limit
(1.5 s) bounds the wait either way. Strength in the browser is inferred from
Python games at matching simulation counts, not from games played in browsers.

## Follow-up: az-r9 on the website (2026-10-05)

az-r9 (champion; `experiments/2026-10-05-alphazero-lite-r9/`) has the same
architecture as az-r7 (64 channels, 4 blocks), so browser speed is unchanged.
Exported to `web/public/models/az-r9/` with parity fixtures
(`web/tests/fixtures/engine-parity-az-r9.json`); all 19 engine tests pass,
including az-r9 network, serial MCTS and batched MCTS parity with Python.
`web/app/page.tsx` now uses az-r9 on devices with 4+ cores (az-r2 otherwise,
unchanged). Checked in a local production build: 600 simulations in about
1.0 s with 4 network workers, page moves answered in about 1 s. az-r7's
files stay in `web/public/models/` as a baseline for parity tests.
