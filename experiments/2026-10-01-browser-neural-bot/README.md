# Browser neural bot (az-r2 guided MCTS in TypeScript)

## Goal

Make the current champion, guided MCTS with the AlphaZero-lite round-2
network (`runs/az-r2/model`), playable on the website. Keep it faithful to the
Python implementation and fast enough to reach the simulation budget that
produced the measured strength.

## Implementation

- `scripts/export_web_model.py`: writes `web/public/models/az-r2/model.json`
  and `weights.bin` (133,388 float32 parameters, 533 KB), plus parity
  fixtures (`web/tests/fixtures/engine-parity.json`, 30 positions from az-r2
  self-play games, seed 0).
- `web/lib/engine/`: TypeScript ports of the pattern board (`patterns.ts`),
  tactics and VCF (`tactics.ts`), network forward pass (`network.ts`), and
  guided MCTS (`mcts.ts`). The 3x3 convolutions run as im2col plus a
  hand-written WebAssembly SIMD matrix multiply (`gemm.wat`, compiled to a
  480-byte module by `npm run build:wasm`), with a plain TypeScript fallback.
- `web/lib/engine/bot.worker.ts`: runs the search in a Web Worker. It is
  bundled to `web/public/engine/bot-worker.js` by `npm run build:engine`,
  because vinext does not rewrite `new Worker(new URL(...))`.
- `web/app/page.tsx`: a new default "Neural" opponent: 1.5 s or 600
  simulations per move, whichever comes first. If the worker fails, the page
  falls back to Look-ahead and says so.

## Verification

`npm run test:engine` (also in CI) checks, against Python:

| Check | Result |
| --- | --- |
| Line levels, 300 random windows | identical |
| Incremental levels, candidates, hash, undo on 30 positions | identical |
| Tactical move generation and depth-4 VCF | identical |
| Network logits and value | max difference < 1e-3 |
| SIMD vs plain TypeScript network | < 1e-3 |
| 64-simulation MCTS chosen move | at least 28/30 agree (ties may flip under float rounding) |

## Simulation budget needed (Python, az-r2 vs `pattern` at 0.2 s/move)

40 games per row, seeds 7000-7019, four-ply paired openings; six matches ran
in parallel, which affects the time-limited `pattern` opponent.

| Fixed simulations | W-L-D | Score | 95% CI |
| --- | --- | --- | --- |
| 32 | 17-23-0 | 42.5% | 29-58% |
| 64 | 20-20-0 | 50.0% | 35-65% |
| 128 | 24-15-1 | 61.3% | 46-75% |

The earlier 100-game result (63%) used about 150 simulations per move. The
browser therefore needs at least about 150 simulations per move.

## Speed (Apple M1 Pro)

| Engine | Simulations per second |
| --- | --- |
| Python + XLA | about 800 |
| TypeScript, plain loops (Node 20) | 37 |
| TypeScript + WASM SIMD (Node 20) | 227-424, depending on position and cache |
| Browser (Chromium, worker, first move including model load) | 300 in 511 ms |

## Limitations

- Strength in the browser is inferred from parity and simulation counts, not
  from human games or a browser-vs-Python match.
- Slower devices will reach fewer simulations within 1.5 s.
- Root move ties are broken randomly, so play is not fully deterministic.
- The bot is strongest as Black, as in self-play.
- `npm run dev` and `npm run build` need Node 22+ (vinext uses `fs.glob`).

## Reproduce

```bash
./.venv/bin/python scripts/export_web_model.py runs/az-r2/model
cd web && npm run build:wasm && npm run build:engine && npm run test:engine
```
