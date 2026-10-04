// Simulations per move for each shipped model at the site's budget, on the same positions.
// Usage: node scripts/bench-engine.mjs [timeLimitMs=1500] [positions=12]
import { readFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';

const root = new URL('..', import.meta.url).pathname;
const outfile = join(mkdtempSync(join(tmpdir(), 'engine-')), 'engine.mjs');
await build({ entryPoints: [join(root, 'lib/engine/index.ts')], bundle: true, format: 'esm', outfile, logLevel: 'error' });
const engine = await import(pathToFileURL(outfile).href);
const timeLimitMs = Number(process.argv[2] ?? 1500);
const count = Number(process.argv[3] ?? 12);
const cases = JSON.parse(readFileSync(join(root, 'tests/fixtures/engine-parity.json'), 'utf8')).cases
  .filter((c) => c.kind === 'QUIET').slice(0, count);

for (const name of ['az-r2', 'az-r7']) {
  const manifest = JSON.parse(readFileSync(join(root, `public/models/${name}/model.json`), 'utf8'));
  const weights = readFileSync(join(root, `public/models/${name}/weights.bin`));
  const net = engine.Network.fromBuffers(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
  const sims = [];
  for (const testCase of cases) {
    const board = new engine.PatternBoard();
    for (const action of testCase.moves) board.play(engine.ACTION_TO_INDEX[action]);
    sims.push(engine.guidedMcts(board, net, { timeLimitMs, maxSimulations: 1e9 }).simulations);
  }
  sims.sort((a, b) => a - b);
  const mean = sims.reduce((a, b) => a + b, 0) / sims.length;
  console.log(`${name}: ${cases.length} quiet positions, ${timeLimitMs} ms: mean ${mean.toFixed(0)} sims, min ${sims[0]}, median ${sims[sims.length >> 1]}, max ${sims.at(-1)}`);
}
