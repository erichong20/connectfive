// Parity tests: the TypeScript engine must match the Python implementation.
// Fixtures come from scripts/export_web_model.py. Run with `npm run test:engine`.
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';

const root = new URL('..', import.meta.url).pathname;
const outfile = join(mkdtempSync(join(tmpdir(), 'engine-')), 'engine.mjs');
await build({ entryPoints: [join(root, 'lib/engine/index.ts')], bundle: true, format: 'esm', outfile, logLevel: 'error' });
const engine = await import(pathToFileURL(outfile).href);

const fixtures = JSON.parse(readFileSync(join(root, 'tests/fixtures/engine-parity.json'), 'utf8'));
const manifest = JSON.parse(readFileSync(join(root, 'public/models/az-r2/model.json'), 'utf8'));
const weights = readFileSync(join(root, 'public/models/az-r2/weights.bin'));
const network = () => engine.Network.fromBuffers(
  manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength),
);
const KINDS = ['QUIET', 'WIN', 'LOSS', 'FORCE_WIN', 'BLOCK', 'DEFEND'];

function boardFor(moves) {
  const board = new engine.PatternBoard();
  for (const action of moves) board.play(engine.ACTION_TO_INDEX[action]);
  return board;
}

test('line levels match Python on random windows', () => {
  for (const { window, level } of fixtures.windows) assert.equal(engine.lineLevelForTest(window), level, window.join(''));
});

test('incremental board matches a rebuilt board and Python levels', () => {
  for (const testCase of fixtures.cases) {
    const board = boardFor(testCase.moves);
    const rebuilt = engine.PatternBoard.fromArray(board.toArray(), board.player);
    assert.equal(rebuilt.hash, board.hash);
    assert.deepEqual([...rebuilt.candidates].sort((a, b) => a - b), [...board.candidates].sort((a, b) => a - b));
    for (const [action, [black, white]] of Object.entries(testCase.levels)) {
      const index = engine.ACTION_TO_INDEX[Number(action)];
      assert.deepEqual([...board.levels[1].subarray(index * 4, index * 4 + 4)], black);
      assert.deepEqual([...board.levels[2].subarray(index * 4, index * 4 + 4)], white);
    }
    // Undo restores everything.
    const moves = board.moves.length;
    for (let i = 0; i < moves; i += 1) board.undo();
    assert.equal(board.hash, 0);
    assert.equal(board.candidates.size, 0);
  }
});

test('tactics match Python move generation and VCF', () => {
  for (const testCase of fixtures.cases) {
    const board = boardFor(testCase.moves);
    const tactics = new engine.Tactics(board);
    const { kind, moves } = tactics.generate();
    assert.equal(KINDS[kind], testCase.kind);
    let actions = moves.map((index) => engine.INDEX_TO_ACTION[index]);
    if (testCase.kind === 'BLOCK') actions = actions.sort((a, b) => a - b);
    assert.deepEqual(actions, testCase.generated);
    const vcf = kind === 0 ? tactics.vcf(4) : null;
    assert.equal(vcf === null ? null : engine.INDEX_TO_ACTION[vcf], testCase.vcf);
  }
});

test('network outputs match Python', () => {
  const net = network();
  let worst = 0;
  for (const testCase of fixtures.cases) {
    const { logits, value } = net.evaluate(boardFor(testCase.moves));
    for (let i = 0; i < 225; i += 1) worst = Math.max(worst, Math.abs(logits[i] - testCase.logits[i]));
    assert.ok(Math.abs(value - testCase.value) < 1e-3, `value ${value} vs ${testCase.value}`);
  }
  assert.ok(worst < 1e-3, `max logit difference ${worst}`);
});

test('fixed-size MCTS picks the same move as Python', () => {
  let agree = 0;
  for (const testCase of fixtures.cases) {
    const result = engine.guidedMcts(boardFor(testCase.moves), network(), {
      maxSimulations: fixtures.simulations, timeLimitMs: 1e9,
    });
    if (result.actions[0] === testCase.mcts_actions[0]) agree += 1;
  }
  // Float rounding can flip exact visit ties, so allow a small mismatch rate.
  assert.ok(agree >= fixtures.cases.length - 2, `${agree}/${fixtures.cases.length} agree`);
});

test('MCTS speed (informational)', () => {
  const net = network();
  const board = boardFor(fixtures.cases[0].moves);
  const result = engine.guidedMcts(board, net, { timeLimitMs: 1_000 });
  console.log(`  ${result.simulations} simulations in ${result.elapsedMs.toFixed(0)} ms, ${net.calls} network calls`);
});

test('SIMD and plain TypeScript convolutions agree', () => {
  const simd = network();
  const plain = engine.Network.fromBuffers(
    manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength), false,
  );
  assert.equal(simd.backend, 'wasm-simd');
  assert.equal(plain.backend, 'js');
  for (const testCase of fixtures.cases.slice(0, 5)) {
    const a = simd.evaluate(boardFor(testCase.moves));
    const b = plain.evaluate(boardFor(testCase.moves));
    for (let i = 0; i < 225; i += 1) assert.ok(Math.abs(a.logits[i] - b.logits[i]) < 1e-3);
    assert.ok(Math.abs(a.value - b.value) < 1e-3);
  }
});
