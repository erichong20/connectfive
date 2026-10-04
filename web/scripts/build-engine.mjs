// Bundle the neural-bot Web Worker (lib/engine/bot.worker.ts) into a static
// file under public/, so the page can load it without bundler worker support.
import { build } from 'esbuild';

const root = new URL('..', import.meta.url).pathname;
for (const [entry, out] of [['bot.worker.ts', 'bot-worker.js'], ['net.worker.ts', 'net-worker.js']]) await build({
  entryPoints: [`${root}lib/engine/${entry}`],
  outfile: `${root}public/engine/${out}`,
  bundle: true,
  format: 'esm',
  target: 'es2020',
  minify: true,
  legalComments: 'none',
  logLevel: 'warning',
});
console.log('wrote public/engine/bot-worker.js and net-worker.js');
