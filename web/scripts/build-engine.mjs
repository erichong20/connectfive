// Bundle the neural-bot Web Worker (lib/engine/bot.worker.ts) into a static
// file under public/, so the page can load it without bundler worker support.
import { build } from 'esbuild';

const root = new URL('..', import.meta.url).pathname;
await build({
  entryPoints: [`${root}lib/engine/bot.worker.ts`],
  outfile: `${root}public/engine/bot-worker.js`,
  bundle: true,
  format: 'esm',
  target: 'es2020',
  minify: true,
  legalComments: 'none',
  logLevel: 'warning',
});
console.log('wrote public/engine/bot-worker.js');
