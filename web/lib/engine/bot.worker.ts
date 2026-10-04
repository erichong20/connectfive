/// <reference lib="webworker" />
// Runs the neural bot (guided MCTS) off the main thread so the page stays responsive.

import { guidedMcts, guidedMctsBatched, type MctsResult } from './mcts';
import { Network, type ModelManifest } from './network';
import { PatternBoard } from './patterns';
import { NetworkPool } from './pool';
import { VcfCache } from './tactics';

export type BotRequest = {
  id: number;
  /** Flat 225-cell board: -1 empty, 0 black, 1 white. */
  board: number[];
  player: 0 | 1;
  modelUrl: string;
  timeLimitMs: number;
  maxSimulations: number;
  /** Network workers for a parallel (batched) search; 1 or absent searches serially. */
  workers?: number;
  /** URL of the bundled network worker (net-worker.js), required when workers > 1. */
  netWorkerUrl?: string;
  /** Model for a serial search if parallel workers are unavailable. */
  fallbackModelUrl?: string;
};

export type BotResponse =
  | {
      id: number;
      action: number;
      simulations: number;
      elapsedMs: number;
      value: number;
      backend: string;
      /** Model that actually played, and why parallel search was skipped, if it was. */
      modelUrl: string;
      parallelFailed: string | null;
    }
  | { id: number; error: string };

let loading: { url: string; network: Promise<Network> } | null = null;
// VCF results are exact facts about positions, so one cache serves every game.
const vcfCache = new VcfCache();
let pool: NetworkPool | null = null;
let parallelFailed: string | null = null;

function poolFor(modelUrl: string, size: number, workerUrl: string) {
  if (!pool || pool.modelUrl !== modelUrl || pool.size !== size) {
    pool?.terminate();
    pool = new NetworkPool(modelUrl, size, workerUrl);
  }
  return pool;
}

function loadNetwork(url: string) {
  if (loading?.url !== url) {
    const base = url.endsWith('/') ? url : `${url}/`;
    loading = {
      url,
      network: Promise.all([
        fetch(`${base}model.json`).then((response) => {
          if (!response.ok) throw new Error(`model.json: HTTP ${response.status}`);
          return response.json() as Promise<ModelManifest>;
        }),
        fetch(`${base}weights.bin`).then((response) => {
          if (!response.ok) throw new Error(`weights.bin: HTTP ${response.status}`);
          return response.arrayBuffer();
        }),
      ]).then(([manifest, weights]) => Network.fromBuffers(manifest, weights)),
    };
    loading.network.catch(() => {
      loading = null;
    });
  }
  return loading.network;
}

self.onmessage = async (event: MessageEvent<BotRequest>) => {
  const request = event.data;
  try {
    const options = { timeLimitMs: request.timeLimitMs, maxSimulations: request.maxSimulations, vcfCache };
    const workers = request.workers ?? 1;
    const serial = async (modelUrl: string) =>
      [guidedMcts(PatternBoard.fromArray(request.board, request.player), await loadNetwork(modelUrl), options), modelUrl] as const;
    let outcome: readonly [MctsResult, string];
    if (workers > 1 && request.netWorkerUrl && !parallelFailed) {
      try {
        const network = await loadNetwork(request.modelUrl);
        const pool = poolFor(request.modelUrl, workers, request.netWorkerUrl);
        const board = PatternBoard.fromArray(request.board, request.player);
        outcome = [
          await guidedMctsBatched(board, (leaf) => network.encodeRequest(leaf), (leaves) => pool.evaluateBatch(leaves), {
            ...options,
            batchSize: workers,
          }),
          request.modelUrl,
        ];
      } catch (error) {
        // Parallel search is unavailable here: use the serial fallback from now on.
        parallelFailed = error instanceof Error ? error.message : String(error);
        pool?.terminate();
        pool = null;
        outcome = await serial(request.fallbackModelUrl ?? request.modelUrl);
      }
    } else {
      outcome = await serial(parallelFailed && request.fallbackModelUrl ? request.fallbackModelUrl : request.modelUrl);
    }
    const [result, modelUrl] = outcome;
    const network = await loadNetwork(modelUrl);
    const action = result.actions[Math.floor(Math.random() * result.actions.length)];
    const response: BotResponse = {
      id: request.id,
      action,
      simulations: result.simulations,
      elapsedMs: result.elapsedMs,
      value: result.value,
      backend: network.backend,
      modelUrl,
      parallelFailed,
    };
    self.postMessage(response);
  } catch (error) {
    self.postMessage({ id: request.id, error: error instanceof Error ? error.message : String(error) });
  }
};
