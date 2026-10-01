/// <reference lib="webworker" />
// Runs the neural bot (guided MCTS) off the main thread so the page stays responsive.

import { guidedMcts } from './mcts';
import { Network, type ModelManifest } from './network';
import { PatternBoard } from './patterns';

export type BotRequest = {
  id: number;
  /** Flat 225-cell board: -1 empty, 0 black, 1 white. */
  board: number[];
  player: 0 | 1;
  modelUrl: string;
  timeLimitMs: number;
  maxSimulations: number;
};

export type BotResponse =
  | { id: number; action: number; simulations: number; elapsedMs: number; value: number; backend: string }
  | { id: number; error: string };

let loading: { url: string; network: Promise<Network> } | null = null;

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
    const network = await loadNetwork(request.modelUrl);
    const board = PatternBoard.fromArray(request.board, request.player);
    const result = guidedMcts(board, network, {
      timeLimitMs: request.timeLimitMs,
      maxSimulations: request.maxSimulations,
    });
    const action = result.actions[Math.floor(Math.random() * result.actions.length)];
    const response: BotResponse = {
      id: request.id,
      action,
      simulations: result.simulations,
      elapsedMs: result.elapsedMs,
      value: result.value,
      backend: network.backend,
    };
    self.postMessage(response);
  } catch (error) {
    self.postMessage({ id: request.id, error: error instanceof Error ? error.message : String(error) });
  }
};
