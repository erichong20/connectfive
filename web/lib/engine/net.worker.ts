/// <reference lib="webworker" />
// One network evaluator for the parallel neural bot: receives input features,
// returns policy logits and value. Several of these run side by side.

import { Network, type ModelManifest } from './network';

export type NetRequest = { id: number; modelUrl: string; features: Float32Array[] };
export type NetResponse =
  | { id: number; results: { logits: Float32Array; value: number }[] }
  | { id: number; error: string };

let loading: { url: string; network: Promise<Network> } | null = null;

function loadNetwork(url: string) {
  if (loading?.url !== url) {
    const base = url.endsWith('/') ? url : `${url}/`;
    loading = {
      url,
      network: Promise.all([
        fetch(`${base}model.json`).then((response) => response.json() as Promise<ModelManifest>),
        fetch(`${base}weights.bin`).then((response) => response.arrayBuffer()),
      ]).then(([manifest, weights]) => Network.fromBuffers(manifest, weights)),
    };
  }
  return loading.network;
}

self.onmessage = async (event: MessageEvent<NetRequest>) => {
  const { id, modelUrl, features } = event.data;
  try {
    const network = await loadNetwork(modelUrl);
    const results = features.map((input) => {
      const { logits, value } = network.evaluateFeatures(input);
      return { logits: Float32Array.from(logits), value };
    });
    self.postMessage({ id, results } satisfies NetResponse, results.map((r) => r.logits.buffer));
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) } satisfies NetResponse);
  }
};
