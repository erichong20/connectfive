// A pool of network workers for guidedMctsBatched: each batch is split across
// the workers and evaluated in parallel. Results are cached by position hash.

import type { Evaluation, LeafRequest } from './mcts';
import type { NetRequest, NetResponse } from './net.worker';

export class NetworkPool {
  private readonly workers: Worker[];
  private readonly cache = new Map<number, Evaluation>();
  private nextId = 0;
  private readonly waiting = new Map<number, (response: NetResponse) => void>();
  /** Set when any worker fails to start or crashes; the pool is then unusable. */
  broken: string | null = null;

  constructor(
    readonly modelUrl: string,
    readonly size: number,
    workerUrl: string,
    readonly timeoutMs = 10_000,
  ) {
    this.workers = Array.from({ length: size }, () => {
      const worker = new Worker(workerUrl, { type: 'module' });
      worker.onmessage = (event: MessageEvent<NetResponse>) => {
        this.waiting.get(event.data.id)?.(event.data);
        this.waiting.delete(event.data.id);
      };
      worker.onerror = (event) => this.fail(event.message || 'network worker failed');
      return worker;
    });
  }

  private fail(message: string) {
    this.broken ??= message;
    for (const [id, settle] of this.waiting) settle({ id, error: message });
    this.waiting.clear();
  }

  private call(worker: Worker, features: Float32Array[]): Promise<Evaluation[]> {
    if (this.broken) return Promise.reject(new Error(this.broken));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      // The first call also loads the model, so allow generously before giving up.
      const timer = setTimeout(() => this.fail('network worker timed out'), this.timeoutMs);
      this.waiting.set(id, (response) => {
        clearTimeout(timer);
        if ('error' in response) reject(new Error(response.error));
        else resolve(response.results);
      });
      worker.postMessage({ id, modelUrl: this.modelUrl, features } satisfies NetRequest);
    });
  }

  async evaluateBatch(leaves: LeafRequest[]): Promise<Evaluation[]> {
    const missing = [...new Map(leaves.filter(({ hash }) => !this.cache.has(hash)).map((leaf) => [leaf.hash, leaf])).values()];
    const shares: LeafRequest[][] = this.workers.map(() => []);
    missing.forEach((leaf, i) => shares[i % shares.length].push(leaf));
    const answers = await Promise.all(
      shares.map((share, i) => (share.length ? this.call(this.workers[i], share.map((leaf) => leaf.features)) : Promise.resolve([]))),
    );
    if (this.cache.size > 200_000) this.cache.clear();
    shares.forEach((share, i) => share.forEach((leaf, j) => this.cache.set(leaf.hash, answers[i][j])));
    return leaves.map(({ hash }) => this.cache.get(hash) as Evaluation);
  }

  terminate() {
    for (const worker of this.workers) worker.terminate();
  }
}
