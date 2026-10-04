// Port of GuidedMCTS (src/connectfive/guided_search.py): PUCT tree search
// with network priors and values, and exact pattern tactics as shortcuts.
// Values are always from the perspective of the side to move at that node.
//
// `guidedMcts` evaluates one leaf at a time. `guidedMctsBatched` picks several
// leaves under virtual loss and evaluates them together (e.g. on a pool of
// network workers), matching GuidedMCTS(batch_size=...) in Python.

import type { Network } from './network';
import { ACTION_TO_INDEX, INDEX_TO_ACTION, PatternBoard } from './patterns';
import { FORCE_WIN, LOSS, QUIET, Tactics, VcfCache, WIN } from './tactics';

class Node {
  visits = 0;
  valueSum = 0;
  children: Map<number, Node> | null = null;
  terminal: number | null = null;
  /** Batched search: picked for network evaluation, not yet expanded. */
  pending = false;
  constructor(public prior: number) {}
  get q() {
    return this.visits ? this.valueSum / this.visits : 0;
  }
}

export type MctsOptions = {
  timeLimitMs?: number;
  maxSimulations?: number;
  cPuct?: number;
  topK?: number;
  leafVcfDepth?: number;
  /** Share VCF results across searches, e.g. across the moves of a game. */
  vcfCache?: VcfCache;
  now?: () => number;
};

export type MctsResult = {
  /** Most-visited root actions (ties included), in board action order. */
  actions: number[];
  /** Root actions sorted by visits. */
  visits: [number, number][];
  /** Root value estimate for the side to move, in [-1, 1]. */
  value: number;
  simulations: number;
  elapsedMs: number;
  reason: 'opening' | 'forced' | 'mcts';
};

export type Evaluation = { logits: ArrayLike<number>; value: number };

/** A leaf waiting for the network: its position's hash and input features. */
export type LeafRequest = { hash: number; features: Float32Array };

export type BatchedOptions = MctsOptions & {
  /** Leaves picked per batch under virtual loss (default 4). */
  batchSize?: number;
  virtualLoss?: number;
};

type Generated = { kind: number; moves: number[] };

function makeSearch(board: PatternBoard, options: MctsOptions) {
  const cPuct = options.cPuct ?? 1.5;
  const topK = options.topK ?? 16;
  const leafVcfDepth = options.leafVcfDepth ?? 4;
  const tactics = new Tactics(board, options.vcfCache);

  /** Resolve full boards and exact tactics: a value, or the moves needing the network. */
  const expandTactics = (node: Node): number | Generated => {
    if (board.isFull()) {
      node.terminal = 0;
      return 0;
    }
    const generated = tactics.generate();
    const { kind, moves } = generated;
    if (kind === WIN || kind === FORCE_WIN) {
      node.terminal = 1;
      node.children = new Map([[moves[0], new Node(1)]]);
      return 1;
    }
    if (kind === LOSS) {
      node.terminal = -1;
      node.children = new Map([[moves[0], new Node(1)]]);
      return -1;
    }
    if (kind === QUIET && leafVcfDepth) {
      const win = tactics.vcf(leafVcfDepth);
      if (win !== null) {
        node.terminal = 1;
        node.children = new Map([[win, new Node(1)]]);
        return 1;
      }
    }
    return generated;
  };

  const attach = (node: Node, { kind, moves }: Generated, { logits, value }: Evaluation): number => {
    const logit = (index: number) => logits[INDEX_TO_ACTION[index]];
    if (kind === QUIET) {
      // Stable sort, as in Python: equal logits keep the pattern ordering.
      moves = moves
        .map((move, position) => ({ move, position }))
        .sort((a, b) => logit(b.move) - logit(a.move) || a.position - b.position)
        .slice(0, topK)
        .map(({ move }) => move);
    }
    let max = -Infinity;
    for (const move of moves) max = Math.max(max, logit(move));
    const weights = moves.map((move) => Math.exp(logit(move) - max));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    node.children = new Map(moves.map((move, i) => [move, new Node(weights[i] / total)]));
    return value;
  };

  const select = (node: Node): [number, Node] => {
    const rootVisits = Math.sqrt(Math.max(1, node.visits));
    const fpu = -node.q - 0.2;
    let best: [number, Node] | null = null;
    let bestScore = -Infinity;
    for (const [move, child] of node.children as Map<number, Node>) {
      const q = child.visits ? -child.q : fpu;
      const score = q + (cPuct * child.prior * rootVisits) / (1 + child.visits);
      if (score > bestScore) {
        best = [move, child];
        bestScore = score;
      }
    }
    return best as [number, Node];
  };

  const backup = (path: Node[], value: number) => {
    for (let i = path.length - 1; i >= 0; i -= 1) {
      path[i].visits += 1;
      path[i].valueSum += value;
      value = -value;
    }
  };

  /** Walk down to a leaf; returns the path and the number of moves played on `board`. */
  const descend = (root: Node): [Node[], number] => {
    const path = [root];
    let node = root;
    let played = 0;
    while (node.children !== null && node.terminal === null && !node.pending) {
      const [move, child] = select(node);
      board.play(move);
      played += 1;
      node = child;
      path.push(node);
    }
    return [path, played];
  };

  const undo = (played: number) => {
    for (let i = 0; i < played; i += 1) board.undo();
  };

  return { expandTactics, attach, select, backup, descend, undo };
}

function finish(root: Node, simulations: number, elapsedMs: number, reason: MctsResult['reason']): MctsResult {
  const children = [...(root.children as Map<number, Node>)];
  const most = Math.max(...children.map(([, child]) => child.visits));
  return {
    actions: children
      .filter(([, child]) => child.visits === most)
      .map(([move]) => INDEX_TO_ACTION[move])
      .sort((a, b) => a - b),
    visits: children
      .map(([move, child]) => [INDEX_TO_ACTION[move], child.visits] as [number, number])
      .sort((a, b) => b[1] - a[1]),
    value: root.q,
    simulations,
    elapsedMs,
    reason,
  };
}

/** Root setup shared by both searches; returns a reason if no search is needed. */
function openRoot(board: PatternBoard, root: Node, rootValue: () => number): MctsResult['reason'] | null {
  if (!board.candidates.size) {
    const child = new Node(1);
    child.visits = 1;
    root.children = new Map([[ACTION_TO_INDEX[7 * 15 + 7], child]]);
    return 'opening';
  }
  root.visits = 1;
  root.valueSum = rootValue();
  if (root.terminal !== null || (root.children as Map<number, Node>).size === 1) return 'forced';
  return null;
}

export function guidedMcts(board: PatternBoard, network: Network, options: MctsOptions = {}): MctsResult {
  const now = options.now ?? (() => performance.now());
  const timeLimit = options.timeLimitMs ?? 1_000;
  const maxSimulations = options.maxSimulations ?? 100_000;
  const search = makeSearch(board, options);
  const started = now();

  const expand = (node: Node): number => {
    const resolved = search.expandTactics(node);
    return typeof resolved === 'number' ? resolved : search.attach(node, resolved, network.evaluate(board));
  };

  const root = new Node(1);
  let simulations = 0;
  const reason = openRoot(board, root, () => expand(root));
  if (reason === null) {
    const deadline = started + timeLimit;
    while (simulations < maxSimulations && now() < deadline) {
      const [path, played] = search.descend(root);
      const leaf = path[path.length - 1];
      const value = leaf.terminal !== null ? leaf.terminal : expand(leaf);
      search.undo(played);
      search.backup(path, value);
      simulations += 1;
    }
  }
  return finish(root, simulations, now() - started, reason ?? 'mcts');
}

/**
 * Batched search: pick up to `batchSize` leaves under virtual loss (each picked
 * path gets a temporary visit, and every non-root node on it a temporary
 * +virtualLoss for its own side, i.e. a loss for the parent choosing it), then
 * evaluate the pending leaves together and back them up. Tactical leaves are
 * backed up at once; picking stops at a leaf that is already pending.
 */
export async function guidedMctsBatched(
  board: PatternBoard,
  encode: (board: PatternBoard) => LeafRequest,
  evaluateBatch: (leaves: LeafRequest[]) => Promise<Evaluation[]>,
  options: BatchedOptions = {},
): Promise<MctsResult> {
  const now = options.now ?? (() => performance.now());
  const timeLimit = options.timeLimitMs ?? 1_000;
  const maxSimulations = options.maxSimulations ?? 100_000;
  const batchSize = options.batchSize ?? 4;
  const virtualLoss = options.virtualLoss ?? 1;
  const search = makeSearch(board, options);
  const started = now();

  const root = new Node(1);
  let reason: MctsResult['reason'] | null = null;
  if (!board.candidates.size) {
    reason = openRoot(board, root, () => 0);
  } else {
    root.visits = 1;
    const resolved = search.expandTactics(root);
    if (typeof resolved === 'number') root.valueSum = resolved;
    else {
      const [evaluation] = await evaluateBatch([encode(board)]);
      root.valueSum = search.attach(root, resolved, evaluation);
    }
    if (root.terminal !== null || (root.children as Map<number, Node>).size === 1) reason = 'forced';
  }

  let simulations = 0;
  if (reason === null) {
    const deadline = started + timeLimit;
    while (simulations < maxSimulations && now() < deadline) {
      const size = Math.min(batchSize, maxSimulations - simulations);
      const pending: { path: Node[]; generated: Generated; request: LeafRequest }[] = [];
      for (let pick = 0; pick < size; pick += 1) {
        const [path, played] = search.descend(root);
        const leaf = path[path.length - 1];
        if (leaf.pending) {
          search.undo(played);
          break;
        }
        let value: number | null = leaf.terminal;
        let generated: Generated | null = null;
        let request: LeafRequest | null = null;
        if (value === null) {
          const resolved = search.expandTactics(leaf);
          if (typeof resolved === 'number') value = resolved;
          else {
            generated = resolved;
            request = encode(board);
          }
        }
        search.undo(played);
        if (generated === null || request === null) {
          search.backup(path, value as number);
          simulations += 1;
          continue;
        }
        leaf.pending = true;
        path.forEach((node, depth) => {
          node.visits += 1;
          if (depth) node.valueSum += virtualLoss;
        });
        pending.push({ path, generated, request });
      }
      if (pending.length) {
        const evaluations = await evaluateBatch(pending.map(({ request }) => request));
        pending.forEach(({ path, generated }, i) => {
          path.forEach((node, depth) => {
            node.visits -= 1;
            if (depth) node.valueSum -= virtualLoss;
          });
          const leaf = path[path.length - 1];
          leaf.pending = false;
          search.backup(path, search.attach(leaf, generated, evaluations[i]));
          simulations += 1;
        });
      }
    }
  }
  return finish(root, simulations, now() - started, reason ?? 'mcts');
}
