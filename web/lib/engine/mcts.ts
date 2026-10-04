// Port of GuidedMCTS (src/connectfive/guided_search.py): PUCT tree search
// with network priors and values, and exact pattern tactics as shortcuts.
// Values are always from the perspective of the side to move at that node.

import type { Network } from './network';
import { ACTION_TO_INDEX, INDEX_TO_ACTION, PatternBoard } from './patterns';
import { FORCE_WIN, LOSS, QUIET, Tactics, VcfCache, WIN } from './tactics';

class Node {
  visits = 0;
  valueSum = 0;
  children: Map<number, Node> | null = null;
  terminal: number | null = null;
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

export function guidedMcts(board: PatternBoard, network: Network, options: MctsOptions = {}): MctsResult {
  const now = options.now ?? (() => performance.now());
  const timeLimit = options.timeLimitMs ?? 1_000;
  const maxSimulations = options.maxSimulations ?? 100_000;
  const cPuct = options.cPuct ?? 1.5;
  const topK = options.topK ?? 16;
  const leafVcfDepth = options.leafVcfDepth ?? 4;
  const tactics = new Tactics(board, options.vcfCache);
  const started = now();

  const expand = (node: Node): number => {
    if (board.isFull()) {
      node.terminal = 0;
      return 0;
    }
    const generated = tactics.generate();
    const kind = generated.kind;
    let moves = generated.moves;
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
    const { logits, value } = network.evaluate(board);
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

  const simulate = (root: Node) => {
    const path = [root];
    let node = root;
    let played = 0;
    while (node.children !== null && node.terminal === null) {
      const [move, child] = select(node);
      board.play(move);
      played += 1;
      node = child;
      path.push(node);
    }
    let value = node.terminal !== null ? node.terminal : expand(node);
    for (let i = 0; i < played; i += 1) board.undo();
    for (let i = path.length - 1; i >= 0; i -= 1) {
      path[i].visits += 1;
      path[i].valueSum += value;
      value = -value;
    }
  };

  const root = new Node(1);
  let reason: MctsResult['reason'] = 'mcts';
  let simulations = 0;
  if (!board.candidates.size) {
    const center = ACTION_TO_INDEX[7 * 15 + 7];
    const child = new Node(1);
    child.visits = 1;
    root.children = new Map([[center, child]]);
    reason = 'opening';
  } else {
    root.visits = 1;
    root.valueSum = expand(root);
    const children = root.children as Map<number, Node>;
    if (root.terminal !== null || children.size === 1) {
      reason = 'forced';
    } else {
      const deadline = started + timeLimit;
      while (simulations < maxSimulations && now() < deadline) {
        simulate(root);
        simulations += 1;
      }
    }
  }

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
    elapsedMs: now() - started,
    reason,
  };
}
