"""Searches guided by the learned policy/value network.

Two ways to combine the v2 network with the pattern board:

- ``GuidedAlphaBeta``: the pattern alpha-beta search, but quiet moves near the
  root are ranked by the network policy instead of hand-written scores, and
  quiet leaves can optionally be scored by the value head.
- ``GuidedMCTS``: PUCT tree search (AlphaZero-style selection) using network
  priors and values, with the pattern board's exact tactics as shortcuts:
  wins, forced blocks, open-four wins, and VCF proofs are resolved without
  asking the network.

Both use a wall-clock limit so they can be compared fairly with ``pattern``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from connectfive.network import (
    NetworkConfig,
    PolicyValueNetwork,
    encode_board,
    load_checkpoint,
)
from connectfive.pattern_search import (
    FORCE_WIN,
    LOSS,
    QUIET,
    WIN,
    BudgetExhausted,
    PatternSearch,
    PatternSearchResult,
    PatternSearchStats,
)
from connectfive.patterns import ACTION_TO_INDEX, INDEX_TO_ACTION, PatternBoard


class NetworkEvaluator:
    """Cached single-position policy/value calls on a ``PatternBoard``."""

    def __init__(self, params: Any, config: NetworkConfig):
        self.config = config
        model = PolicyValueNetwork(config)
        self._apply = jax.jit(lambda x: model.apply(params, x))
        self.cache: dict[int, tuple[np.ndarray, float]] = {}
        self.calls = 0

    @classmethod
    def from_checkpoint(cls, path: Path) -> NetworkEvaluator:
        loaded = load_checkpoint(path, jax.random.PRNGKey(0))
        return cls(loaded.params, loaded.config)

    def __call__(self, board: PatternBoard) -> tuple[np.ndarray, float]:
        """Return (logits over 225 actions, value for the side to move)."""

        cached = self.cache.get(board.hash)
        if cached is not None:
            return cached
        self.calls += 1
        array = board.to_array()
        features = encode_board(array, board.player, self.config.input_planes)
        logits, value = self._apply(jnp.asarray(features[None]))
        result = (np.asarray(logits[0], dtype=np.float64), float(value[0]))
        if len(self.cache) > 200_000:
            self.cache.clear()
        self.cache[board.hash] = result
        return result


class GuidedAlphaBeta(PatternSearch):
    """Pattern alpha-beta with network move ordering and optional value leaves."""

    def __init__(self, board, evaluator: NetworkEvaluator, policy_depth: int = 2,
                 value_leaves: bool = False, value_scale: float = 6_000.0, **options):
        super().__init__(board, **options)
        self.evaluator = evaluator
        self.policy_depth = policy_depth
        self.value_leaves = value_leaves
        self.value_scale = value_scale
        self._ply = 0

    def negamax(self, depth, alpha, beta, ply):
        self._ply = ply
        return super().negamax(depth, alpha, beta, ply)

    def generate(self, width):
        kind, moves = super().generate(width if self._ply >= self.policy_depth else 10**6)
        if kind != QUIET or self._ply >= self.policy_depth:
            return kind, moves
        logits, _ = self.evaluator(self.board)
        moves.sort(key=lambda i: -logits[INDEX_TO_ACTION[i]])
        return kind, moves[:width]

    def evaluate(self):
        if not self.value_leaves:
            return super().evaluate()
        _, value = self.evaluator(self.board)
        value = max(-0.999, min(0.999, value))
        return int(self.value_scale * math.atanh(value))

    def run(self):
        self._ply = 0
        return super().run()


class VcfCache:
    """VCF results shared by successive searches (e.g. all moves of a game).

    Entries are exact facts about a position, so sharing them never changes a
    search's answer, only its cost. Cleared when it grows past ``limit``.
    """

    def __init__(self, limit: int = 2_000_000):
        self.failures: dict[int, int] = {}
        self.wins: dict[tuple[int, int], int] = {}
        self.limit = limit

    def attach(self, search: PatternSearch) -> None:
        if len(self.failures) + len(self.wins) > self.limit:
            self.failures.clear()
            self.wins.clear()
        search.vcf_failures = self.failures
        search.vcf_wins = self.wins


@dataclass
class _Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, _Node] | None = None
    terminal: float | None = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class GuidedMCTS:
    """PUCT search; values are always from the perspective of the side to move."""

    def __init__(self, board: PatternBoard, evaluator: NetworkEvaluator,
                 time_limit: float = 0.2, max_simulations: int = 100_000,
                 c_puct: float = 1.5, top_k: int = 16, leaf_vcf_depth: int = 4,
                 root_noise: float = 0.0, noise_alpha: float = 0.3,
                 rng: np.random.Generator | None = None,
                 vcf_cache: VcfCache | None = None):
        self.board = board
        # Self-play exploration: mix Dirichlet noise into the root priors.
        self.root_noise = root_noise
        self.noise_alpha = noise_alpha
        self.rng = rng or np.random.default_rng(0)
        self.evaluator = evaluator
        self.time_limit = time_limit
        self.max_simulations = max_simulations
        self.c_puct = c_puct
        self.top_k = top_k
        self.tactics = PatternSearch(board, node_budget=10**9, leaf_vcf_depth=0)
        if vcf_cache is not None:
            vcf_cache.attach(self.tactics)
        self.leaf_vcf_depth = leaf_vcf_depth
        self.simulations = 0

    def _expand(self, node: _Node) -> float:
        """Create children and return the leaf value for the side to move."""

        board = self.board
        if board.is_full():
            node.terminal = 0.0
            return 0.0
        kind, moves = self.tactics.generate(10**6)
        if kind in (WIN, FORCE_WIN):
            node.terminal = 1.0
            node.children = {moves[0]: _Node(prior=1.0)}
            return 1.0
        if kind == LOSS:
            node.terminal = -1.0
            node.children = {moves[0]: _Node(prior=1.0)}
            return -1.0
        if kind == QUIET and self.leaf_vcf_depth:
            try:
                win = self.tactics.vcf(self.leaf_vcf_depth)
            except BudgetExhausted:
                win = None
            if win is not None:
                node.terminal = 1.0
                node.children = {win: _Node(prior=1.0)}
                return 1.0
        logits, value = self.evaluator(board)
        if kind == QUIET:
            moves = sorted(moves, key=lambda i: -logits[INDEX_TO_ACTION[i]])[: self.top_k]
        scores = np.array([logits[INDEX_TO_ACTION[i]] for i in moves])
        priors = np.exp(scores - scores.max())
        priors /= priors.sum()
        node.children = {move: _Node(prior=float(p)) for move, p in zip(moves, priors)}
        return value

    def _select(self, node: _Node) -> tuple[int, _Node]:
        root_visits = math.sqrt(max(1, node.visits))
        # First-play urgency: unvisited children look slightly worse than the parent.
        fpu = -node.q - 0.2
        best, best_score = None, -math.inf
        for move, child in node.children.items():
            q = -child.q if child.visits else fpu
            score = q + self.c_puct * child.prior * root_visits / (1 + child.visits)
            if score > best_score:
                best, best_score = (move, child), score
        return best

    def _simulate(self, root: _Node) -> None:
        board = self.board
        path = [root]
        node = root
        played = 0
        while node.children is not None and node.terminal is None:
            move, node = self._select(node)
            board.play(move)
            played += 1
            path.append(node)
        if node.terminal is not None:
            value = node.terminal
        else:
            value = self._expand(node)
        for _ in range(played):
            board.undo()
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value

    def run(self) -> PatternSearchResult:
        started = time.perf_counter()
        deadline = started + self.time_limit
        root = _Node(prior=1.0)
        if not self.board.candidates:
            center = ACTION_TO_INDEX[7 * 15 + 7]
            root.children = {center: _Node(prior=1.0, visits=1)}
        else:
            root_value = self._expand(root)
            root.visits, root.value_sum = 1, root_value
            if self.root_noise and root.terminal is None and len(root.children) > 1:
                noise = self.rng.dirichlet([self.noise_alpha] * len(root.children))
                for child, eta in zip(root.children.values(), noise):
                    child.prior = (1 - self.root_noise) * child.prior + self.root_noise * eta
        reason = "mcts"
        if root.terminal is not None or len(root.children) == 1:
            reason = "forced"
        else:
            while self.simulations < self.max_simulations and time.perf_counter() < deadline:
                self._simulate(root)
                self.simulations += 1
        most = max(child.visits for child in root.children.values())
        actions = tuple(sorted(
            INDEX_TO_ACTION[m] for m, c in root.children.items() if c.visits == most
        ))
        root_scores = tuple(
            (INDEX_TO_ACTION[m], c.visits) for m, c in
            sorted(root.children.items(), key=lambda item: -item[1].visits)
        )
        stats = PatternSearchStats(
            nodes=self.simulations, vcf_nodes=self.tactics.vcf_nodes, completed_depth=0,
            budget_exhausted=False, elapsed_ms=(time.perf_counter() - started) * 1_000,
            transposition_hits=0, cutoffs=0, reason=reason,
        )
        score = int(1000 * root.q)
        return PatternSearchResult(actions, score, root_scores, stats)


@dataclass
class GuidedAgent:
    """Agent wrapper: ``mode`` is ``alphabeta``, ``alphabeta-value``, or ``mcts``."""

    evaluator: NetworkEvaluator
    mode: str = "mcts"
    time_limit: float = 0.2
    name: str = "guided"
    # When set, MCTS uses a fixed simulation count instead of the time limit.
    simulations: int | None = None
    vcf_cache: VcfCache = field(default_factory=VcfCache, repr=False)
    last_search: PatternSearchResult | None = field(default=None, repr=False)

    def select_action(self, state, key) -> int:
        board = PatternBoard.from_array(np.asarray(state._board), int(state.current_player))
        if self.mode == "mcts":
            if self.simulations is None:
                search = GuidedMCTS(board, self.evaluator, time_limit=self.time_limit,
                                    vcf_cache=self.vcf_cache)
            else:
                search = GuidedMCTS(board, self.evaluator, time_limit=1e9,
                                    max_simulations=self.simulations,
                                    vcf_cache=self.vcf_cache)
            result = search.run()
        else:
            result = GuidedAlphaBeta(
                board, self.evaluator, value_leaves=self.mode == "alphabeta-value",
                node_budget=10**9, time_limit=self.time_limit, max_depth=12,
            ).run()
        self.last_search = result
        choices = result.actions
        selected = int(choices[int(jax.random.randint(key, (), 0, len(choices)))])
        if not bool(state.legal_action_mask[selected]):
            raise RuntimeError("guided search produced an illegal move")
        return selected
