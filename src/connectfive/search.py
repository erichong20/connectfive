"""CPU-budgeted negamax search with alpha-beta pruning."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import jax
import numpy as np

from connectfive.agents import (
    _board_after,
    _choose,
    _is_exact_five_after,
    _nearby_legal_actions,
    _ordered_candidates,
    _pattern_value,
    _run_and_open_ends,
)
from connectfive.env import State

MATE_SCORE = 10_000_000
INFINITY = MATE_SCORE + 1


@dataclass(frozen=True)
class SearchStats:
    """Observable work performed by one search decision."""

    nodes: int
    leaves: int
    cutoffs: int
    transposition_hits: int
    completed_depth: int
    budget_exhausted: bool
    elapsed_ms: float
    threat_extensions: int = 0


@dataclass(frozen=True)
class SearchResult:
    """Best moves and diagnostics returned by negamax."""

    actions: tuple[int, ...]
    score: int
    stats: SearchStats
    root_scores: tuple[tuple[int, int], ...] = ()


@dataclass
class NegamaxAgent:
    """Iterative-deepening negamax under a fixed node budget."""

    name: str = "negamax"
    max_depth: int = 3
    candidate_width: int = 10
    node_budget: int = 1_500
    last_search: SearchResult | None = None

    def select_action(self, state: State, key: jax.Array) -> int:
        board = np.asarray(state._board)
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        if not len(legal):
            raise ValueError("the negamax agent was asked to move with no legal actions")
        search = NegamaxSearch(
            max_depth=self.max_depth,
            candidate_width=self.candidate_width,
            node_budget=self.node_budget,
        )
        result = search.run(board, int(state.current_player), legal)
        selected = _choose(key, result.actions)
        self.last_search = replace(result, actions=(selected,))
        return selected


@dataclass
class ThreatSearchAgent:
    """Negamax with forcing-line extensions and fork-aware leaf evaluation."""

    name: str = "threatsearch"
    max_depth: int = 3
    candidate_width: int = 10
    node_budget: int = 250
    extension_depth: int = 2
    forcing_width: int = 8
    last_search: SearchResult | None = None

    def select_action(self, state: State, key: jax.Array) -> int:
        board = np.asarray(state._board)
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        if not len(legal):
            raise ValueError("the threat-search agent was asked to move with no legal actions")
        search = ThreatNegamaxSearch(
            max_depth=self.max_depth,
            candidate_width=self.candidate_width,
            node_budget=self.node_budget,
            extension_depth=self.extension_depth,
            forcing_width=self.forcing_width,
        )
        result = search.run(board, int(state.current_player), legal)
        selected = _choose(key, result.actions)
        self.last_search = replace(result, actions=(selected,))
        return selected


class NegamaxSearch:
    """Search one position while tracking a strict, inspectable work budget."""

    def __init__(self, max_depth: int, candidate_width: int, node_budget: int):
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if candidate_width < 1:
            raise ValueError("candidate_width must be positive")
        if node_budget < 1:
            raise ValueError("node_budget must be positive")
        self.max_depth = max_depth
        self.candidate_width = candidate_width
        self.node_budget = node_budget
        self.nodes = 0
        self.leaves = 0
        self.cutoffs = 0
        self.transposition_hits = 0
        self.budget_exhausted = False
        self.threat_extensions = 0
        self.current_iteration_depth = 0
        self.table: dict[tuple[bytes, int, int], int] = {}

    def run(self, board: np.ndarray, player: int, legal: np.ndarray) -> SearchResult:
        started = time.perf_counter()
        opponent = 1 - player
        root_moves = _ordered_candidates(
            board, legal, player, opponent, self.candidate_width
        )
        best_actions = (int(root_moves[0]),)
        best_score = -INFINITY
        root_scores: tuple[tuple[int, int], ...] = ()
        completed_depth = 0

        for depth in range(1, self.max_depth + 1):
            self.current_iteration_depth = depth
            iteration = self._search_root(board, player, legal, root_moves, depth)
            if iteration is None:
                break
            best_actions, best_score, root_scores = iteration
            completed_depth = depth
            if best_score >= MATE_SCORE - depth:
                break

        stats = SearchStats(
            nodes=self.nodes,
            leaves=self.leaves,
            cutoffs=self.cutoffs,
            transposition_hits=self.transposition_hits,
            completed_depth=completed_depth,
            budget_exhausted=self.budget_exhausted,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
            threat_extensions=self.threat_extensions,
        )
        return SearchResult(best_actions, best_score, stats, root_scores)

    def _search_root(
        self,
        board: np.ndarray,
        player: int,
        legal: np.ndarray,
        moves: np.ndarray,
        depth: int,
    ) -> tuple[tuple[int, ...], int, tuple[tuple[int, int], ...]] | None:
        best_score = -INFINITY
        best_actions: list[int] = []
        root_scores = []
        alpha = -INFINITY

        for move in moves:
            if self.nodes >= self.node_budget:
                self.budget_exhausted = True
                return None
            action = int(move)
            self.nodes += 1
            if _is_exact_five_after(board, action, player):
                score = MATE_SCORE
            else:
                child = _board_after(board, action, player)
                child_legal = legal[legal != action]
                score = -self._negamax(
                    child, 1 - player, child_legal, depth - 1, -INFINITY, -alpha, 1
                )
                if self.budget_exhausted:
                    return None
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
            root_scores.append((action, score))
            alpha = max(alpha, score)
        return tuple(best_actions), best_score, tuple(root_scores)

    def _negamax(
        self,
        board: np.ndarray,
        player: int,
        legal: np.ndarray,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
    ) -> int:
        if self.nodes >= self.node_budget:
            self.budget_exhausted = True
            return 0
        self.nodes += 1

        if depth == 0 or not len(legal):
            self.leaves += 1
            return self._leaf_value(board, player, legal, alpha, beta, ply)

        key = (board.tobytes(), player, depth)
        cached = self.table.get(key)
        if cached is not None:
            self.transposition_hits += 1
            return cached

        opponent = 1 - player
        moves = _ordered_candidates(
            board, legal, player, opponent, self.candidate_width
        )
        best = -INFINITY
        cutoff = False
        for move in moves:
            action = int(move)
            if _is_exact_five_after(board, action, player):
                score = MATE_SCORE - ply
            else:
                child = _board_after(board, action, player)
                child_legal = legal[legal != action]
                score = -self._negamax(
                    child, opponent, child_legal, depth - 1, -beta, -alpha, ply + 1
                )
                if self.budget_exhausted:
                    return 0
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                self.cutoffs += 1
                cutoff = True
                break
        if not cutoff:
            self.table[key] = best
        return best

    def _leaf_value(
        self,
        board: np.ndarray,
        player: int,
        legal: np.ndarray,
        alpha: int,
        beta: int,
        ply: int,
    ) -> int:
        del alpha, beta, ply
        return self._evaluate(board, player, legal)

    def _evaluate(self, board: np.ndarray, player: int, legal: np.ndarray) -> int:
        if not len(legal):
            return 0
        opponent = 1 - player
        candidates = _nearby_legal_actions(board, legal)
        own_wins = sum(_is_exact_five_after(board, move, player) for move in candidates)
        if own_wins:
            return MATE_SCORE // 2 + 50_000 * (own_wins - 1)
        opponent_wins = sum(
            _is_exact_five_after(board, move, opponent) for move in candidates
        )
        if opponent_wins >= 2:
            return -MATE_SCORE // 2
        own_best = max(_attack_score(board, move, player) for move in candidates)
        opponent_best = max(_attack_score(board, move, opponent) for move in candidates)
        forced_block_penalty = 100_000 if opponent_wins == 1 else 0
        return own_best - opponent_best - forced_block_penalty


class ThreatNegamaxSearch(NegamaxSearch):
    """Extend nominal leaves through wins, blocks, and fork sequences."""

    def __init__(
        self,
        max_depth: int,
        candidate_width: int,
        node_budget: int,
        extension_depth: int,
        forcing_width: int,
    ):
        super().__init__(max_depth, candidate_width, node_budget)
        if extension_depth < 0:
            raise ValueError("extension_depth must not be negative")
        if forcing_width < 1:
            raise ValueError("forcing_width must be positive")
        self.extension_depth = extension_depth
        self.forcing_width = forcing_width
        self._fork_cache: dict[tuple[bytes, int], tuple[int, ...]] = {}

    def _leaf_value(
        self,
        board: np.ndarray,
        player: int,
        legal: np.ndarray,
        alpha: int,
        beta: int,
        ply: int,
    ) -> int:
        remaining = self.extension_depth if self.current_iteration_depth >= 2 else 0
        return self._quiescence(
            board, player, legal, alpha, beta, ply, remaining
        )

    def _quiescence(
        self,
        board: np.ndarray,
        player: int,
        legal: np.ndarray,
        alpha: int,
        beta: int,
        ply: int,
        remaining: int,
    ) -> int:
        stand_pat = self._evaluate(board, player, legal)
        if remaining == 0 or not len(legal):
            return stand_pat
        moves, must_respond = self._forcing_moves(board, player, legal)
        if not len(moves):
            return stand_pat

        best = -INFINITY if must_respond else stand_pat
        if not must_respond:
            alpha = max(alpha, stand_pat)
            if alpha >= beta:
                return stand_pat
        for move in moves:
            if self.nodes >= self.node_budget:
                self.budget_exhausted = True
                return 0
            action = int(move)
            self.nodes += 1
            self.threat_extensions += 1
            if _is_exact_five_after(board, action, player):
                score = MATE_SCORE - ply
            else:
                child = _board_after(board, action, player)
                child_legal = legal[legal != action]
                score = -self._quiescence(
                    child,
                    1 - player,
                    child_legal,
                    -beta,
                    -alpha,
                    ply + 1,
                    remaining - 1,
                )
                if self.budget_exhausted:
                    return 0
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                self.cutoffs += 1
                break
        return best

    def _forcing_moves(
        self, board: np.ndarray, player: int, legal: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        opponent = 1 - player
        candidates = _nearby_legal_actions(board, legal)
        wins = [
            int(move) for move in candidates
            if _is_exact_five_after(board, move, player)
        ]
        if wins:
            return np.asarray(wins[: self.forcing_width], dtype=np.int32), True
        blocks = [
            int(move) for move in candidates
            if _is_exact_five_after(board, move, opponent)
        ]
        if blocks:
            return np.asarray(blocks[: self.forcing_width], dtype=np.int32), True

        own_forks = list(self._fork_moves(board, player, legal))
        if own_forks:
            return np.asarray(own_forks[: self.forcing_width], dtype=np.int32), False
        opponent_forks = list(self._fork_moves(board, opponent, legal))
        return np.asarray(opponent_forks[: self.forcing_width], dtype=np.int32), False

    def _fork_moves(
        self, board: np.ndarray, player: int, legal: np.ndarray
    ) -> tuple[int, ...]:
        key = (board.tobytes(), player)
        cached = self._fork_cache.get(key)
        if cached is not None:
            return cached
        opponent = 1 - player
        pool = _ordered_candidates(
            board,
            legal,
            player,
            opponent,
            self.forcing_width,
        )
        forks = []
        for move in pool:
            action = int(move)
            if _is_exact_five_after(board, action, player):
                continue
            child = _board_after(board, action, player)
            child_legal = legal[legal != action]
            winning_replies = sum(
                _is_exact_five_after(child, reply, player)
                for reply in _line_replies(child, child_legal, action)
            )
            if winning_replies >= 2:
                forks.append(action)
        result = tuple(forks)
        self._fork_cache[key] = result
        return result

    def _evaluate(self, board: np.ndarray, player: int, legal: np.ndarray) -> int:
        if not len(legal):
            return 0
        opponent = 1 - player
        candidates = _nearby_legal_actions(board, legal)
        own_wins = sum(_is_exact_five_after(board, move, player) for move in candidates)
        if own_wins:
            return MATE_SCORE // 2 + 50_000 * (own_wins - 1)
        opponent_wins = sum(
            _is_exact_five_after(board, move, opponent) for move in candidates
        )
        if opponent_wins >= 2:
            return -MATE_SCORE // 2
        evaluation_width = self.forcing_width
        own_candidates = _ordered_candidates(
            board, legal, player, opponent, evaluation_width
        )
        opponent_candidates = _ordered_candidates(
            board, legal, opponent, player, evaluation_width
        )
        own_best = max(
            _threat_attack_score(board, move, player) for move in own_candidates
        )
        opponent_best = max(
            _threat_attack_score(board, move, opponent) for move in opponent_candidates
        )
        forced_block_penalty = 100_000 if opponent_wins == 1 else 0
        return own_best - opponent_best - forced_block_penalty


def _attack_score(board: np.ndarray, action: int, player: int) -> int:
    """Score only one player's potential so evaluation stays antisymmetric."""

    row, col = divmod(int(action), board.shape[0])
    score = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        run, open_ends = _run_and_open_ends(board, row, col, player, dr, dc)
        score += _pattern_value(run, open_ends)
    for rr in range(max(0, row - 2), min(board.shape[0], row + 3)):
        for cc in range(max(0, col - 2), min(board.shape[1], col + 3)):
            if board[rr, cc] == player:
                distance = max(abs(rr - row), abs(cc - col))
                score += 8 if distance == 1 else 2
    center = board.shape[0] // 2
    return score - abs(row - center) - abs(col - center)


def _threat_attack_score(board: np.ndarray, action: int, player: int) -> int:
    """Add broken five-cell windows to the contiguous attack score."""

    score = _attack_score(board, action, player)
    row, col = divmod(int(action), board.shape[0])
    virtual = _board_after(board, action, player)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        direction_best = 0
        for offset in range(-4, 1):
            cells = [
                (row + (offset + step) * dr, col + (offset + step) * dc)
                for step in range(5)
            ]
            if not all(
                0 <= rr < board.shape[0] and 0 <= cc < board.shape[1]
                for rr, cc in cells
            ):
                continue
            values = [virtual[rr, cc] for rr, cc in cells]
            stones = sum(value == player for value in values)
            empties = sum(value == -1 for value in values)
            if stones == 4 and empties == 1:
                direction_best = max(direction_best, 6_000)
            elif stones == 3 and empties == 2:
                direction_best = max(direction_best, 500)
            elif stones == 2 and empties == 3:
                direction_best = max(direction_best, 40)
        score += direction_best
    return score


def _line_replies(
    board: np.ndarray, legal: np.ndarray, action: int
) -> tuple[int, ...]:
    """Legal cells whose line could have changed because of ``action``."""

    legal_set = set(map(int, legal))
    row, col = divmod(int(action), board.shape[0])
    replies = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for distance in range(-4, 5):
            if distance == 0:
                continue
            rr = row + distance * dr
            cc = col + distance * dc
            if 0 <= rr < board.shape[0] and 0 <= cc < board.shape[1]:
                reply = rr * board.shape[0] + cc
                if reply in legal_set:
                    replies.append(reply)
    return tuple(dict.fromkeys(replies))
