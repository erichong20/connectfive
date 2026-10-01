"""Threat-aware alpha-beta search on the incremental pattern board.

Compared with ``connectfive.search.NegamaxAgent`` this search:

- evaluates positions with cached line levels instead of rescanning the board;
- stores transposition entries with exact/lower/upper bound flags;
- prunes by threat: must-block moves and open-three defences are generated
  before quiet moves, and forced single replies do not consume depth;
- proves wins with a VCF (victory by continuous fours) search at the root and
  at quiet leaves;
- searches root moves against a lowered window, so every move within
  ``score_margin`` of the best has an exact score. Those scores give soft
  policy targets and fair random tie-breaking.

The older agents are kept unchanged as historical baselines.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from connectfive.patterns import (
    ACTION_TO_INDEX,
    BLACK,
    INDEX_TO_ACTION,
    WHITE,
    PatternBoard,
)

MATE = 10_000_000
WIN_THRESHOLD = MATE - 10_000
VCF_WIN = MATE - 5_000
INFINITY = MATE + 1
EXACT, LOWER, UPPER = 0, 1, 2

# Move-generation outcomes.
QUIET, WIN, LOSS, FORCE_WIN, BLOCK, DEFEND = range(6)


class BudgetExhausted(Exception):
    """Raised inside search when the node or time budget runs out."""


@dataclass(frozen=True)
class PatternSearchStats:
    nodes: int
    vcf_nodes: int
    completed_depth: int
    budget_exhausted: bool
    elapsed_ms: float
    transposition_hits: int
    cutoffs: int
    reason: str


@dataclass(frozen=True)
class PatternSearchResult:
    """Chosen move, its score for the mover, and exact near-best root scores."""

    actions: tuple[int, ...]
    score: int
    root_scores: tuple[tuple[int, int], ...]
    stats: PatternSearchStats

    def policy_target(self, temperature: float = 800.0) -> dict[int, float]:
        """Softmax over the exact near-best root scores (a soft teacher target)."""

        if not self.root_scores:
            return {action: 1.0 / len(self.actions) for action in self.actions}
        best = max(score for _, score in self.root_scores)
        if best >= WIN_THRESHOLD or best <= -WIN_THRESHOLD:
            # Mates are not comparable on the heuristic scale: keep only the best.
            chosen = [action for action, score in self.root_scores if score == best]
            return {action: 1.0 / len(chosen) for action in chosen}
        weights = {
            action: math.exp((score - best) / temperature)
            for action, score in self.root_scores
        }
        total = sum(weights.values())
        return {action: weight / total for action, weight in weights.items()}


def score_to_value(score: int, scale: float = 6_000.0) -> float:
    """Map a search score to the [-1, 1] range used by the value head."""

    if score >= WIN_THRESHOLD:
        return 1.0
    if score <= -WIN_THRESHOLD:
        return -1.0
    return math.tanh(score / scale)


class PatternSearch:
    """One search over one position, with budgets and inspectable counters."""

    def __init__(
        self,
        board: PatternBoard,
        max_depth: int = 6,
        width: int = 12,
        node_budget: int = 4_000,
        time_limit: float | None = None,
        vcf_depth: int = 8,
        leaf_vcf_depth: int = 4,
        score_margin: int = 2_000,
        max_ply: int = 40,
    ):
        self.board = board
        self.max_depth = max_depth
        self.width = width
        self.node_budget = node_budget
        self.time_limit = time_limit
        self.vcf_depth = vcf_depth
        self.leaf_vcf_depth = leaf_vcf_depth
        self.score_margin = score_margin
        self.max_ply = max_ply
        self.nodes = 0
        self.vcf_nodes = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.table: dict[int, tuple[int, int, int, int]] = {}
        self.vcf_failures: dict[int, int] = {}
        self.deadline = None

    # ----- move generation -------------------------------------------------

    def generate(self, width: int) -> tuple[int, list[int]]:
        """Classify the position for the side to move and list sensible moves."""

        board = self.board
        me = board.turn
        opponent = WHITE if me == BLACK else BLACK
        mine = board.summary[me]
        theirs = board.summary[opponent]
        candidates = board.candidates

        wins = [i for i in candidates if mine[i][3]]
        if wins:
            return WIN, [min(wins)]
        blocks = [i for i in candidates if theirs[i][3]]
        if len(blocks) >= 2:
            return LOSS, sorted(blocks)
        if blocks:
            return BLOCK, blocks
        forcing = [i for i in candidates if mine[i][4]]
        if forcing:
            return FORCE_WIN, [max(forcing, key=lambda i: (mine[i][0], -i))]

        threats = [i for i in candidates if theirs[i][4]]
        if threats:
            # The opponent can make an open or double four next move. Answer by
            # occupying one of their four-making cells or by counter-attacking
            # with our own four.
            defence = {i for i in candidates if theirs[i][5] or mine[i][5]}
            ordered = sorted(
                defence, key=lambda i: (-(mine[i][0] + theirs[i][0]), i)
            )
            return DEFEND, ordered[: max(width, 2 * len(threats) + 4)]

        ordered = sorted(
            candidates, key=lambda i: (-(mine[i][0] + theirs[i][0]), i)
        )
        return QUIET, ordered[:width]

    # ----- evaluation --------------------------------------------------------

    def evaluate(self) -> int:
        board = self.board
        me = board.turn
        opponent = WHITE if me == BLACK else BLACK
        mine = board.summary[me]
        theirs = board.summary[opponent]
        return sum(mine[i][1] - theirs[i][2] for i in board.candidates)

    # ----- victory by continuous fours ---------------------------------------

    def vcf(self, depth: int) -> int | None:
        """Return a first move proving a win by fours for the side to move."""

        board = self.board
        me = board.turn
        opponent = WHITE if me == BLACK else BLACK
        mine = board.summary[me]
        theirs = board.summary[opponent]
        wins = [i for i in board.candidates if mine[i][3]]
        if wins:
            return min(wins)
        if depth <= 0:
            return None
        failed = self.vcf_failures.get(board.hash)
        if failed is not None and failed >= depth:
            return None
        blocks = [i for i in board.candidates if theirs[i][3]]
        if len(blocks) >= 2:
            return None
        fours = [i for i in board.candidates if mine[i][5]]
        if blocks:
            fours = [i for i in fours if i == blocks[0]]
        fours.sort(key=lambda i: (-mine[i][0], i))
        for move in fours:
            self._tick(vcf=True)
            board.play(move)
            try:
                replies = board.winning_cells(me)
                proved = len(replies) >= 2
                if len(replies) == 1:
                    board.play(replies[0])
                    try:
                        proved = self.vcf(depth - 1) is not None
                    finally:
                        board.undo()
            finally:
                board.undo()
            if proved:
                return move
        self.vcf_failures[board.hash] = depth
        return None

    # ----- alpha-beta --------------------------------------------------------

    def _tick(self, vcf: bool = False) -> None:
        if vcf:
            self.vcf_nodes += 1
        else:
            self.nodes += 1
        if self.nodes + self.vcf_nodes > self.node_budget:
            raise BudgetExhausted
        if (
            self.deadline is not None
            and (self.nodes & 63) == 0
            and time.perf_counter() > self.deadline
        ):
            raise BudgetExhausted

    def negamax(self, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        board = self.board
        if board.is_full():
            return 0
        kind, moves = self.generate(self.width)
        if kind == WIN:
            return MATE - ply - 1
        if kind == LOSS:
            return -(MATE - ply - 2)
        if kind == FORCE_WIN:
            return MATE - ply - 3
        if not moves:
            return self.evaluate()

        forced = kind == BLOCK
        if (depth <= 0 and not forced) or ply >= self.max_ply:
            if kind == QUIET and self.leaf_vcf_depth and self.vcf(self.leaf_vcf_depth) is not None:
                return VCF_WIN - ply
            return self.evaluate()

        original_alpha = alpha
        entry = self.table.get(board.hash)
        best_move = None
        if entry is not None:
            entry_depth, flag, value, best_move = entry
            if entry_depth >= depth:
                if flag == EXACT:
                    self.tt_hits += 1
                    return value
                if flag == LOWER and value >= beta:
                    self.tt_hits += 1
                    return value
                if flag == UPPER and value <= alpha:
                    self.tt_hits += 1
                    return value
            if best_move in moves:
                moves = [best_move] + [move for move in moves if move != best_move]

        child_depth = depth if forced else depth - 1
        best = -INFINITY
        best_move = moves[0]
        for move in moves:
            board.play(move)
            try:
                score = -self.negamax(child_depth, -beta, -alpha, ply + 1)
            finally:
                board.undo()
            if score > best:
                best = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                self.cutoffs += 1
                break

        if best <= original_alpha:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.table[board.hash] = (depth, flag, best, best_move)
        return best

    # ----- root --------------------------------------------------------------

    def run(self) -> PatternSearchResult:
        started = time.perf_counter()
        if self.time_limit is not None:
            self.deadline = started + self.time_limit
        board = self.board
        if not board.candidates:
            # Empty board: the centre is the conventional and strongest start.
            center = ACTION_TO_INDEX[(15 // 2) * 15 + 15 // 2]
            return self._result((center,), 0, ((center, 0),), 0, False, started, "opening")

        kind, moves = self.generate(self.width)
        if kind in (WIN, FORCE_WIN):
            score = MATE - 1 if kind == WIN else MATE - 3
            return self._result((moves[0],), score, ((moves[0], score),), 0, False, started,
                                "win" if kind == WIN else "open-four")
        if kind == LOSS:
            score = -(MATE - 2)
            return self._result(tuple(moves), score, tuple((m, score) for m in moves), 0,
                                False, started, "lost")
        if kind == BLOCK and len(moves) == 1:
            pass  # still search, so the value estimate is meaningful

        try:
            move = self.vcf(self.vcf_depth) if kind != BLOCK else None
        except BudgetExhausted:
            move = None
        if move is not None:
            score = VCF_WIN
            return self._result((move,), score, ((move, score),), 0, False, started, "vcf")

        best_moves = (moves[0],)
        best_score = -INFINITY
        root_scores: tuple[tuple[int, int], ...] = ()
        completed = 0
        exhausted = False
        previous: dict[int, int] = {}
        for depth in range(1, self.max_depth + 1):
            ordered = sorted(moves, key=lambda m: -previous.get(m, -INFINITY))
            try:
                iteration = self._search_root(ordered, depth)
            except BudgetExhausted:
                exhausted = True
                break
            best_score, best_moves, root_scores, previous = iteration
            completed = depth
            if abs(best_score) >= WIN_THRESHOLD:
                break
        return self._result(best_moves, best_score, root_scores, completed, exhausted, started,
                            "search")

    def _search_root(self, moves: list[int], depth: int):
        board = self.board
        best = -INFINITY
        scores: dict[int, int] = {}
        exact: dict[int, int] = {}
        for move in moves:
            floor = best - self.score_margin if best > -INFINITY else -INFINITY
            if best >= WIN_THRESHOLD:
                floor = best - 1
            self.nodes += 1
            board.play(move)
            try:
                score = -self.negamax(depth - 1, -INFINITY, -floor, 1)
            finally:
                board.undo()
            scores[move] = score
            if score > floor:
                exact[move] = score
            best = max(best, score)
        best_moves = tuple(sorted(m for m, s in exact.items() if s == best))
        near = tuple(
            (m, s) for m, s in sorted(exact.items(), key=lambda item: (-item[1], item[0]))
            if s >= best - self.score_margin
        )
        return best, best_moves, near, scores

    def _result(self, actions, score, root_scores, depth, exhausted, started, reason):
        stats = PatternSearchStats(
            nodes=self.nodes,
            vcf_nodes=self.vcf_nodes,
            completed_depth=depth,
            budget_exhausted=exhausted,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
            transposition_hits=self.tt_hits,
            cutoffs=self.cutoffs,
            reason=reason,
        )
        return PatternSearchResult(
            actions=tuple(INDEX_TO_ACTION[i] for i in actions),
            score=int(score),
            root_scores=tuple((INDEX_TO_ACTION[i], int(s)) for i, s in root_scores),
            stats=stats,
        )


def search_board(board: np.ndarray, player: int, **options) -> PatternSearchResult:
    """Search an environment board (-1 empty, 0 black, 1 white)."""

    return PatternSearch(PatternBoard.from_array(board, player), **options).run()


@dataclass
class PatternSearchAgent:
    """Classical teacher: pattern evaluation, threat pruning, VCF, alpha-beta."""

    name: str = "pattern"
    max_depth: int = 6
    width: int = 12
    node_budget: int = 4_000
    vcf_depth: int = 8
    last_search: PatternSearchResult | None = field(default=None, repr=False)

    def select_action(self, state, key) -> int:
        import jax

        board = np.asarray(state._board)
        result = search_board(
            board,
            int(state.current_player),
            max_depth=self.max_depth,
            width=self.width,
            node_budget=self.node_budget,
            vcf_depth=self.vcf_depth,
        )
        choices = result.actions
        index = int(jax.random.randint(key, (), 0, len(choices)))
        selected = int(choices[index])
        legal = np.asarray(state.legal_action_mask)
        if not legal[selected]:
            raise RuntimeError("pattern search produced an illegal move")
        self.last_search = result
        return selected
