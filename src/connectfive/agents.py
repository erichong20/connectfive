"""Reusable agents for playing and evaluating Connect Five."""

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from connectfive.env import BOARD_SIZE, EMPTY, State


class Agent(Protocol):
    """Small common interface shared by human-facing and evaluation code."""

    name: str

    def select_action(self, state: State, key: jax.Array) -> int:
        """Return one action for ``state`` using ``key`` for tie-breaking."""


@dataclass(frozen=True)
class RandomAgent:
    """Uniformly sample one legal action."""

    name: str = "random"

    def select_action(self, state: State, key: jax.Array) -> int:
        probabilities = state.legal_action_mask.astype(jnp.float32)
        probabilities /= probabilities.sum()
        return int(jax.random.choice(key, probabilities.size, p=probabilities))


@dataclass(frozen=True)
class TacticalAgent:
    """One-ply Gomoku bot with explicit, inspectable tactical priorities.

    The bot wins or blocks an exact five first, then scores contiguous threats,
    nearby stones, and centrality. It is deliberately not a search bot: keeping
    this milestone to one ply makes its strengths and blind spots easy to see.
    """

    name: str = "tactical"

    def select_action(self, state: State, key: jax.Array) -> int:
        board = np.asarray(state._board)
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        if not len(legal):
            raise ValueError("the tactical agent was asked to move with no legal actions")

        player = int(state.current_player)
        opponent = 1 - player
        candidates = _nearby_legal_actions(board, legal)

        wins = [action for action in candidates if _is_exact_five_after(board, action, player)]
        if wins:
            return _choose(key, wins)

        forced_blocks = [
            action for action in candidates if _is_exact_five_after(board, action, opponent)
        ]
        if forced_blocks:
            return _choose(key, forced_blocks)

        scores = np.array(
            [_tactical_score(board, action, player, opponent) for action in candidates]
        )
        best = candidates[scores == scores.max()]
        return _choose(key, best)


@dataclass(frozen=True)
class LookaheadAgent:
    """Budgeted two-ply search layered on top of the tactical evaluator.

    The tactical bot asks only "how good is my move?". This bot asks "after
    this move, what is the strongest reply my opponent can make?" and chooses
    the move with the best worst-case result. Candidate caps keep the policy
    responsive in the browser and inexpensive in CPU evaluations.
    """

    name: str = "lookahead"
    root_width: int = 16
    reply_width: int = 12

    def select_action(self, state: State, key: jax.Array) -> int:
        board = np.asarray(state._board)
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        if not len(legal):
            raise ValueError("the lookahead agent was asked to move with no legal actions")

        player = int(state.current_player)
        opponent = 1 - player
        wins = [action for action in _nearby_legal_actions(board, legal)
                if _is_exact_five_after(board, action, player)]
        if wins:
            return _choose(key, wins)

        candidates = _ordered_candidates(
            board, legal, player, opponent, self.root_width
        )
        baseline_replies = _ordered_candidates(
            board, legal, opponent, player, self.reply_width
        )
        values = []
        for action in candidates:
            after_move = _board_after(board, action, player)
            replies_legal = legal[legal != action]
            replies = _reply_candidates(
                after_move,
                replies_legal,
                opponent,
                player,
                action,
                baseline_replies,
                self.reply_width,
            )
            immediate_value = _tactical_score(board, action, player, opponent)
            worst_reply = min(
                _reply_value(after_move, reply, player, opponent) for reply in replies
            ) if len(replies) else 0
            values.append(immediate_value + worst_reply)

        values_array = np.asarray(values)
        best = candidates[values_array == values_array.max()]
        return _choose(key, best)


def make_agent(name: str) -> Agent:
    """Construct a built-in agent by its command-line name."""

    if name == "random":
        return RandomAgent()
    if name == "tactical":
        return TacticalAgent()
    if name == "lookahead":
        return LookaheadAgent()
    if name == "negamax":
        from connectfive.search import NegamaxAgent

        return NegamaxAgent()
    if name == "threatsearch":
        from connectfive.search import ThreatSearchAgent

        return ThreatSearchAgent()
    if name == "pattern":
        from connectfive.pattern_search import PatternSearchAgent

        return PatternSearchAgent()
    raise ValueError(
        f"unknown agent {name!r}; choose random, tactical, lookahead, negamax, "
        "threatsearch, or pattern"
    )


def _choose(key: jax.Array, actions) -> int:
    choices = np.asarray(actions, dtype=np.int32)
    index = int(jax.random.randint(key, (), 0, len(choices)))
    return int(choices[index])


def _nearby_legal_actions(board: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Return legal moves within two intersections of a stone.

    Exact-five tactics can only occur next to an existing stone. A two-cell
    neighborhood also gives the positional scorer room to extend patterns while
    avoiding a pointless scan of the whole empty board.
    """

    occupied = np.argwhere(board != EMPTY)
    if not len(occupied):
        center = (BOARD_SIZE // 2) * BOARD_SIZE + BOARD_SIZE // 2
        return np.asarray([center], dtype=np.int32)

    nearby: list[int] = []
    for action in legal:
        row, col = divmod(int(action), BOARD_SIZE)
        if np.any(np.max(np.abs(occupied - np.array([row, col])), axis=1) <= 2):
            nearby.append(int(action))
    return np.asarray(nearby if nearby else legal, dtype=np.int32)


def _ordered_candidates(
    board: np.ndarray,
    legal: np.ndarray,
    player: int,
    opponent: int,
    width: int,
) -> np.ndarray:
    """Order forcing moves first, then retain the best heuristic candidates."""

    nearby = _nearby_legal_actions(board, legal)
    wins = [int(move) for move in nearby if _is_exact_five_after(board, move, player)]
    blocks = [
        int(move) for move in nearby
        if move not in wins and _is_exact_five_after(board, move, opponent)
    ]
    remainder = [int(move) for move in nearby if move not in wins and move not in blocks]
    remainder.sort(
        key=lambda move: (-_tactical_score(board, move, player, opponent), move)
    )
    ordered = wins + blocks + remainder
    return np.asarray(ordered[:width], dtype=np.int32)


def _board_after(board: np.ndarray, action: int, player: int) -> np.ndarray:
    result = board.copy()
    result.reshape(-1)[int(action)] = player
    return result


def _reply_candidates(
    board: np.ndarray,
    legal: np.ndarray,
    player: int,
    opponent: int,
    focus: int,
    baseline: np.ndarray,
    width: int,
) -> np.ndarray:
    """Reorder a small reply pool plus every forcing move after ``focus``."""

    nearby = _nearby_legal_actions(board, legal)
    wins = [int(move) for move in nearby if _is_exact_five_after(board, move, player)]
    blocks = [
        int(move) for move in nearby
        if move not in wins and _is_exact_five_after(board, move, opponent)
    ]
    focus_row, focus_col = divmod(int(focus), BOARD_SIZE)
    local = [
        int(move) for move in nearby
        if max(
            abs(int(move) // BOARD_SIZE - focus_row),
            abs(int(move) % BOARD_SIZE - focus_col),
        ) <= 2
    ]
    legal_set = set(map(int, legal))
    pool = list(dict.fromkeys(
        wins + blocks + [int(move) for move in baseline if int(move) in legal_set] + local
    ))
    forcing = set(wins + blocks)
    remainder = [move for move in pool if move not in forcing]
    remainder.sort(
        key=lambda move: (-_tactical_score(board, move, player, opponent), move)
    )
    return np.asarray((wins + blocks + remainder)[:width], dtype=np.int32)


def _reply_value(board: np.ndarray, reply: int, player: int, opponent: int) -> int:
    """Evaluate a leaf after the opponent chooses one concrete reply."""

    if _is_exact_five_after(board, reply, opponent):
        return -1_000_000
    reply_threat = _tactical_score(board, reply, opponent, player)
    leaf = _board_after(board, reply, opponent)
    legal = np.flatnonzero(leaf.reshape(-1) == EMPTY)
    candidates = _nearby_legal_actions(leaf, legal)

    own_wins = sum(_is_exact_five_after(leaf, move, player) for move in candidates)
    if own_wins:
        return 500_000 + 50_000 * (own_wins - 1)

    opponent_wins = sum(
        _is_exact_five_after(leaf, move, opponent) for move in candidates
    )
    if opponent_wins >= 2:
        return -500_000

    forced_threat_penalty = 100_000 if opponent_wins == 1 else 0
    return -11 * reply_threat // 10 - forced_threat_penalty


def _is_exact_five_after(board: np.ndarray, action: int, player: int) -> bool:
    row, col = divmod(int(action), BOARD_SIZE)
    if board[row, col] != EMPTY:
        return False
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        run, _ = _run_and_open_ends(board, row, col, player, dr, dc)
        if run == 5:
            return True
    return False


def _tactical_score(board: np.ndarray, action: int, player: int, opponent: int) -> int:
    row, col = divmod(int(action), BOARD_SIZE)
    score = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        own_run, own_open = _run_and_open_ends(board, row, col, player, dr, dc)
        opp_run, opp_open = _run_and_open_ends(board, row, col, opponent, dr, dc)
        score += _pattern_value(own_run, own_open)
        score += 11 * _pattern_value(opp_run, opp_open) // 10

    for rr in range(max(0, row - 2), min(BOARD_SIZE, row + 3)):
        for cc in range(max(0, col - 2), min(BOARD_SIZE, col + 3)):
            if board[rr, cc] != EMPTY:
                distance = max(abs(rr - row), abs(cc - col))
                score += 8 if distance == 1 else 2

    center = BOARD_SIZE // 2
    score -= abs(row - center) + abs(col - center)
    return score


def _pattern_value(run: int, open_ends: int) -> int:
    # Runs above five are intentionally worthless under exact-five rules.
    if run == 5:
        return 100_000
    if run == 4:
        return 12_000 if open_ends == 2 else 3_000 if open_ends == 1 else 0
    if run == 3:
        return 1_200 if open_ends == 2 else 250 if open_ends == 1 else 0
    if run == 2:
        return 80 if open_ends == 2 else 15 if open_ends == 1 else 0
    return 0


def _run_and_open_ends(
    board: np.ndarray, row: int, col: int, player: int, dr: int, dc: int
) -> tuple[int, int]:
    """Measure the contiguous run and its open ends after a hypothetical move."""

    lengths = []
    open_ends = 0
    for sign in (-1, 1):
        length = 0
        distance = 1
        while True:
            rr = row + sign * dr * distance
            cc = col + sign * dc * distance
            if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
                break
            if board[rr, cc] != player:
                if board[rr, cc] == EMPTY:
                    open_ends += 1
                break
            length += 1
            distance += 1
        lengths.append(length)
    return 1 + sum(lengths), open_ends
