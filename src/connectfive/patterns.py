"""Incremental exact-five pattern board for fast classical search.

The older search code rescans every candidate move with Python loops at every
leaf. This module instead keeps, for every empty intersection, a small "line
level" in each of the four directions for each player. A stone only changes
levels along its own four lines, within five cells, so a move updates at most
4 x 11 cells instead of re-evaluating the whole board.

Line levels describe what a stone *placed on this empty cell* would create in
one direction, under exact-five rules (an overline is never a five):

    6  exact five                     (an immediate win)
    5  open four                      (two distinct winning cells in the line)
    4  four                           (exactly one winning cell)
    3  open three                     (one more stone can make an open four)
    2  closed three                   (one more stone can make a four)
    1  open two                       (one more stone can make an open three)
    0  nothing useful

The level is computed once for each 10-cell neighbourhood pattern and memoized,
so the search only performs table lookups. This module deliberately avoids JAX:
it is pure Python so it can run in plain worker processes.
"""

from __future__ import annotations

import random
from functools import cache
from operator import itemgetter

import numpy as np

SIZE = 15
PAD = 5
WIDTH = SIZE + 2 * PAD
EMPTY, BLACK, WHITE, WALL = 0, 1, 2, 3
DIRECTIONS = (1, WIDTH, WIDTH + 1, WIDTH - 1)
OFFSETS = tuple(
    tuple(k * d for k in range(-PAD, PAD + 1) if k)
    for d in DIRECTIONS
)
ACTION_TO_INDEX = tuple(
    (a // SIZE + PAD) * WIDTH + a % SIZE + PAD for a in range(SIZE * SIZE)
)
INDEX_TO_ACTION = {index: action for action, index in enumerate(ACTION_TO_INDEX)}
WINDOW_GETTERS = {
    index: tuple(
        itemgetter(*(index + offset for offset in OFFSETS[direction]))
        for direction in range(4)
    )
    for index in ACTION_TO_INDEX
}
NEIGHBOURS = {
    index: tuple(
        index + dr * WIDTH + dc
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if (dr or dc) and (index + dr * WIDTH + dc) in INDEX_TO_ACTION
    )
    for index in ACTION_TO_INDEX
}
LINE_CELLS = {
    index: tuple(
        tuple(index + k * d for k in range(-PAD, PAD + 1)
              if (index + k * d) in INDEX_TO_ACTION)
        for d in DIRECTIONS
    )
    for index in ACTION_TO_INDEX
}

# Search-facing scores. Larger thresholds dominate the additive terms below.
SCORE_FIVE = 10_000_000
SCORE_FORCE = 1_000_000  # open four or double four: wins unless the opponent has a five
SCORE_FOUR_THREE = 100_000
SCORE_DOUBLE_THREE = 50_000
_ORDER_WEIGHTS = (0, 40, 300, 2_500, 2_000, 0, 0)
# Static evaluation: the side to move converts its threats first, the opponent
# only threatens. Levels 5 and 6 for the mover are handled by search as wins.
_ATTACK_WEIGHTS = (0, 30, 200, 1_500, 800, 0, 0)
_DEFENCE_WEIGHTS = (0, 25, 150, 1_000, 600, 5_000, 0)


def _run_through_center(window: tuple[int, ...]) -> int:
    run = 1
    for index in range(PAD - 1, -1, -1):
        if window[index] != 1:
            break
        run += 1
    for index in range(PAD + 1, 2 * PAD + 1):
        if window[index] != 1:
            break
        run += 1
    return run


@cache
def _level(window: tuple[int, ...]) -> int:
    """Level of an 11-cell window whose centre holds the player's stone.

    Cells are 0 empty, 1 own stone, 2 blocked (opponent stone or wall).
    """

    run = _run_through_center(window)
    if run == 5:
        return 6
    if run > 5:
        return 0  # an overline can never become an exact five in this line
    winning_cells = 0
    empties = [i for i in range(1, 2 * PAD) if window[i] == 0]
    for i in empties:
        after = window[:i] + (1,) + window[i + 1:]
        if _run_through_center(after) == 5:
            winning_cells += 1
    if winning_cells >= 2:
        return 5
    if winning_cells == 1:
        return 4
    best = 0
    for i in empties:
        best = max(best, _level(window[:i] + (1,) + window[i + 1:]))
        if best == 5:
            break
    return {5: 3, 4: 2, 3: 1}.get(best, 0)


_TRANSLATE = {
    BLACK: (0, 1, 2, 2),
    WHITE: (0, 2, 1, 2),
}


@cache
def line_levels(raw: tuple[int, ...]) -> tuple[int, int]:
    """Black and white levels for a 10-cell raw neighbourhood (centre omitted)."""

    result = []
    for player in (BLACK, WHITE):
        translate = _TRANSLATE[player]
        mapped = tuple(translate[value] for value in raw)
        result.append(_level(mapped[:PAD] + (1,) + mapped[PAD:]))
    return result[0], result[1]


@cache
def cell_summary(levels: tuple[int, int, int, int]) -> tuple[int, int, int, bool, bool, bool]:
    """Combine four direction levels into (order, attack, defence, win, force, four)."""

    counts = [0] * 7
    for level in levels:
        counts[level] += 1
    win = counts[6] > 0
    fours = counts[5] + counts[4]
    force = not win and (counts[5] > 0 or fours >= 2)
    order = sum(_ORDER_WEIGHTS[level] for level in levels)
    attack = sum(_ATTACK_WEIGHTS[level] for level in levels)
    defence = sum(_DEFENCE_WEIGHTS[level] for level in levels)
    if win:
        order += SCORE_FIVE
    elif force:
        order += SCORE_FORCE
        defence += 8_000
    elif counts[4] and counts[3]:
        order += SCORE_FOUR_THREE
        attack += 20_000
        defence += 10_000
    elif counts[3] >= 2:
        order += SCORE_DOUBLE_THREE
        attack += 20_000
        defence += 5_000
    return order, attack, defence, win, force, win or fours > 0


_ZOBRIST_RNG = random.Random(20260930)
ZOBRIST = {
    player: {index: _ZOBRIST_RNG.getrandbits(64) for index in ACTION_TO_INDEX}
    for player in (BLACK, WHITE)
}
_BLANK = cell_summary((0, 0, 0, 0))


class PatternBoard:
    """Mutable board with make/undo moves and incrementally cached patterns.

    ``summary[player][index]`` is ``cell_summary`` of that empty cell's four
    direction levels: (order, attack, defence, win, force, four).
    """

    def __init__(self) -> None:
        self.cells = [WALL] * (WIDTH * WIDTH)
        for index in ACTION_TO_INDEX:
            self.cells[index] = EMPTY
        self.turn = BLACK
        self.hash = 0
        self.moves: list[int] = []
        self.near = [0] * (WIDTH * WIDTH)
        self.candidates: set[int] = set()
        self.levels = {
            player: dict.fromkeys(ACTION_TO_INDEX, (0, 0, 0, 0))
            for player in (BLACK, WHITE)
        }
        self.summary = {
            player: dict.fromkeys(ACTION_TO_INDEX, _BLANK) for player in (BLACK, WHITE)
        }
        # Per move: the (levels, summary, cell, old level, old summary) entries it
        # changed, so ``undo`` restores them instead of recomputing patterns.
        self._history: list[list[tuple]] = []

    @classmethod
    def from_array(cls, board: np.ndarray, player: int) -> PatternBoard:
        """Build from an environment board (-1 empty, 0 black, 1 white)."""

        result = cls()
        flat = np.asarray(board).reshape(-1)
        for action in np.flatnonzero(flat != -1):
            index = ACTION_TO_INDEX[int(action)]
            stone = BLACK if int(flat[action]) == 0 else WHITE
            result.cells[index] = stone
            result.hash ^= ZOBRIST[stone][index]
            result.moves.append(int(action))
            for neighbour in NEIGHBOURS[index]:
                result.near[neighbour] += 1
        result.turn = BLACK if int(player) == 0 else WHITE
        # Stones loaded from an array have no recorded changes; undo recomputes.
        result._history = [None] * len(result.moves)
        for index in ACTION_TO_INDEX:
            if result.cells[index] == EMPTY:
                if result.near[index]:
                    result.candidates.add(index)
                for direction in range(4):
                    result._refresh(index, direction)
        return result

    def to_array(self) -> np.ndarray:
        """Return the environment representation (-1 empty, 0 black, 1 white)."""

        board = np.full(SIZE * SIZE, -1, dtype=np.int8)
        for action, index in enumerate(ACTION_TO_INDEX):
            if self.cells[index] != EMPTY:
                board[action] = self.cells[index] - 1
        return board.reshape(SIZE, SIZE)

    @property
    def player(self) -> int:
        """Environment player id (0 black, 1 white) of the side to move."""

        return self.turn - 1

    def play(self, index: int) -> None:
        stone = self.turn
        cells = self.cells
        cells[index] = stone
        self.hash ^= ZOBRIST[stone][index]
        self.moves.append(INDEX_TO_ACTION[index])
        candidates = self.candidates
        candidates.discard(index)
        near = self.near
        for neighbour in NEIGHBOURS[index]:
            near[neighbour] += 1
            if cells[neighbour] == EMPTY:
                candidates.add(neighbour)
        self._history.append(self._refresh_lines(index))
        self.turn = WHITE if stone == BLACK else BLACK

    def undo(self) -> None:
        index = ACTION_TO_INDEX[self.moves.pop()]
        stone = self.cells[index]
        self.cells[index] = EMPTY
        self.hash ^= ZOBRIST[stone][index]
        near = self.near
        candidates = self.candidates
        for neighbour in NEIGHBOURS[index]:
            near[neighbour] -= 1
            if not near[neighbour]:
                candidates.discard(neighbour)
        if near[index]:
            candidates.add(index)
        # The undone cell's own levels were never touched while it was occupied,
        # so they are already correct for the restored position.
        changes = self._history.pop()
        if changes is None:
            self._refresh_lines(index)
        else:
            for levels, summary, cell, old_level, old_summary in reversed(changes):
                levels[cell] = old_level
                summary[cell] = old_summary
        self.turn = stone

    def _refresh_lines(self, index: int) -> list[tuple]:
        """Recompute the empty cells on ``index``'s four lines; return the changes."""

        cells = self.cells
        black_levels, white_levels = self.levels[BLACK], self.levels[WHITE]
        black_summary, white_summary = self.summary[BLACK], self.summary[WHITE]
        changes = []
        for direction, line in enumerate(LINE_CELLS[index]):
            for cell in line:
                if cells[cell] != EMPTY:
                    continue
                black, white = line_levels(WINDOW_GETTERS[cell][direction](cells))
                old = black_levels[cell]
                if old[direction] != black:
                    changes.append((black_levels, black_summary, cell, old, black_summary[cell]))
                    new = old[:direction] + (black,) + old[direction + 1:]
                    black_levels[cell] = new
                    black_summary[cell] = cell_summary(new)
                old = white_levels[cell]
                if old[direction] != white:
                    changes.append((white_levels, white_summary, cell, old, white_summary[cell]))
                    new = old[:direction] + (white,) + old[direction + 1:]
                    white_levels[cell] = new
                    white_summary[cell] = cell_summary(new)
        return changes

    def _refresh(self, index: int, direction: int) -> None:
        black, white = line_levels(WINDOW_GETTERS[index][direction](self.cells))
        for player, level in ((BLACK, black), (WHITE, white)):
            levels = self.levels[player]
            old = levels[index]
            if old[direction] != level:
                new = old[:direction] + (level,) + old[direction + 1:]
                levels[index] = new
                self.summary[player][index] = cell_summary(new)

    def winning_cells(self, player: int) -> list[int]:
        summary = self.summary[player]
        return [index for index in self.candidates if summary[index][3]]

    def is_full(self) -> bool:
        return len(self.moves) == SIZE * SIZE
