import random

import jax
import numpy as np
import pytest

from connectfive import BOARD_SIZE, ConnectFive
from connectfive.pattern_search import (
    INFINITY,
    LOSS,
    MATE,
    WIN_THRESHOLD,
    PatternSearch,
    PatternSearchAgent,
    search_board,
)
from connectfive.patterns import (
    ACTION_TO_INDEX,
    BLACK,
    EMPTY,
    WHITE,
    PatternBoard,
    _level,
)
from connectfive.tactical_suite import load_tactical_suite, position_state


def window(pattern: str) -> tuple[int, ...]:
    """Build an 11-cell window; the centre character must be 'X'."""

    assert len(pattern) == 11 and pattern[5] == "X"
    return tuple({".": 0, "X": 1, "O": 2}[c] for c in pattern)


@pytest.mark.parametrize(
    ("pattern", "level"),
    [
        ("...XXXXX...", 6),  # exact five
        ("..XXXXXX...", 0),  # overline: never a five in this line
        ("..X.XXXX...", 4),  # would-be five at the gap is an overline
        ("....XXXX...", 5),  # open four
        ("...OXXXX...", 4),  # four blocked on one side
        ("...XXX.X...", 4),  # broken four: one winning cell
        ("....XX.X...", 3),  # broken three: the gap makes an open four
        ("....XXX....", 3),  # open three
        ("...OXXX....", 2),  # closed three
        ("...O.XXX...", 3),  # three with space to become an open four
        ("....XX.....", 1),  # open two
        (".....X.....", 0),
    ],
)
def test_line_levels_follow_exact_five_rules(pattern, level):
    assert _level(window(pattern)) == level


def board_from_stones(stones, player=0):
    board = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)
    for row, col, color in stones:
        board[row, col] = color
    return board, player


def test_incremental_updates_match_a_full_rebuild():
    rng = random.Random(3)
    board = PatternBoard()
    board.play(ACTION_TO_INDEX[112])
    for _ in range(150):
        if rng.random() < 0.3 and len(board.moves) > 1:
            board.undo()
        else:
            board.play(rng.choice(sorted(board.candidates)))
        rebuilt = PatternBoard.from_array(board.to_array(), board.player)
        assert rebuilt.hash == board.hash
        assert rebuilt.candidates == board.candidates
        for index in ACTION_TO_INDEX:
            if board.cells[index] == EMPTY:
                for player in (BLACK, WHITE):
                    assert rebuilt.summary[player][index] == board.summary[player][index]


def test_overline_cell_is_not_a_winning_cell():
    stones = [(7, c, 0) for c in (1, 2, 3, 4, 6)]
    board = PatternBoard.from_array(*board_from_stones(stones))
    assert ACTION_TO_INDEX[7 * BOARD_SIZE + 5] not in board.winning_cells(BLACK)


def test_search_solves_the_tactical_suite():
    agent = PatternSearchAgent(node_budget=1_000)
    for index, position in enumerate(load_tactical_suite()):
        action = agent.select_action(position_state(position), jax.random.PRNGKey(index))
        assert action in position.acceptable_actions, position.name


def test_search_blocks_an_open_three():
    stones = [(7, 6, 1), (7, 7, 1), (7, 8, 1), (3, 3, 0), (10, 11, 0)]
    result = search_board(*board_from_stones(stones, player=0), node_budget=2_000)
    blocks = {7 * BOARD_SIZE + 5, 7 * BOARD_SIZE + 9, 7 * BOARD_SIZE + 4, 7 * BOARD_SIZE + 10}
    assert set(result.actions) <= blocks


def test_double_five_threat_is_recognised_as_lost():
    stones = [(7, c, 1) for c in range(3, 7)] + [(9, c, 1) for c in range(3, 7)]
    stones += [(7, 2, 0), (9, 2, 0), (0, 0, 0)]
    board = PatternBoard.from_array(*board_from_stones(stones, player=0))
    kind, _ = PatternSearch(board).generate(12)
    assert kind == LOSS


def test_vcf_finds_a_win_by_continuous_fours():
    # Black to move: a closed four-three built from fours only.
    stones = [
        (7, 4, 0), (7, 5, 0), (7, 6, 0), (7, 3, 1),  # closed three -> four at (7,7)
        (4, 7, 0), (5, 7, 0), (6, 7, 0), (3, 7, 1),  # after (7,7): column four too
        (12, 12, 1), (12, 13, 1),
    ]
    board = PatternBoard.from_array(*board_from_stones(stones, player=0))
    search = PatternSearch(board, node_budget=5_000)
    move = search.vcf(6)
    assert move is not None
    result = search_board(*board_from_stones(stones, player=0))
    assert result.score >= WIN_THRESHOLD


def _reference(search: PatternSearch, depth: int, ply: int) -> int:
    """Plain negamax without pruning or tables, mirroring PatternSearch rules."""

    from connectfive.pattern_search import BLOCK, FORCE_WIN, QUIET, VCF_WIN, WIN

    board = search.board
    if board.is_full():
        return 0
    kind, moves = search.generate(search.width)
    if kind == WIN:
        return MATE - ply - 1
    if kind == LOSS:
        return -(MATE - ply - 2)
    if kind == FORCE_WIN:
        return MATE - ply - 3
    forced = kind == BLOCK
    if (depth <= 0 and not forced) or ply >= search.max_ply:
        if kind == QUIET and search.leaf_vcf_depth and search.vcf(search.leaf_vcf_depth) is not None:
            return VCF_WIN - ply
        return search.evaluate()
    best = -INFINITY
    for move in moves:
        board.play(move)
        best = max(best, -_reference(search, depth if forced else depth - 1, ply + 1))
        board.undo()
    return best


@pytest.mark.parametrize("seed", range(4))
def test_alpha_beta_and_table_match_plain_negamax(seed):
    rng = random.Random(seed)
    board = PatternBoard()
    board.play(ACTION_TO_INDEX[112])
    for _ in range(rng.randrange(5, 12)):
        board.play(rng.choice(sorted(board.candidates)))
    options = {"width": 5, "node_budget": 10**7, "leaf_vcf_depth": 2}
    fast = PatternSearch(board, **options)
    reference = PatternSearch(board, **options)
    for depth in (1, 2, 3):
        # Reusing the table across iterations is exactly what iterative deepening does.
        assert fast.negamax(depth, -INFINITY, INFINITY, 0) == _reference(reference, depth, 0)


def test_root_scores_are_exact_and_contain_the_choice():
    env = ConnectFive()
    state = env.init(jax.random.PRNGKey(0))
    for action in (112, 113, 127, 97):
        state = env.step(state, action)
    result = search_board(np.asarray(state._board), int(state.current_player))
    scores = dict(result.root_scores)
    assert all(action in scores for action in result.actions)
    assert max(scores.values()) == result.score
    target = result.policy_target()
    assert abs(sum(target.values()) - 1) < 1e-9


def test_pattern_board_undo_restores_exactly_what_a_rebuild_computes():
    import random

    import numpy as np

    from connectfive.patterns import ACTION_TO_INDEX, PatternBoard

    rng = random.Random(4)
    for trial in range(20):
        board = PatternBoard()
        if trial % 2:  # also undo past stones loaded from an array
            start = np.full((15, 15), -1, dtype=np.int8)
            for k, action in enumerate(rng.sample(range(225), 6)):
                start.reshape(-1)[action] = k % 2
            board = PatternBoard.from_array(start, 0)
        snapshots = []
        for _ in range(rng.randint(5, 40)):
            empty = [i for i in ACTION_TO_INDEX if board.cells[i] == 0]
            snapshots.append((board.hash, {p: dict(s) for p, s in board.summary.items()},
                              {p: dict(l) for p, l in board.levels.items()},
                              set(board.candidates)))
            board.play(rng.choice(empty))
            rebuilt = PatternBoard.from_array(board.to_array(), board.player)
            # Occupied cells keep stale levels by design; only empty ones are read.
            for cell in (i for i in ACTION_TO_INDEX if board.cells[i] == 0):
                for player in (1, 2):
                    assert board.levels[player][cell] == rebuilt.levels[player][cell]
                    assert board.summary[player][cell] == rebuilt.summary[player][cell]
        while snapshots:
            board.undo()
            hash_, summary, levels, candidates = snapshots.pop()
            assert (board.hash, board.summary, board.levels, board.candidates) == (
                hash_, summary, levels, candidates)
        if trial % 2:
            for _ in range(6):
                board.undo()
            assert board.levels == PatternBoard().levels
