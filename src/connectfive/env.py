"""PGX-compatible freestyle Connect Five on a 19x19 board."""

from typing import Any

import jax.numpy as jnp
from pgx import core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

BOARD_SIZE = 19
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE
EMPTY = -1


@dataclass
class State(core.State):
    """Immutable Connect Five state.

    ``_board`` contains -1 for empty, 0 for black, and 1 for white.
    """

    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=jnp.bool_)
    rewards: Array = jnp.zeros(2, dtype=jnp.float32)
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(NUM_ACTIONS, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _board: Array = jnp.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=jnp.int8)

    @property
    def env_id(self) -> Any:
        return "connect_five"


class ConnectFive(core.Env):
    """Freestyle Gomoku with the common PGX environment interface."""

    def _init(self, key: PRNGKey) -> State:
        del key  # The standard starting position is deterministic.
        return State()

    def _step(self, state: core.State, action: Array, key: PRNGKey | None) -> State:
        del key
        assert isinstance(state, State)

        row, col = jnp.divmod(action, BOARD_SIZE)
        player = state.current_player
        board = state._board.at[row, col].set(player.astype(jnp.int8))
        won = _has_five(board, row, col, player)
        board_full = state._step_count == NUM_ACTIONS
        terminated = won | board_full
        win_rewards = jnp.where(
            player == 0,
            jnp.array([1.0, -1.0], dtype=jnp.float32),
            jnp.array([-1.0, 1.0], dtype=jnp.float32),
        )
        rewards = jnp.where(won, win_rewards, jnp.zeros(2, dtype=jnp.float32))

        return state.replace(
            current_player=1 - player,
            rewards=rewards,
            terminated=terminated,
            legal_action_mask=(board.reshape(-1) == EMPTY),
            _board=board,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return jnp.stack(
            (state._board == player_id, state._board == (1 - player_id)), axis=-1
        )

    @property
    def id(self) -> Any:
        return "connect_five"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _has_five(board: Array, row: Array, col: Array, player: Array) -> Array:
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    wins = []
    for dr, dc in directions:
        run = 1 + _ray_length(board, row, col, player, dr, dc)
        run += _ray_length(board, row, col, player, -dr, -dc)
        wins.append(run >= 5)
    return jnp.any(jnp.stack(wins))


def _ray_length(
    board: Array, row: Array, col: Array, player: Array, dr: int, dc: int
) -> Array:
    count = jnp.int32(0)
    contiguous = jnp.bool_(True)
    for distance in range(1, 5):
        r = row + dr * distance
        c = col + dc * distance
        in_bounds = (0 <= r) & (r < BOARD_SIZE) & (0 <= c) & (c < BOARD_SIZE)
        stone_matches = board[jnp.clip(r, 0, BOARD_SIZE - 1), jnp.clip(c, 0, BOARD_SIZE - 1)] == player
        contiguous = contiguous & in_bounds & stone_matches
        count += contiguous
    return count
