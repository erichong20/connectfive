"""Loader and evaluator for small, human-readable tactical fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from connectfive.agents import Agent
from connectfive.env import BOARD_SIZE, EMPTY, ConnectFive

DEFAULT_SUITE = Path(__file__).parents[2] / "tactics" / "essential.json"


@dataclass(frozen=True)
class TacticalPosition:
    name: str
    description: str
    player: int
    stones: tuple[tuple[int, int, int], ...]
    acceptable_actions: tuple[int, ...]


@dataclass(frozen=True)
class TacticalCaseResult:
    name: str
    passed: bool
    selected_action: int
    acceptable_actions: tuple[int, ...]


def load_tactical_suite(path: Path = DEFAULT_SUITE) -> tuple[TacticalPosition, ...]:
    raw_positions = json.loads(path.read_text())
    positions = []
    for raw in raw_positions:
        acceptable = tuple(
            int(row) * BOARD_SIZE + int(col) for row, col in raw["acceptable_moves"]
        )
        positions.append(
            TacticalPosition(
                name=raw["name"],
                description=raw["description"],
                player=int(raw["player"]),
                stones=tuple(tuple(map(int, stone)) for stone in raw["stones"]),
                acceptable_actions=acceptable,
            )
        )
    return tuple(positions)


def position_state(position: TacticalPosition):
    env = ConnectFive()
    state = env.init(jax.random.PRNGKey(0))
    board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.int8)
    for row, col, player in position.stones:
        board[row, col] = player
    return state.replace(
        current_player=jnp.int32(position.player),
        _board=jnp.asarray(board),
        legal_action_mask=jnp.asarray(board.reshape(-1) == EMPTY),
        _step_count=jnp.int32(len(position.stones)),
    )


def evaluate_tactical_suite(
    agent: Agent,
    positions: tuple[TacticalPosition, ...] | None = None,
    seed: int = 0,
) -> tuple[TacticalCaseResult, ...]:
    suite = positions or load_tactical_suite()
    results = []
    key = jax.random.PRNGKey(seed)
    for position in suite:
        key, move_key = jax.random.split(key)
        selected = agent.select_action(position_state(position), move_key)
        results.append(
            TacticalCaseResult(
                name=position.name,
                passed=selected in position.acceptable_actions,
                selected_action=selected,
                acceptable_actions=position.acceptable_actions,
            )
        )
    return tuple(results)
