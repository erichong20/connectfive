"""Deterministic game recording and head-to-head evaluation."""

import dataclasses
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from connectfive.agents import Agent
from connectfive.env import NUM_ACTIONS, ConnectFive

_ENV = ConnectFive()
_STEP = jax.jit(_ENV.step)


@dataclass(frozen=True)
class GameRecord:
    black: str
    white: str
    seed: int
    moves: tuple[int, ...]
    rewards: tuple[float, float]
    winner: int | None
    illegal_player: int | None = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class MatchSummary:
    agent: str
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    agent_illegal_moves: int
    opponent_illegal_moves: int
    total_moves: int
    elapsed_seconds: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def average_game_length(self) -> float:
        return self.total_moves / self.games

    def as_dict(self) -> dict:
        result = dataclasses.asdict(self)
        result["win_rate"] = self.win_rate
        result["average_game_length"] = self.average_game_length
        return result


def play_game(black: Agent, white: Agent, seed: int = 0) -> GameRecord:
    """Play and record one deterministic game between two agents."""

    key = jax.random.PRNGKey(seed)
    state = _ENV.init(key)
    agents = (black, white)
    moves: list[int] = []
    illegal_player = None

    while not bool(state.terminated):
        player = int(state.current_player)
        key, move_key = jax.random.split(key)
        action = int(agents[player].select_action(state, move_key))
        moves.append(action)
        in_range = 0 <= action < NUM_ACTIONS
        if not in_range:
            illegal_player = player
            rewards = (-1.0, 1.0) if player == 0 else (1.0, -1.0)
            break
        if not bool(state.legal_action_mask[action]):
            illegal_player = player
        state = _STEP(state, jnp.int32(action))
    else:
        rewards = tuple(float(value) for value in state.rewards)

    winner = 0 if rewards[0] > 0 else 1 if rewards[1] > 0 else None
    return GameRecord(
        black=black.name,
        white=white.name,
        seed=seed,
        moves=tuple(moves),
        rewards=rewards,
        winner=winner,
        illegal_player=illegal_player,
    )


def evaluate_agents(agent: Agent, opponent: Agent, games: int, seed: int = 0) -> MatchSummary:
    """Evaluate ``agent`` with colors alternated each game."""

    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number for paired colors")

    start = time.perf_counter()
    wins = losses = draws = agent_illegal = opponent_illegal = total_moves = 0
    for game_index in range(games):
        agent_player = game_index % 2
        black, white = (agent, opponent) if agent_player == 0 else (opponent, agent)
        record = play_game(black, white, seed + game_index)
        total_moves += len(record.moves)
        if record.winner is None:
            draws += 1
        elif record.winner == agent_player:
            wins += 1
        else:
            losses += 1
        if record.illegal_player == agent_player:
            agent_illegal += 1
        elif record.illegal_player is not None:
            opponent_illegal += 1

    return MatchSummary(
        agent=agent.name,
        opponent=opponent.name,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        agent_illegal_moves=agent_illegal,
        opponent_illegal_moves=opponent_illegal,
        total_moves=total_moves,
        elapsed_seconds=time.perf_counter() - start,
    )
