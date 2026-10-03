"""Deterministic game recording and head-to-head evaluation."""

import dataclasses
import math
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from connectfive.agents import Agent
from connectfive.env import BOARD_SIZE, NUM_ACTIONS, ConnectFive

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
    opening_moves: tuple[int, ...] = ()
    move_diagnostics: tuple["MoveDiagnostic", ...] = ()
    illegal_player: int | None = None
    # Why a game was stopped early as a draw: "dead" (no five can ever be made,
    # see ``is_dead_draw``) or "ply_cap" (a self-play length limit).
    adjudicated: str | None = None

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
    agent_black_wins: int
    agent_black_losses: int
    agent_black_draws: int
    agent_black_games: int
    agent_white_wins: int
    agent_white_losses: int
    agent_white_draws: int
    agent_white_games: int
    agent_move_count: int
    agent_total_move_ms: float
    agent_total_search_nodes: int
    agent_total_completed_depth: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def average_game_length(self) -> float:
        return self.total_moves / self.games

    @property
    def average_agent_move_ms(self) -> float:
        return self.agent_total_move_ms / self.agent_move_count if self.agent_move_count else 0.0

    @property
    def average_search_nodes(self) -> float:
        return (
            self.agent_total_search_nodes / self.agent_move_count
            if self.agent_move_count
            else 0.0
        )

    @property
    def average_completed_depth(self) -> float:
        return (
            self.agent_total_completed_depth / self.agent_move_count
            if self.agent_move_count
            else 0.0
        )

    @property
    def score_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def score_rate_95_ci(self) -> tuple[float, float]:
        """Approximate Wilson interval with each draw counted as half a point."""

        z = 1.959963984540054
        proportion = self.score_rate
        denominator = 1 + z * z / self.games
        center = (proportion + z * z / (2 * self.games)) / denominator
        margin = z * math.sqrt(
            proportion * (1 - proportion) / self.games
            + z * z / (4 * self.games * self.games)
        ) / denominator
        return center - margin, center + margin

    def as_dict(self) -> dict:
        result = dataclasses.asdict(self)
        result["win_rate"] = self.win_rate
        result["score_rate"] = self.score_rate
        result["score_rate_95_ci"] = self.score_rate_95_ci
        result["average_game_length"] = self.average_game_length
        result["average_agent_move_ms"] = self.average_agent_move_ms
        result["average_search_nodes"] = self.average_search_nodes
        result["average_completed_depth"] = self.average_completed_depth
        return result


def combine_summaries(summaries: list[dict]) -> MatchSummary:
    """Add up match blocks (``MatchSummary.as_dict`` output) between one pair of agents."""

    if not summaries:
        raise ValueError("no summaries to combine")
    names = {(entry["agent"], entry["opponent"]) for entry in summaries}
    if len(names) != 1:
        raise ValueError(f"summaries mix different agents: {sorted(names)}")
    fields = {}
    for field_ in dataclasses.fields(MatchSummary):
        values = [entry[field_.name] for entry in summaries]
        fields[field_.name] = values[0] if field_.name in ("agent", "opponent") else sum(values)
    return MatchSummary(**fields)


@dataclass(frozen=True)
class MoveDiagnostic:
    player: int
    action: int
    elapsed_ms: float = field(compare=False)
    search_nodes: int = 0
    completed_depth: int = 0


def play_game(
    black: Agent,
    white: Agent,
    seed: int = 0,
    opening_moves: tuple[int, ...] = (),
) -> GameRecord:
    """Play and record one deterministic game between two agents."""

    key = jax.random.PRNGKey(seed)
    state = _ENV.init(key)
    agents = (black, white)
    moves: list[int] = []
    diagnostics: list[MoveDiagnostic] = []
    illegal_player = None

    for action in opening_moves:
        if bool(state.terminated) or not 0 <= action < NUM_ACTIONS:
            raise ValueError("opening moves must form a legal, non-terminal sequence")
        if not bool(state.legal_action_mask[action]):
            raise ValueError("opening moves must not repeat an occupied intersection")
        moves.append(action)
        state = _STEP(state, jnp.int32(action))
    if bool(state.terminated):
        raise ValueError("opening moves must not finish the game")

    while not bool(state.terminated):
        player = int(state.current_player)
        key, move_key = jax.random.split(key)
        move_started = time.perf_counter()
        action = int(agents[player].select_action(state, move_key))
        elapsed_ms = (time.perf_counter() - move_started) * 1_000
        search = getattr(agents[player], "last_search", None)
        stats = getattr(search, "stats", None)
        diagnostics.append(
            MoveDiagnostic(
                player=player,
                action=action,
                elapsed_ms=elapsed_ms,
                search_nodes=int(getattr(stats, "nodes", 0)),
                completed_depth=int(getattr(stats, "completed_depth", 0)),
            )
        )
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
        opening_moves=opening_moves,
        move_diagnostics=tuple(diagnostics),
        illegal_player=illegal_player,
    )


def generate_opening(seed: int, plies: int) -> tuple[int, ...]:
    """Generate a reproducible local opening without completing a five."""

    if not 0 <= plies <= 8:
        raise ValueError("opening plies must be between 0 and 8")
    if plies == 0:
        return ()
    rng = np.random.default_rng(seed)
    board = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)
    moves = []
    center = BOARD_SIZE // 2
    for ply in range(plies):
        empty = np.argwhere(board == -1)
        occupied = np.argwhere(board != -1)
        if not len(occupied):
            candidates = empty[
                np.max(np.abs(empty - np.array([center, center])), axis=1) <= 2
            ]
        else:
            candidates = empty[
                np.array([
                    np.any(np.max(np.abs(occupied - point), axis=1) <= 2)
                    for point in empty
                ])
            ]
        row, col = candidates[int(rng.integers(len(candidates)))]
        board[row, col] = ply % 2
        moves.append(int(row) * BOARD_SIZE + int(col))
    return tuple(moves)


def evaluate_agents(
    agent: Agent,
    opponent: Agent,
    games: int,
    seed: int = 0,
    opening_plies: int = 0,
) -> MatchSummary:
    """Evaluate ``agent`` with colors alternated each game."""

    summary, _ = evaluate_agents_with_records(
        agent, opponent, games, seed, opening_plies
    )
    return summary


def evaluate_agents_with_records(
    agent: Agent,
    opponent: Agent,
    games: int,
    seed: int = 0,
    opening_plies: int = 0,
) -> tuple[MatchSummary, tuple[GameRecord, ...]]:
    """Evaluate two agents and retain replayable records for every game."""

    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number for paired colors")

    start = time.perf_counter()
    wins = losses = draws = agent_illegal = opponent_illegal = total_moves = 0
    black_wins = black_losses = black_draws = 0
    white_wins = white_losses = white_draws = 0
    agent_move_count = total_move_ms = total_nodes = total_depth = 0
    records = []
    for game_index in range(games):
        agent_player = game_index % 2
        black, white = (agent, opponent) if agent_player == 0 else (opponent, agent)
        pair_seed = seed + game_index // 2
        opening = generate_opening(pair_seed, opening_plies)
        record = play_game(black, white, pair_seed, opening)
        records.append(record)
        total_moves += len(record.moves)
        if record.winner is None:
            draws += 1
            if agent_player == 0:
                black_draws += 1
            else:
                white_draws += 1
        elif record.winner == agent_player:
            wins += 1
            if agent_player == 0:
                black_wins += 1
            else:
                white_wins += 1
        else:
            losses += 1
            if agent_player == 0:
                black_losses += 1
            else:
                white_losses += 1
        if record.illegal_player == agent_player:
            agent_illegal += 1
        elif record.illegal_player is not None:
            opponent_illegal += 1
        for diagnostic in record.move_diagnostics:
            if diagnostic.player == agent_player:
                agent_move_count += 1
                total_move_ms += diagnostic.elapsed_ms
                total_nodes += diagnostic.search_nodes
                total_depth += diagnostic.completed_depth

    summary = MatchSummary(
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
        agent_black_wins=black_wins,
        agent_black_losses=black_losses,
        agent_black_draws=black_draws,
        agent_black_games=games // 2,
        agent_white_wins=white_wins,
        agent_white_losses=white_losses,
        agent_white_draws=white_draws,
        agent_white_games=games // 2,
        agent_move_count=agent_move_count,
        agent_total_move_ms=total_move_ms,
        agent_total_search_nodes=total_nodes,
        agent_total_completed_depth=total_depth,
    )
    return summary, tuple(records)
