"""Parallel teacher self-play that labels every position with search results.

Each labelled position keeps:

- the teacher's best move (hard target) and a soft distribution over the
  near-best moves whose root scores were searched exactly;
- the teacher's score for the mover, squashed to [-1, 1];
- the real final outcome from the mover's perspective;
- game id, ply, player, and the generating teacher configuration.

Diversity comes from a seeded random opening and from *sampling* the first few
moves after the opening from the soft target. The label is still the teacher's
recommendation, never the sampled move.

This module avoids JAX so games can run in plain worker processes.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from multiprocessing import get_context

import numpy as np

from connectfive.match import GameRecord, generate_opening
from connectfive.pattern_search import PatternSearch, score_to_value
from connectfive.patterns import ACTION_TO_INDEX, SIZE, PatternBoard

NUM_ACTIONS = SIZE * SIZE


@dataclass(frozen=True)
class TeacherConfig:
    max_depth: int = 6
    width: int = 12
    node_budget: int = 2_000
    vcf_depth: int = 8
    opening_plies: int = 4
    sample_plies: int = 6
    policy_temperature: float = 800.0
    max_game_plies: int = SIZE * SIZE

    @property
    def name(self) -> str:
        return f"pattern-n{self.node_budget}-d{self.max_depth}-w{self.width}"


@dataclass(frozen=True)
class LabelledPosition:
    ply: int
    player: int
    best_action: int
    policy: tuple[tuple[int, float], ...]
    search_value: float
    search_score: int
    reason: str


@dataclass(frozen=True)
class TeacherGame:
    seed: int
    moves: tuple[int, ...]
    opening_moves: tuple[int, ...]
    winner: int | None
    positions: tuple[LabelledPosition, ...]
    elapsed_nodes: int


def play_teacher_game(seed: int, config: TeacherConfig) -> TeacherGame:
    """Play one seeded self-play game and label every post-opening position."""

    rng = random.Random(seed)
    board = PatternBoard()
    opening = generate_opening(seed, config.opening_plies)
    for action in opening:
        board.play(ACTION_TO_INDEX[action])
    positions = []
    winner = None
    total_nodes = 0
    while not board.is_full() and len(board.moves) < config.max_game_plies:
        ply = len(board.moves)
        search = PatternSearch(
            board,
            max_depth=config.max_depth,
            width=config.width,
            node_budget=config.node_budget,
            vcf_depth=config.vcf_depth,
        )
        result = search.run()
        total_nodes += result.stats.nodes + result.stats.vcf_nodes
        policy = result.policy_target(config.policy_temperature)
        best = rng.choice(result.actions)
        positions.append(
            LabelledPosition(
                ply=ply,
                player=board.player,
                best_action=best,
                policy=tuple(sorted(policy.items())),
                search_value=score_to_value(result.score),
                search_score=result.score,
                reason=result.stats.reason,
            )
        )
        if ply - len(opening) < config.sample_plies and len(policy) > 1:
            actions, weights = zip(*sorted(policy.items()))
            move = rng.choices(actions, weights=weights)[0]
        else:
            move = best
        index = ACTION_TO_INDEX[move]
        mover = board.turn
        wins = board.summary[mover][index][3]
        board.play(index)
        if wins:
            winner = mover - 1
            break
    return TeacherGame(
        seed=seed,
        moves=tuple(board.moves),
        opening_moves=opening,
        winner=winner,
        positions=tuple(positions),
        elapsed_nodes=total_nodes,
    )


def _play(args: tuple[int, TeacherConfig]) -> TeacherGame:
    return play_teacher_game(*args)


def generate_teacher_games(
    seeds: list[int], config: TeacherConfig, workers: int = 1
) -> list[TeacherGame]:
    """Play games in parallel; results keep the order of ``seeds``."""

    jobs = [(seed, config) for seed in seeds]
    if workers <= 1:
        return [_play(job) for job in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(_play, jobs, chunksize=1)


def game_record(game: TeacherGame, config: TeacherConfig) -> GameRecord:
    """Express a teacher game in the shared replayable record format."""

    rewards = (0.0, 0.0)
    if game.winner == 0:
        rewards = (1.0, -1.0)
    elif game.winner == 1:
        rewards = (-1.0, 1.0)
    return GameRecord(
        black=config.name,
        white=config.name,
        seed=game.seed,
        moves=game.moves,
        rewards=rewards,
        winner=game.winner,
        opening_moves=game.opening_moves,
    )


def teacher_games_to_arrays(games: list[TeacherGame], config: TeacherConfig) -> dict:
    """Build the arrays of a ``SupervisedDataset`` from labelled games."""

    from connectfive.network import encode_board

    features, legal, actions, values, search_values = [], [], [], [], []
    policies, game_ids, plies, players = [], [], [], []
    for game_id, game in enumerate(games):
        board = np.full(NUM_ACTIONS, -1, dtype=np.int8)
        labelled = {position.ply: position for position in game.positions}
        for ply, action in enumerate(game.moves):
            position = labelled.get(ply)
            player = ply % 2
            if position is not None:
                assert position.player == player
                features.append(encode_board(board.reshape(SIZE, SIZE), player))
                legal.append(board == -1)
                actions.append(position.best_action)
                outcome = 0.0 if game.winner is None else (
                    1.0 if game.winner == player else -1.0
                )
                values.append(outcome)
                search_values.append(position.search_value)
                policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for move, probability in position.policy:
                    policy[move] = probability
                policies.append(policy)
                game_ids.append(game_id)
                plies.append(ply)
                players.append(player)
            board[action] = player
    return {
        "features": np.stack(features),
        "legal_action_masks": np.stack(legal),
        "actions": np.asarray(actions, dtype=np.int16),
        "values": np.asarray(values, dtype=np.float32),
        "game_ids": np.asarray(game_ids, dtype=np.int32),
        "plies": np.asarray(plies, dtype=np.int16),
        "players": np.asarray(players, dtype=np.int8),
        "generators": np.asarray([config.name] * len(actions), dtype=np.str_),
        "policy_targets": np.stack(policies).astype(np.float16),
        "search_values": np.asarray(search_values, dtype=np.float32),
    }


def config_dict(config: TeacherConfig) -> dict:
    return {**asdict(config), "name": config.name}


def label_game(
    seed: int,
    moves: tuple[int, ...],
    opening_moves: tuple[int, ...],
    winner: int | None,
    config: TeacherConfig,
) -> TeacherGame:
    """Label every post-opening position of a game someone else played.

    This is the DAgger step: the *student* chooses which positions occur, and
    the teacher says what it would have played in each of them.
    """

    board = PatternBoard()
    for action in opening_moves:
        board.play(ACTION_TO_INDEX[action])
    positions = []
    total_nodes = 0
    for ply in range(len(opening_moves), len(moves)):
        result = PatternSearch(
            board,
            max_depth=config.max_depth,
            width=config.width,
            node_budget=config.node_budget,
            vcf_depth=config.vcf_depth,
        ).run()
        total_nodes += result.stats.nodes + result.stats.vcf_nodes
        policy = result.policy_target(config.policy_temperature)
        positions.append(
            LabelledPosition(
                ply=ply,
                player=board.player,
                best_action=min(result.actions),
                policy=tuple(sorted(policy.items())),
                search_value=score_to_value(result.score),
                search_score=result.score,
                reason=result.stats.reason,
            )
        )
        board.play(ACTION_TO_INDEX[moves[ply]])
    return TeacherGame(
        seed=seed,
        moves=tuple(moves),
        opening_moves=tuple(opening_moves),
        winner=winner,
        positions=tuple(positions),
        elapsed_nodes=total_nodes,
    )


def _label(args) -> TeacherGame:
    return label_game(*args)


def label_games(games: list[tuple], config: TeacherConfig, workers: int = 1) -> list[TeacherGame]:
    """Label ``(seed, moves, opening_moves, winner)`` games in parallel."""

    jobs = [(*game, config) for game in games]
    if workers <= 1:
        return [_label(job) for job in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(_label, jobs, chunksize=1)
