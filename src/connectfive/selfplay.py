"""AlphaZero-lite self-play: guided MCTS plays itself and records visit targets.

Each labelled position keeps the MCTS root visit distribution (policy target),
the root value estimate (search value), and the real final outcome. Root
Dirichlet noise and visit-proportional sampling in the first moves provide
exploration; later moves play the most-visited child.

Workers load the network once and run single-threaded XLA, so several games
can run in parallel on one CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from connectfive.match import generate_opening
from connectfive.teacher import LabelledPosition, TeacherGame


@dataclass(frozen=True)
class SelfPlayConfig:
    checkpoint: str
    simulations: int = 200
    root_noise: float = 0.25
    noise_alpha: float = 0.3
    opening_plies: int = 4
    sample_plies: int = 8
    c_puct: float = 1.5
    # Reject random openings the network rates as lopsided (|value| above this).
    balance_threshold: float | None = None
    balance_attempts: int = 32

    @property
    def name(self) -> str:
        return f"selfplay:{Path(self.checkpoint).name}-s{self.simulations}"


_EVALUATOR = None


def _init_worker(checkpoint: str) -> None:
    global _EVALUATOR
    from connectfive.guided_search import NetworkEvaluator

    _EVALUATOR = NetworkEvaluator.from_checkpoint(Path(checkpoint))


def choose_opening(seed: int, config: SelfPlayConfig, evaluator) -> tuple[int, ...]:
    """Return a seeded opening, optionally the first one the network calls balanced.

    Black moves first and wins most random openings, so the value targets are
    skewed. With ``balance_threshold`` set, candidate openings are drawn from
    seeds ``seed * 64 + attempt`` and the first whose network value (for the
    side to move) is within the threshold is used; otherwise the most balanced.
    """

    from connectfive.patterns import ACTION_TO_INDEX, PatternBoard

    if config.balance_threshold is None:
        return generate_opening(seed, config.opening_plies)
    if not 1 <= config.balance_attempts <= 64:
        raise ValueError("balance_attempts must be between 1 and 64")
    best, best_value = (), float("inf")
    for attempt in range(config.balance_attempts):
        opening = generate_opening(seed * 64 + attempt, config.opening_plies)
        board = PatternBoard()
        for action in opening:
            board.play(ACTION_TO_INDEX[action])
        value = abs(evaluator(board)[1])
        if value <= config.balance_threshold:
            return opening
        if value < best_value:
            best, best_value = opening, value
    return best


def play_selfplay_game(seed: int, config: SelfPlayConfig) -> TeacherGame:
    from connectfive.guided_search import GuidedMCTS
    from connectfive.patterns import ACTION_TO_INDEX, PatternBoard

    if _EVALUATOR is None:
        _init_worker(config.checkpoint)
    rng = np.random.default_rng(seed)
    board = PatternBoard()
    opening = choose_opening(seed, config, _EVALUATOR)
    for action in opening:
        board.play(ACTION_TO_INDEX[action])
    positions, winner, simulations = [], None, 0
    while not board.is_full():
        ply = len(board.moves)
        search = GuidedMCTS(
            board, _EVALUATOR, time_limit=1e9, max_simulations=config.simulations,
            c_puct=config.c_puct, root_noise=config.root_noise,
            noise_alpha=config.noise_alpha, rng=rng,
        )
        result = search.run()
        simulations += search.simulations
        visits = dict(result.root_scores)
        total = sum(visits.values())
        best = int(rng.choice(result.actions))
        if total:
            policy = {action: count / total for action, count in visits.items() if count}
        else:
            # Forced or proven positions skip simulation: the target is the forced move.
            policy = {best: 1.0}
        positions.append(LabelledPosition(
            ply=ply, player=board.player, best_action=best,
            policy=tuple(sorted(policy.items())),
            search_value=max(-1.0, min(1.0, result.score / 1000)),
            search_score=result.score, reason=result.stats.reason,
        ))
        if ply - len(opening) < config.sample_plies and len(policy) > 1:
            actions, weights = zip(*sorted(policy.items()))
            move = int(rng.choice(actions, p=np.asarray(weights) / sum(weights)))
        else:
            move = best
        index = ACTION_TO_INDEX[move]
        mover = board.turn
        won = board.summary[mover][index][3]
        board.play(index)
        if won:
            winner = mover - 1
            break
    return TeacherGame(seed, tuple(board.moves), opening, winner, tuple(positions), simulations)


def _play(args) -> TeacherGame:
    return play_selfplay_game(*args)


def generate_selfplay_games(seeds: list[int], config: SelfPlayConfig,
                            workers: int = 1) -> list[TeacherGame]:
    jobs = [(seed, config) for seed in seeds]
    if workers <= 1:
        return [_play(job) for job in jobs]
    # Spawned workers inherit this before importing JAX: one XLA thread each.
    previous = os.environ.get("XLA_FLAGS")
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    try:
        with get_context("spawn").Pool(
            workers, initializer=_init_worker, initargs=(config.checkpoint,)
        ) as pool:
            return pool.map(_play, jobs, chunksize=1)
    finally:
        if previous is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = previous
