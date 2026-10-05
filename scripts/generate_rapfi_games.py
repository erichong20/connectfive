"""Rapfi teacher data: Rapfi plays itself from seeded openings; every position is labelled.

Each game starts from ``generate_opening(seed, plies)`` with a seeded number of
plies in ``[--min-plies, --max-plies]`` (diversity: Rapfi is deterministic).
Rapfi then plays both sides at a fixed node budget, one ``BOARD`` request per
move, and our pattern board referees (exact five; games still open at
``--ply-cap`` are adjudicated draws). Each post-opening position records
Rapfi's move as a one-hot policy target, its evaluation for the side to move
(``search_score``, raw Rapfi units; ``search_value = tanh(eval / --eval-scale)``)
and, through the dataset builder, the real final outcome.

Games are appended to ``<out>.games.jsonl`` as they finish and reused on
restart, as in self-play. Example (pilot):

    ./.venv/bin/python scripts/generate_rapfi_games.py \\
        --engine runs/engines/rapfi/build/pbrain-rapfi --nodes 10000 \\
        --games 5000 --seed 200000 --workers 9 --out runs/rapfi-teacher/pilot.npz
"""

import argparse
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from connectfive.dataset import SupervisedDataset, save_dataset, verify_record
from connectfive.match import generate_opening
from connectfive.patterns import ACTION_TO_INDEX, PatternBoard
from connectfive.piskvork import EngineError, PiskvorkEngine
from connectfive.selfplay import game_to_json, load_game_log
from connectfive.teacher import LabelledPosition, TeacherGame, game_record, teacher_games_to_arrays


@dataclass(frozen=True)
class RapfiTeacherConfig:
    engine: str
    nodes: int
    min_plies: int = 2
    max_plies: int = 8
    ply_cap: int = 150
    eval_scale: float = 600.0

    @property
    def name(self) -> str:
        return f"rapfi-n{self.nodes}"


_ENGINE: PiskvorkEngine | None = None


def _engine(config: RapfiTeacherConfig) -> PiskvorkEngine:
    global _ENGINE
    if _ENGINE is None:
        path = Path(config.engine)
        _ENGINE = PiskvorkEngine(
            [str(path.resolve())], cwd=path.parent, name="rapfi",
            info={"rule": 1, "THREAD_NUM": 1, "MAX_NODE": config.nodes,
                  "TIMEOUT_TURN": 30_000, "TIMEOUT_MATCH": 100_000_000},
        )
        _ENGINE.start()
    return _ENGINE


def play_rapfi_game(seed: int, config: RapfiTeacherConfig) -> TeacherGame:
    plies = random.Random(seed).randint(config.min_plies, config.max_plies)
    opening = tuple(generate_opening(seed, plies))
    board = PatternBoard()
    history = list(opening)
    for action in opening:
        board.play(ACTION_TO_INDEX[action])
    positions, winner, adjudicated, nodes = [], None, None, 0
    _engine(config).restart()
    while len(history) < 225:
        if len(history) >= config.ply_cap:
            adjudicated = "ply_cap"
            break
        action, _ = _engine(config).move(history, timeout=60)
        index = ACTION_TO_INDEX[action] if action >= 0 else -1
        if index < 0 or board.cells[index] != 0:
            raise EngineError(f"rapfi played an illegal move {action} in game {seed}")
        score = _engine(config).last_eval() or 0
        nodes += config.nodes
        positions.append(LabelledPosition(
            ply=len(history), player=board.player, best_action=action, policy=((action, 1.0),),
            search_value=math.tanh(score / config.eval_scale), search_score=score, reason="rapfi",
        ))
        mover = board.turn
        won = board.summary[mover][index][3]
        board.play(index)
        history.append(action)
        if won:
            winner = mover - 1
            break
    return TeacherGame(seed, tuple(history), opening, winner, tuple(positions), nodes, adjudicated)


def _play(args) -> TeacherGame | None:
    """Play one game; after an engine failure restart Rapfi and retry once, else skip."""

    global _ENGINE
    for attempt in range(2):
        try:
            return play_rapfi_game(*args)
        except EngineError as error:
            if _ENGINE is not None:
                _ENGINE.close()
            _ENGINE = None
            if attempt:
                print(f"skipping seed {args[0]}: {error}", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--min-plies", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=8)
    parser.add_argument("--ply-cap", type=int, default=150)
    parser.add_argument("--eval-scale", type=float, default=600.0)
    parser.add_argument("--verify-sample", type=int, default=100,
                        help="replay this many random games through the JAX environment")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = RapfiTeacherConfig(str(args.engine), args.nodes, args.min_plies, args.max_plies,
                                args.ply_cap, args.eval_scale)
    seeds = list(range(args.seed, args.seed + args.games))
    log = args.out.with_suffix(".games.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_game_log(log)
    remaining = [seed for seed in seeds if seed not in done]
    started = time.perf_counter()
    with log.open("a") as handle, get_context("spawn").Pool(args.workers) as pool:
        for game in pool.imap_unordered(_play, [(seed, config) for seed in remaining], chunksize=4):
            if game is None:
                continue
            done[game.seed] = game
            handle.write(game_to_json(game) + "\n")
            handle.flush()
    seconds = time.perf_counter() - started
    games = [done[seed] for seed in seeds if seed in done]
    records = tuple(game_record(game, config) for game in games)
    for record in random.Random(0).sample(records, min(args.verify_sample, len(records))):
        verify_record(record)
    dataset = SupervisedDataset(**teacher_games_to_arrays(games, config))
    winners = [game.winner for game in games]
    summary = {
        "black_wins": winners.count(0), "white_wins": winners.count(1), "draws": winners.count(None),
        "capped_draws": sum(game.adjudicated == "ply_cap" for game in games),
        "mean_length": float(np.mean([len(game.moves) for game in games])),
        "seconds": seconds, "workers": args.workers, "resumed_games": len(seeds) - len(remaining),
        "verified_games": min(args.verify_sample, len(records)),
        "skipped_games": len(seeds) - len(games),
    }
    save_dataset(args.out, dataset, records, metadata={
        "teacher": {**config.__dict__, "name": config.name}, "seeds": [seeds[0], seeds[-1]],
        "summary": summary, "platform": platform.platform(),
    })
    print(f"saved {len(dataset)} positions from {len(games)} games to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
