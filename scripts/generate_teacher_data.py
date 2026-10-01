"""Generate pattern-search self-play games labelled by the teacher's search."""

import argparse
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from connectfive.dataset import SupervisedDataset, save_dataset, verify_record
from connectfive.teacher import (
    TeacherConfig,
    config_dict,
    game_record,
    generate_teacher_games,
    teacher_games_to_arrays,
)


def git_state() -> dict:
    def run(*args):
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--node-budget", type=int, default=2_000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--sample-plies", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("runs/teacher-v2/games.npz"))
    args = parser.parse_args()

    config = TeacherConfig(
        max_depth=args.max_depth,
        width=args.width,
        node_budget=args.node_budget,
        opening_plies=args.opening_plies,
        sample_plies=args.sample_plies,
    )
    seeds = list(range(args.seed, args.seed + args.games))
    started = time.perf_counter()
    games = generate_teacher_games(seeds, config, workers=args.workers)
    generation_seconds = time.perf_counter() - started
    records = tuple(game_record(game, config) for game in games)
    for record in records:
        verify_record(record)
    dataset = SupervisedDataset(**teacher_games_to_arrays(games, config))
    winners = [game.winner for game in games]
    lengths = [len(game.moves) for game in games]
    summary = {
        "black_wins": winners.count(0),
        "white_wins": winners.count(1),
        "draws": winners.count(None),
        "mean_length": float(np.mean(lengths)),
        "generation_seconds": generation_seconds,
        "workers": args.workers,
        "total_search_nodes": sum(game.elapsed_nodes for game in games),
    }
    save_dataset(
        args.out,
        dataset,
        records,
        metadata={
            "seeds": [seeds[0], seeds[-1]],
            "teacher": config_dict(config),
            "summary": summary,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "git": git_state(),
        },
    )
    print(f"saved {len(dataset)} positions from {len(games)} games to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
