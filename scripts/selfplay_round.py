"""One AlphaZero-lite round: guided-MCTS self-play with visit-count targets."""

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np

from connectfive.dataset import SupervisedDataset, save_dataset, verify_record
from connectfive.selfplay import SelfPlayConfig, generate_selfplay_games
from connectfive.teacher import game_record, teacher_games_to_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--balance-threshold", type=float, default=None,
                        help="reject openings whose |network value| exceeds this")
    parser.add_argument("--balance-attempts", type=int, default=32)
    parser.add_argument("--adjudicate-draws-from", type=int, default=None,
                        help="stop games as draws once no five is possible, from this ply")
    parser.add_argument("--draw-ply-cap", type=int, default=None,
                        help="stop unfinished games as draws at this ply")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = SelfPlayConfig(checkpoint=str(args.checkpoint), simulations=args.simulations,
                            balance_threshold=args.balance_threshold,
                            balance_attempts=args.balance_attempts,
                            adjudicate_draws_from=args.adjudicate_draws_from,
                            draw_ply_cap=args.draw_ply_cap)
    seeds = list(range(args.seed, args.seed + args.games))
    started = time.perf_counter()
    games = generate_selfplay_games(seeds, config, workers=args.workers)
    seconds = time.perf_counter() - started
    records = tuple(game_record(game, config) for game in games)
    for record in records:
        verify_record(record)
    dataset = SupervisedDataset(**teacher_games_to_arrays(games, config))
    winners = [game.winner for game in games]
    summary = {
        "black_wins": winners.count(0), "white_wins": winners.count(1),
        "draws": winners.count(None),
        "dead_draws": sum(game.adjudicated == "dead" for game in games),
        "capped_draws": sum(game.adjudicated == "ply_cap" for game in games),
        "mean_length": float(np.mean([len(game.moves) for game in games])),
        "seconds": seconds, "workers": args.workers,
        "simulations": sum(game.elapsed_nodes for game in games),
    }
    save_dataset(args.out, dataset, records, metadata={
        "selfplay": {**config.__dict__, "name": config.name},
        "seeds": [seeds[0], seeds[-1]], "summary": summary,
        "platform": platform.platform(),
    })
    print(f"saved {len(dataset)} positions from {len(games)} games to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
