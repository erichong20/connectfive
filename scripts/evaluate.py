"""Run a reproducible, paired-color match between two built-in bots."""

import argparse
import json
from pathlib import Path

from connectfive.agents import make_agent
from connectfive.match import evaluate_agents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", choices=("random", "tactical"))
    parser.add_argument("opponent", choices=("random", "tactical"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="optionally save the summary as JSON")
    args = parser.parse_args()

    summary = evaluate_agents(
        make_agent(args.agent), make_agent(args.opponent), games=args.games, seed=args.seed
    )
    data = summary.as_dict()
    print(
        f"{summary.agent} vs {summary.opponent}: "
        f"{summary.wins}W {summary.losses}L {summary.draws}D "
        f"({summary.win_rate:.1%}) over {summary.games} games; "
        f"{summary.average_game_length:.1f} moves/game; "
        f"{summary.elapsed_seconds:.1f}s"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
