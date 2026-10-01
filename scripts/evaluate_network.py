"""Evaluate a raw policy network against a classical baseline."""

import argparse
import json
from pathlib import Path

import jax

from connectfive.agents import make_agent
from connectfive.match import evaluate_agents
from connectfive.network import NetworkAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--opponent",
        choices=("random", "tactical", "negamax", "pattern"),
        default="random",
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    network = NetworkAgent.from_checkpoint(args.checkpoint, jax.random.PRNGKey(0))
    summary = evaluate_agents(
        network,
        make_agent(args.opponent),
        args.games,
        args.seed,
        args.opening_plies,
    )
    print(
        f"network vs {args.opponent}: {summary.wins}W {summary.losses}L "
        f"{summary.draws}D; score {summary.score_rate:.1%}; "
        f"{summary.average_agent_move_ms:.1f} ms/move"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary.as_dict(), indent=2) + "\n")


if __name__ == "__main__":
    main()
