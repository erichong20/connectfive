"""Compare network-guided searches with pattern search at equal time per move."""

import argparse
import json
from pathlib import Path

from connectfive.agents import make_agent
from connectfive.guided_search import GuidedAgent, NetworkEvaluator
from connectfive.match import evaluate_agents
from connectfive.pattern_search import PatternSearchAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=("mcts", "alphabeta", "alphabeta-value"),
                        default="mcts")
    parser.add_argument("--opponent", default="pattern")
    parser.add_argument("--time-limit", type=float, default=0.2)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--simulations", type=int,
        help="fixed MCTS simulations per move for the focal agent (overrides time)",
    )
    parser.add_argument(
        "--opponent-checkpoint", type=Path,
        help="play guided search with this network instead of --opponent",
    )
    args = parser.parse_args()

    evaluator = NetworkEvaluator.from_checkpoint(args.checkpoint)
    agent = GuidedAgent(
        evaluator, args.mode, args.time_limit, name=f"guided-{args.mode}",
        simulations=args.simulations,
    )
    if args.opponent_checkpoint:
        args.opponent = f"guided-{args.mode}:{args.opponent_checkpoint.parent.name}"
        opponent = GuidedAgent(
            NetworkEvaluator.from_checkpoint(args.opponent_checkpoint), args.mode,
            args.time_limit, name=args.opponent,
        )
    elif args.opponent == "pattern":
        # Same wall-clock budget for the classical baseline.
        opponent = PatternSearchAgent(node_budget=10**9, max_depth=12,
                                      time_limit=args.time_limit)
    else:
        opponent = make_agent(args.opponent)
    summary = evaluate_agents(agent, opponent, args.games, args.seed, args.opening_plies)
    low, high = summary.score_rate_95_ci
    print(
        f"{agent.name} vs {args.opponent} @ {args.time_limit}s: {summary.wins}W "
        f"{summary.losses}L {summary.draws}D; score {summary.score_rate:.1%} "
        f"({low:.0%}-{high:.0%}); {summary.average_agent_move_ms:.0f} ms/move"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary.as_dict(), indent=2) + "\n")


if __name__ == "__main__":
    main()
