"""Generate a reproducible supervised corpus from classical agents."""

import argparse
import platform
from pathlib import Path

from connectfive.agents import TacticalAgent
from connectfive.dataset import records_to_dataset, save_dataset
from connectfive.match import evaluate_agents_with_records
from connectfive.search import NegamaxAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--node-budget", type=int, default=100)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--candidate-width", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("runs/supervised-v0/games.npz"))
    args = parser.parse_args()

    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")
    negamax = NegamaxAgent(
        node_budget=args.node_budget,
        max_depth=args.search_depth,
        candidate_width=args.candidate_width,
    )
    tactical = TacticalAgent()
    summary, records = evaluate_agents_with_records(
        negamax,
        tactical,
        games=args.games,
        seed=args.seed,
        opening_plies=args.opening_plies,
    )
    dataset = records_to_dataset(records)
    save_dataset(
        args.out,
        dataset,
        records,
        metadata={
            "seed": args.seed,
            "opening_plies": args.opening_plies,
            "agents": {
                "negamax": {
                    "node_budget": args.node_budget,
                    "max_depth": args.search_depth,
                    "candidate_width": args.candidate_width,
                },
                "tactical": {},
            },
            "match_summary": summary.as_dict(),
            "platform": platform.platform(),
        },
    )
    print(
        f"saved {len(dataset)} examples from {len(records)} complete games to {args.out}"
    )
    print(
        f"negamax vs tactical: {summary.wins}W {summary.losses}L {summary.draws}D; "
        f"{summary.elapsed_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
