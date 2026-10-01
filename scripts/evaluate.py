"""Run a reproducible, paired-color match between two built-in bots."""

import argparse
import json
from pathlib import Path

from connectfive.agents import make_agent
from connectfive.match import evaluate_agents, evaluate_agents_with_records
from connectfive.search import NegamaxAgent, ThreatSearchAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = ("random", "tactical", "lookahead", "negamax", "threatsearch", "pattern")
    parser.add_argument("agent", choices=choices)
    parser.add_argument("opponent", choices=choices)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=0,
        help="use one reproducible local opening per paired-color game",
    )
    parser.add_argument("--node-budget", type=int, default=1_500)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--candidate-width", type=int, default=10)
    parser.add_argument("--json", type=Path, help="optionally save the summary as JSON")
    parser.add_argument(
        "--failures-json",
        type=Path,
        help="save replayable records for every focal-agent loss",
    )
    args = parser.parse_args()

    agent = make_agent(args.agent)
    if isinstance(agent, (NegamaxAgent, ThreatSearchAgent)):
        agent.node_budget = args.node_budget
        agent.max_depth = args.search_depth
        agent.candidate_width = args.candidate_width
    opponent = make_agent(args.opponent)
    records = ()
    if args.failures_json:
        summary, records = evaluate_agents_with_records(
            agent,
            opponent,
            games=args.games,
            seed=args.seed,
            opening_plies=args.opening_plies,
        )
    else:
        summary = evaluate_agents(
            agent,
            opponent,
            games=args.games,
            seed=args.seed,
            opening_plies=args.opening_plies,
        )
    data = summary.as_dict()
    print(
        f"{summary.agent} vs {summary.opponent}: "
        f"{summary.wins}W {summary.losses}L {summary.draws}D "
        f"({summary.win_rate:.1%}) over {summary.games} games; "
        f"{summary.average_game_length:.1f} moves/game; "
        f"{summary.elapsed_seconds:.1f}s"
    )
    low, high = summary.score_rate_95_ci
    print(
        f"score rate {summary.score_rate:.1%} "
        f"(approx. 95% interval {low:.1%}–{high:.1%}); "
        f"as Black {summary.agent_black_wins}W-{summary.agent_black_losses}L-"
        f"{summary.agent_black_draws}D; as White {summary.agent_white_wins}W-"
        f"{summary.agent_white_losses}L-{summary.agent_white_draws}D"
    )
    print(
        f"agent moves: {summary.average_agent_move_ms:.1f} ms average; "
        f"{summary.average_search_nodes:.1f} search nodes; "
        f"depth {summary.average_completed_depth:.2f}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n")
    if args.failures_json:
        failures = [
            record.as_dict()
            for game_index, record in enumerate(records)
            if record.winner is not None and record.winner != game_index % 2
        ]
        payload = {"summary": data, "failures": failures}
        args.failures_json.parent.mkdir(parents=True, exist_ok=True)
        args.failures_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
