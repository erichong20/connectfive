"""Benchmark Negamax node budgets on the compact tactical suite."""

import argparse
import statistics

import jax

from connectfive.search import NegamaxAgent, ThreatSearchAgent
from connectfive.tactical_suite import load_tactical_suite, position_state


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[250, 500, 1_000, 1_500])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--agent", choices=("negamax", "threatsearch"), default="negamax")
    args = parser.parse_args()
    suite = load_tactical_suite()

    print("budget  solved  mean-ms  p95-ms  mean-nodes  mean-depth")
    for budget in args.budgets:
        agent_class = ThreatSearchAgent if args.agent == "threatsearch" else NegamaxAgent
        agent = agent_class(
            max_depth=args.depth,
            candidate_width=args.width,
            node_budget=budget,
        )
        elapsed = []
        nodes = []
        depths = []
        solved = 0
        for index, position in enumerate(suite):
            action = agent.select_action(
                position_state(position), jax.random.PRNGKey(index)
            )
            solved += action in position.acceptable_actions
            assert agent.last_search is not None
            elapsed.append(agent.last_search.stats.elapsed_ms)
            nodes.append(agent.last_search.stats.nodes)
            depths.append(agent.last_search.stats.completed_depth)
        print(
            f"{budget:>6}  {solved:>2}/{len(suite):<2}  "
            f"{statistics.mean(elapsed):>7.1f}  {percentile_95(elapsed):>6.1f}  "
            f"{statistics.mean(nodes):>10.1f}  {statistics.mean(depths):>10.2f}"
        )


if __name__ == "__main__":
    main()
