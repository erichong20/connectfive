"""Run one built-in agent against the compact tactical regression suite."""

import argparse

from connectfive.agents import make_agent
from connectfive.env import BOARD_SIZE
from connectfive.tactical_suite import evaluate_tactical_suite

COLUMNS = "ABCDEFGHJKLMNOP"


def coordinate(action: int) -> str:
    row, col = divmod(action, BOARD_SIZE)
    return f"{COLUMNS[col]}{row + 1}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "agent",
        choices=("random", "tactical", "lookahead", "negamax", "threatsearch"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent = make_agent(args.agent)
    results = evaluate_tactical_suite(agent, seed=args.seed)
    passed = sum(result.passed for result in results)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        expected = ", ".join(map(coordinate, result.acceptable_actions))
        print(
            f"{status:4} {result.name:28} "
            f"played {coordinate(result.selected_action):>3}; expected {expected}"
        )
    print(f"\n{agent.name}: {passed}/{len(results)} positions solved")
    last_search = getattr(agent, "last_search", None)
    if last_search is not None:
        stats = last_search.stats
        print(
            f"last search: depth {stats.completed_depth}, {stats.nodes} nodes, "
            f"{stats.cutoffs} cutoffs, {stats.transposition_hits} cache hits, "
            f"{stats.elapsed_ms:.1f} ms"
        )


if __name__ == "__main__":
    main()
