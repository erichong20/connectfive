"""Explain one negamax decision on a named tactical fixture."""

import argparse

import jax

from connectfive.env import BOARD_SIZE
from connectfive.search import MATE_SCORE, NegamaxAgent
from connectfive.tactical_suite import load_tactical_suite, position_state

COLUMNS = "ABCDEFGHJKLMNOP"


def coordinate(action: int) -> str:
    row, col = divmod(action, BOARD_SIZE)
    return f"{COLUMNS[col]}{row + 1}"


def display_score(score: int) -> str:
    if score >= MATE_SCORE - 10:
        return "forced win"
    if score <= -MATE_SCORE + 10:
        return "forced loss"
    return str(score)


def main() -> None:
    positions = {position.name: position for position in load_tactical_suite()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("position", choices=tuple(positions))
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    position = positions[args.position]
    agent = NegamaxAgent(
        max_depth=args.depth,
        candidate_width=args.width,
        node_budget=args.nodes,
    )
    selected = agent.select_action(
        position_state(position), jax.random.PRNGKey(args.seed)
    )
    assert agent.last_search is not None
    result = agent.last_search

    print(f"{position.name}: {position.description}")
    print(f"selected {coordinate(selected)} ({display_score(result.score)})")
    print("\nroot candidates at the deepest completed iteration:")
    for action, score in sorted(result.root_scores, key=lambda item: item[1], reverse=True):
        marker = "*" if action == selected else " "
        print(f"{marker} {coordinate(action):>3}  {display_score(score)}")
    stats = result.stats
    print(
        f"\ndepth {stats.completed_depth}; {stats.nodes}/{args.nodes} nodes; "
        f"{stats.cutoffs} alpha-beta cutoffs; "
        f"{stats.transposition_hits} cache hits; {stats.elapsed_ms:.1f} ms"
    )


if __name__ == "__main__":
    main()
