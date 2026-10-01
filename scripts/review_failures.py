"""Summarize replayable match failures and their final forced threats."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from connectfive.agents import _is_exact_five_after
from connectfive.env import BOARD_SIZE, EMPTY

COLUMNS = "ABCDEFGHJKLMNOP"


def coordinate(action: int) -> str:
    row, col = divmod(action, BOARD_SIZE)
    return f"{COLUMNS[col]}{row + 1}"


def forced_defenses(game: dict, agent_name: str) -> list[tuple[int, list[int], int]]:
    loser = 0 if game["black"] == agent_name else 1
    board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.int8)
    events = []
    for ply, action in enumerate(game["moves"]):
        if ply % 2 == loser:
            opponent = 1 - loser
            threats = [
                int(move)
                for move in np.flatnonzero(board.reshape(-1) == EMPTY)
                if _is_exact_five_after(board, move, opponent)
            ]
            if threats:
                events.append((ply, threats, action))
        board.reshape(-1)[action] = ply % 2
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--node-budget", type=int)
    args = parser.parse_args()
    payload = json.loads(args.path.read_text())
    failures = payload["failures"]
    agent_name = payload["summary"]["agent"]
    diagnostics = [
        diagnostic
        for game in failures
        for diagnostic in game["move_diagnostics"]
        if diagnostic["player"] == (0 if game["black"] == agent_name else 1)
    ]
    colors = Counter(
        "Black" if game["black"] == agent_name else "White" for game in failures
    )
    print(f"{len(failures)} losses; colors {dict(colors)}")
    print(f"mean length {statistics.mean(len(game['moves']) for game in failures):.1f}")
    if args.node_budget is not None:
        print(
            f"budget-hit moves "
            f"{sum(item['search_nodes'] >= args.node_budget for item in diagnostics)}/"
            f"{len(diagnostics)}"
        )
    for game in sorted(failures, key=lambda item: len(item["moves"]))[: args.examples]:
        events = forced_defenses(game, agent_name)
        last = events[-1] if events else None
        color = "Black" if game["black"] == agent_name else "White"
        print(f"\nseed {game['seed']}; {agent_name} {color}; {len(game['moves'])} moves")
        if last:
            _, threats, played = last
            print(
                "final forced-defense turn: threats "
                f"{', '.join(map(coordinate, threats))}; played {coordinate(played)}"
            )


if __name__ == "__main__":
    main()
