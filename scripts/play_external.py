"""Play our guided-MCTS bot against an external Gomocup (Piskvork) engine.

Games use the project's rules (15x15, exactly five; overlines do not win) and
are refereed by our own pattern board. Openings are the usual seeded 4-ply
openings, each played twice with colours reversed. An illegal move, crash or
timeout by either side loses that game and is recorded.

Example (Rapfi limited to 4,000 nodes per move, one thread):

    ./.venv/bin/python scripts/play_external.py runs/az-r7/model \\
        --engine runs/engines/rapfi/build/pbrain-rapfi \\
        --info MAX_NODE=4000 --info THREAD_NUM=1 --games 40 --seed 15000 \\
        --time-limit 1.0 --json runs/external/r7-vs-rapfi-n4000.json
"""

import argparse
import json
import math
import time
from pathlib import Path

from connectfive.guided_search import GuidedMCTS, NetworkEvaluator, VcfCache
from connectfive.match import generate_opening
from connectfive.patterns import ACTION_TO_INDEX, PatternBoard
from connectfive.piskvork import EngineError, PiskvorkEngine

SIZE = 15


def wilson(score: float, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    denominator = 1 + z * z / games
    center = (score + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games)) / denominator
    return center - margin, center + margin


def play_game(evaluator, cache, engine, opening, ours_black, args) -> dict:
    board = PatternBoard()
    history: list[int] = []
    our_ms, their_ms = [], []
    for action in opening:
        board.play(ACTION_TO_INDEX[action])
        history.append(action)
    ours = 0 if ours_black else 1
    while len(history) < SIZE * SIZE:
        mover = len(history) % 2
        if mover == ours:
            started = time.perf_counter()
            search = GuidedMCTS(
                board, evaluator, time_limit=args.time_limit if not args.simulations else 1e9,
                max_simulations=args.simulations or 100_000, vcf_cache=cache,
            )
            action = int(search.run().actions[0])
            our_ms.append(1000 * (time.perf_counter() - started))
        else:
            try:
                action, seconds = engine.move(history, timeout=args.engine_timeout)
            except EngineError as error:
                return {"winner": ours, "reason": f"external: {error}", "moves": history,
                        "our_ms": our_ms, "their_ms": their_ms}
            their_ms.append(1000 * seconds)
            if action < 0 or board.cells[ACTION_TO_INDEX[action]] != 0:
                return {"winner": ours, "reason": "external illegal move", "moves": history,
                        "our_ms": our_ms, "their_ms": their_ms}
        index = ACTION_TO_INDEX[action]
        won = board.summary[board.turn][index][3]  # exact five only
        board.play(index)
        history.append(action)
        if won:
            return {"winner": mover, "reason": "five", "moves": history,
                    "our_ms": our_ms, "their_ms": their_ms}
    return {"winner": None, "reason": "full board", "moves": history,
            "our_ms": our_ms, "their_ms": their_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-name", default=None)
    parser.add_argument("--info", action="append", default=[],
                        help="KEY=VALUE sent as 'INFO KEY VALUE' each game (repeatable)")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=1.0, help="our seconds per move")
    parser.add_argument("--simulations", type=int, help="fixed simulations instead of time")
    parser.add_argument("--engine-timeout", type=float, default=30.0)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.games % 2:
        parser.error("--games must be even (each opening is played with both colours)")

    info = dict(item.split("=", 1) for item in args.info)
    info.setdefault("rule", 1)  # Gomocup rule 1: exactly five on 15x15
    engine = PiskvorkEngine([str(args.engine.resolve())], cwd=args.engine.parent, info=info,
                            name=args.engine_name or args.engine.name)
    evaluator = NetworkEvaluator.from_checkpoint(args.checkpoint)
    cache = VcfCache()
    engine.start()
    games = []
    try:
        for game_index in range(args.games):
            seed = args.seed + game_index // 2
            ours_black = game_index % 2 == 0
            if game_index:
                try:
                    engine.restart()
                except EngineError:
                    engine.close()
                    engine.start()
            record = play_game(evaluator, cache, engine, generate_opening(seed, args.opening_plies),
                               ours_black, args)
            record.update(seed=seed, ours_black=ours_black)
            games.append(record)
            ours = 0 if ours_black else 1
            result = "draw" if record["winner"] is None else ("win" if record["winner"] == ours else "loss")
            print(f"game {game_index + 1}/{args.games} seed {seed} ours {'B' if ours_black else 'W'}: "
                  f"{result} ({record['reason']}, {len(record['moves'])} plies)", flush=True)
    finally:
        engine.close()

    wins = sum(g["winner"] == (0 if g["ours_black"] else 1) for g in games)
    draws = sum(g["winner"] is None for g in games)
    losses = len(games) - wins - draws
    score = (wins + draws / 2) / len(games)
    low, high = wilson(score, len(games))
    summary = {
        "checkpoint": str(args.checkpoint), "engine": str(args.engine), "info": info,
        "time_limit": args.time_limit, "simulations": args.simulations, "seed": args.seed,
        "games": len(games), "wins": wins, "losses": losses, "draws": draws,
        "score": score, "score_95_ci": [low, high],
        "wins_as_black": sum(g["winner"] == 0 and g["ours_black"] for g in games),
        "wins_as_white": sum(g["winner"] == 1 and not g["ours_black"] for g in games),
        "external_failures": sum(g["reason"].startswith("external") for g in games),
        "mean_our_ms": sum(sum(g["our_ms"]) for g in games) / max(1, sum(len(g["our_ms"]) for g in games)),
        "mean_their_ms": sum(sum(g["their_ms"]) for g in games)
        / max(1, sum(len(g["their_ms"]) for g in games)),
        "records": games,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary) + "\n")
    print(f"{args.checkpoint} vs {summary['engine']} {info}: {wins}-{losses}-{draws}, "
          f"{score:.1%} (95% CI {low:.1%}-{high:.1%}); external failures "
          f"{summary['external_failures']}; ms/move ours {summary['mean_our_ms']:.0f} "
          f"theirs {summary['mean_their_ms']:.0f}")


if __name__ == "__main__":
    main()
