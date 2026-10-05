"""Relabel our own games with Rapfi (DAgger-style teacher data).

Every post-opening position of every game in a dataset manifest (the
``.json`` written next to a self-play ``.npz``) is shown to Rapfi at a fixed
node budget. Each position gets Rapfi's move as a one-hot policy target and
its evaluation for the side to move as the search value
(``tanh(eval / --eval-scale)``); the value outcome stays the game's real
result. Games keep the manifest's order, so game ids, and therefore
``split_by_game`` held-out sets, match the source dataset.

Labelled games are appended to ``<out>.games.jsonl`` and reused on restart.
"""

import argparse
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from connectfive.dataset import SupervisedDataset, save_dataset
from connectfive.match import GameRecord
from connectfive.piskvork import EngineError, PiskvorkEngine
from connectfive.selfplay import game_to_json, load_game_log
from connectfive.teacher import LabelledPosition, TeacherGame, teacher_games_to_arrays


@dataclass(frozen=True)
class RapfiLabelConfig:
    engine: str
    nodes: int
    eval_scale: float = 300.0

    @property
    def name(self) -> str:
        return f"rapfi-label-n{self.nodes}"


_ENGINE: PiskvorkEngine | None = None


def _engine(config: RapfiLabelConfig) -> PiskvorkEngine:
    global _ENGINE
    if _ENGINE is None:
        path = Path(config.engine)
        _ENGINE = PiskvorkEngine(
            [str(path.resolve())], cwd=path.parent, name="rapfi",
            info={"rule": 1, "THREAD_NUM": 1, "MAX_NODE": config.nodes,
                  "TIMEOUT_TURN": 30_000, "TIMEOUT_MATCH": 100_000_000},
        )
        _ENGINE.start()
    return _ENGINE


def label_game(game: dict, config: RapfiLabelConfig) -> TeacherGame:
    moves, opening = list(game["moves"]), list(game["opening_moves"])
    positions, nodes = [], 0
    for ply in range(len(opening), len(moves)):
        engine = _engine(config)
        engine.restart()
        action, _ = engine.move(moves[:ply], timeout=60)
        if not 0 <= action < 225 or action in moves[:ply]:
            raise EngineError(f"rapfi played an illegal move {action} in game {game['seed']}")
        score = engine.last_eval() or 0
        nodes += config.nodes
        positions.append(LabelledPosition(
            ply=ply, player=ply % 2, best_action=action, policy=((action, 1.0),),
            search_value=math.tanh(score / config.eval_scale), search_score=score, reason="rapfi",
        ))
    return TeacherGame(game["seed"], tuple(moves), tuple(opening), game["winner"],
                       tuple(positions), nodes, game.get("adjudicated"))


def _label(args) -> TeacherGame | None:
    global _ENGINE
    for attempt in range(2):
        try:
            return label_game(*args)
        except EngineError as error:
            if _ENGINE is not None:
                _ENGINE.close()
            _ENGINE = None
            if attempt:
                print(f"skipping game {args[0]['seed']}: {error}", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="dataset manifest (.json) with a 'games' list")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--eval-scale", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = RapfiLabelConfig(str(args.engine), args.nodes, args.eval_scale)
    source = json.loads(args.manifest.read_text())["games"]
    log = args.out.with_suffix(".games.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_game_log(log)
    remaining = [game for game in source if game["seed"] not in done]
    started = time.perf_counter()
    with log.open("a") as handle, get_context("spawn").Pool(args.workers) as pool:
        for labelled in pool.imap_unordered(_label, [(game, config) for game in remaining], chunksize=4):
            if labelled is not None:
                done[labelled.seed] = labelled
                handle.write(game_to_json(labelled) + "\n")
                handle.flush()
    games = [done[game["seed"]] for game in source if game["seed"] in done]
    records = tuple(
        GameRecord(**{**game, "moves": tuple(game["moves"]), "rewards": tuple(game["rewards"]),
                      "opening_moves": tuple(game["opening_moves"]), "move_diagnostics": ()})
        for game in source if game["seed"] in done
    )
    dataset = SupervisedDataset(**teacher_games_to_arrays(games, config))
    agree = sum(
        position.best_action == game.moves[position.ply]
        for game in games for position in game.positions
    ) / max(1, len(dataset))
    summary = {
        "games": len(games), "skipped_games": len(source) - len(games), "positions": len(dataset),
        "seconds": time.perf_counter() - started, "workers": args.workers,
        "rapfi_agrees_with_played_move": agree,
    }
    save_dataset(args.out, dataset, records, metadata={
        "labeller": {**config.__dict__, "name": config.name}, "source": str(args.manifest),
        "summary": summary, "platform": platform.platform(),
    })
    print(f"saved {len(dataset)} positions from {len(games)} games to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
