"""Export a checkpoint for the browser bot, plus Python-vs-TypeScript parity fixtures.

Writes ``model.json`` (architecture and tensor layout) and ``weights.bin``
(little-endian float32) for ``web/lib/engine/network.ts``. The fixtures record,
for positions from recorded games, what the Python implementation computes:
line levels, tactical move generation, VCF, network outputs, and fixed-size
guided MCTS. ``web/tests/engine.test.mjs`` replays them in TypeScript.
"""

import argparse
import json
import random
from pathlib import Path

import jax
import numpy as np

from connectfive.guided_search import GuidedMCTS, NetworkEvaluator
from connectfive.network import encode_board, load_checkpoint
from connectfive.pattern_search import PatternSearch
from connectfive.patterns import ACTION_TO_INDEX, INDEX_TO_ACTION, PatternBoard, _level

KIND_NAMES = ("QUIET", "WIN", "LOSS", "FORCE_WIN", "BLOCK", "DEFEND")


def export_weights(checkpoint: Path, out_dir: Path, name: str) -> None:
    loaded = load_checkpoint(checkpoint, jax.random.PRNGKey(0))
    tensors, chunks, offset = [], [], 0
    flat = jax.tree_util.tree_flatten_with_path(loaded.params["params"])[0]
    for path, leaf in flat:
        key = "/".join(str(part.key) for part in path)
        array = np.asarray(leaf, dtype="<f4").reshape(-1)
        tensors.append({
            "name": key, "shape": list(leaf.shape), "offset": offset, "length": int(array.size),
        })
        chunks.append(array)
        offset += array.size
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(np.concatenate(chunks).tobytes())
    manifest = {
        "format": "connectfive-web-model",
        "version": 1,
        "name": name,
        "source_checkpoint": str(checkpoint),
        "config": {
            "residual_blocks": loaded.config.residual_blocks,
            "channels": loaded.config.channels,
            "value_hidden": loaded.config.value_hidden,
            "input_planes": loaded.config.input_planes,
        },
        "tensors": tensors,
    }
    (out_dir / "model.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {offset} parameters to {out_dir}")


def export_fixtures(checkpoint: Path, games_json: Path, out: Path, positions: int,
                    simulations: int, seed: int) -> None:
    rng = random.Random(seed)
    games = json.loads(games_json.read_text())["games"]
    evaluator = NetworkEvaluator.from_checkpoint(checkpoint)
    planes = evaluator.config.input_planes

    windows = []
    for _ in range(300):
        window = [rng.choice((0, 0, 1, 1, 2)) for _ in range(11)]
        window[5] = 1
        windows.append({"window": window, "level": _level(tuple(window))})

    cases = []
    for game in rng.sample(games, positions):
        moves = game["moves"]
        ply = rng.randrange(1, len(moves))
        board = PatternBoard()
        for action in moves[:ply]:
            board.play(ACTION_TO_INDEX[action])
        array = board.to_array()
        tactics = PatternSearch(board, node_budget=10**9, leaf_vcf_depth=0)
        kind, generated = tactics.generate(10**6)
        vcf = tactics.vcf(4) if kind == 0 else None
        features = encode_board(array, board.player, planes)
        logits, value = evaluator(board)
        search = GuidedMCTS(board, evaluator, time_limit=1e9, max_simulations=simulations)
        result = search.run()
        batched = GuidedMCTS(board, evaluator, time_limit=1e9, max_simulations=simulations,
                             batch_size=4).run()
        summaries = {
            str(action): [list(board.levels[player][index]) for player in (1, 2)]
            for action, index in enumerate(ACTION_TO_INDEX)
            if index in board.candidates
        }
        cases.append({
            "moves": moves[:ply],
            "player": board.player,
            "kind": KIND_NAMES[kind],
            "generated": sorted(INDEX_TO_ACTION[i] for i in generated)
            if kind == 4 else [INDEX_TO_ACTION[i] for i in generated],
            "vcf": None if vcf is None else INDEX_TO_ACTION[vcf],
            "levels": summaries,
            "features_checksum": float(features.sum()),
            "logits": [round(float(x), 5) for x in logits],
            "value": round(float(value), 6),
            "mcts_actions": list(result.actions),
            "mcts_visits": [list(pair) for pair in result.root_scores],
            "mcts_reason": result.stats.reason,
            "mcts_batch4_actions": list(batched.actions),
            "mcts_batch4_visits": [list(pair) for pair in batched.root_scores],
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(checkpoint), "simulations": simulations, "seed": seed,
        "windows": windows, "cases": cases,
    }) + "\n")
    print(f"wrote {len(cases)} parity cases to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name", default="az-r2")
    parser.add_argument("--out", type=Path, default=Path("web/public/models/az-r2"))
    parser.add_argument("--games", type=Path, default=Path("runs/az-r2/games.json"))
    parser.add_argument("--fixtures", type=Path, default=Path("web/tests/fixtures/engine-parity.json"))
    parser.add_argument("--positions", type=int, default=30)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    export_weights(args.checkpoint, args.out, args.name)
    export_fixtures(args.checkpoint, args.games, args.fixtures, args.positions,
                    args.simulations, args.seed)


if __name__ == "__main__":
    main()
