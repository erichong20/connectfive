"""Break down a network's value predictions on held-out self-play games.

Reports, per checkpoint, value error against the real outcome by game phase
(plies before the end), a calibration table, and the same numbers for the
stored root search values, so a weak value head can be traced to its labels.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from connectfive.dataset import load_dataset, split_by_game
from connectfive.network import PolicyValueNetwork, load_checkpoint, with_input_planes

PHASES = ((0, 4), (4, 10), (10, 20), (20, 40), (40, 999))


def predict(path: Path, dataset, indices, chunk: int = 2_048) -> np.ndarray:
    loaded = load_checkpoint(path, jax.random.PRNGKey(0))
    model = PolicyValueNetwork(loaded.config)
    apply = jax.jit(lambda x: model.apply(loaded.params, x)[1])
    parts = []
    for start in range(0, len(indices), chunk):
        part = indices[start:start + chunk]
        features = with_input_planes(dataset.features[part], loaded.config.input_planes)
        parts.append(np.asarray(apply(jnp.asarray(features))))
    return np.concatenate(parts)


def summarize(values: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    decisive = outcome != 0
    return {
        "n": int(len(values)),
        "mae": float(np.abs(values - outcome).mean()),
        "sign_acc": float((np.sign(values[decisive]) == outcome[decisive]).mean())
        if decisive.any() else float("nan"),
        "mean_pred": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    _, held_out = split_by_game(dataset, args.validation_fraction, args.seed)
    outcome = dataset.values[held_out]
    plies = dataset.plies[held_out]
    game_ids = dataset.game_ids[held_out]
    last_ply = {g: dataset.plies[dataset.game_ids == g].max() for g in np.unique(game_ids)}
    to_end = np.array([last_ply[g] - p for g, p in zip(game_ids, plies)])
    # Mover-relative labels: Black-to-move positions carry Black's result.
    black_to_move = dataset.players[held_out] == 0

    sources = {"search_value": dataset.search_values[held_out]}
    for path in args.checkpoints:
        sources[str(path)] = predict(path, dataset, held_out)

    report = {
        "dataset": str(args.dataset), "held_out_positions": int(len(held_out)),
        "held_out_games": int(len(np.unique(game_ids))),
        "constant_mae": float(np.abs(outcome - outcome.mean()).mean()),
        "colour_baseline": {},
        "sources": {},
    }
    colour = np.where(black_to_move, outcome[black_to_move].mean(), outcome[~black_to_move].mean())
    report["colour_baseline"] = summarize(colour, outcome)
    for name, values in sources.items():
        entry = {"all": summarize(values, outcome), "plies_to_end": {}, "calibration": []}
        for low, high in PHASES:
            mask = (to_end >= low) & (to_end < high)
            if mask.any():
                entry["plies_to_end"][f"{low}-{high}"] = summarize(values[mask], outcome[mask])
        for low in np.arange(-1.0, 1.0, 0.25):
            mask = (values >= low) & (values < low + 0.25 + (low == 0.75))
            if mask.any():
                entry["calibration"].append({
                    "bin": f"{low:+.2f}", "n": int(mask.sum()),
                    "mean_pred": float(values[mask].mean()),
                    "mean_outcome": float(outcome[mask].mean()),
                })
        report["sources"][name] = entry

    print(f"{args.dataset}: {report['held_out_games']} held-out games, "
          f"{report['held_out_positions']} positions; constant MAE "
          f"{report['constant_mae']:.3f}; colour baseline {report['colour_baseline']}")
    for name, entry in report["sources"].items():
        print(f"\n{name}: {entry['all']}")
        for phase, stats in entry["plies_to_end"].items():
            print(f"  to-end {phase:>7}: mae {stats['mae']:.3f} sign {stats['sign_acc']:.3f} n {stats['n']}")
        print("  calibration: " + "  ".join(
            f"{c['bin']}:{c['mean_outcome']:+.2f}({c['n']})" for c in entry["calibration"]
        ))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
