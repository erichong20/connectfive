"""Pick seeded openings that an external engine (e.g. Rapfi) rates as balanced.

Random 4-ply openings usually favour Black so much that results depend on the
colour and opening more than on the players. Each candidate opening is shown
to the engine as a position with Black to move; openings whose reported
evaluation is within ``--max-eval`` of zero are kept, in seed order.
"""

import argparse
import json
from pathlib import Path

from connectfive.match import generate_opening
from connectfive.piskvork import EngineError, PiskvorkEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--info", action="append", default=[])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--plies", type=int, default=4)
    parser.add_argument("--max-eval", type=int, default=150)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    info = {"rule": 1, **dict(item.split("=", 1) for item in args.info)}
    engine = PiskvorkEngine([str(args.engine.resolve())], cwd=args.engine.parent, info=info)
    engine.start()
    kept, evaluations = [], []
    try:
        for seed in range(args.seed, args.seed + args.max_candidates):
            opening = list(generate_opening(seed, args.plies))
            try:
                engine.restart()
                engine.move(opening, timeout=60)
                value = engine.last_eval()
            except EngineError:
                # Skip this candidate and start a fresh engine process.
                engine.close()
                engine.start()
                value = None
            evaluations.append(value)
            if value is not None and abs(value) <= args.max_eval:
                kept.append({"seed": seed, "moves": opening, "eval": value})
                if len(kept) == args.count:
                    break
    finally:
        engine.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "engine": str(args.engine), "info": info, "plies": args.plies, "max_eval": args.max_eval,
        "candidates": len(evaluations), "openings": kept,
    }, indent=1) + "\n")
    known = [v for v in evaluations if v is not None]
    print(f"kept {len(kept)} of {len(evaluations)} candidates; "
          f"median |eval| {sorted(abs(v) for v in known)[len(known) // 2] if known else 'n/a'}")


if __name__ == "__main__":
    main()
