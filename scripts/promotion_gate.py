"""Promotion gate: candidate vs champion in guided MCTS, run as parallel blocks.

Each block is one ``scripts/evaluate_guided.py`` process playing ``--block``
paired games (both colours per opening); block k uses opening seeds
``seed + k * block / 2`` onwards, so the whole gate uses one contiguous seed
range. The candidate is promoted only if the 95% interval of its score
excludes 50%. With ``--min-score`` the exit status instead checks a plain
score threshold (used as an early stop, not a promotion).

Exit status: 0 passed, 3 failed, 1 error.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from connectfive.match import combine_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("champion", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-limit", type=float, default=0.2)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--simulations", type=int,
                        help="fixed simulations per move for the candidate (overrides time)")
    parser.add_argument("--native-batch", type=int, default=0,
                        help="both sides search with the native C core (leaves per batch)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="candidate's leaves per batch under virtual loss")
    parser.add_argument("--opponent-simulations", type=int,
                        help="fixed simulations per move for the champion")
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.games % args.block or args.block % 2:
        sys.exit("--games must be a multiple of an even --block")

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + k * args.block // 2 for k in range(args.games // args.block)]

    def run(seed: int) -> dict:
        path = args.out / f"block-{seed}.json"
        if not path.exists():  # finished blocks are reused, so a gate can resume
            with (args.out / f"block-{seed}.log").open("w") as log:
                subprocess.run(
                    [sys.executable, "scripts/evaluate_guided.py", str(args.candidate),
                     "--opponent-checkpoint", str(args.champion), "--games", str(args.block),
                     "--seed", str(seed), "--time-limit", str(args.time_limit),
                     "--json", str(path)]
                    + (["--simulations", str(args.simulations)] if args.simulations else [])
                    + (["--batch-size", str(args.batch_size)] if args.batch_size > 1 else [])
                    + (["--native-batch", str(args.native_batch)] if args.native_batch else [])
                    + (["--opponent-simulations", str(args.opponent_simulations)]
                       if args.opponent_simulations else []),
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )
        return json.loads(path.read_text())

    with ThreadPoolExecutor(args.parallel) as pool:
        blocks = list(pool.map(run, seeds))
    summary = combine_summaries(blocks)
    if summary.agent_illegal_moves or summary.opponent_illegal_moves:
        print("illegal moves recorded; gate invalid")
        sys.exit(1)
    low, high = summary.score_rate_95_ci
    if args.min_score is None:
        passed, rule = low > 0.5, "95% interval excludes 50%"
    else:
        passed, rule = summary.score_rate >= args.min_score, f"score >= {args.min_score:.0%}"
    result = {
        "candidate": str(args.candidate), "champion": str(args.champion),
        "seeds": [seeds[0], seeds[-1] + args.block // 2 - 1],
        "time_limit": args.time_limit, "simulations": args.simulations,
        "opponent_simulations": args.opponent_simulations,
        "batch_size": args.batch_size, "native_batch": args.native_batch, "rule": rule, "passed": passed,
        **summary.as_dict(),
    }
    (args.out / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.candidate} vs {args.champion}: {summary.wins}-{summary.losses}-"
          f"{summary.draws}, {summary.score_rate:.1%} (95% CI {low:.1%}-{high:.1%}); "
          f"{rule}: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 3)


if __name__ == "__main__":
    main()
