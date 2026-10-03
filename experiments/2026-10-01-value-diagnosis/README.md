# Value-head diagnosis after AlphaZero-lite round 4

## Question

Round 4's network had held-out value MAE 0.605 against a constant baseline
of 0.609. Is the value head broken (bad labels or targets), or is the
headline metric misleading?

## Method

- `scripts/diagnose_value.py`: on the held-out 10% of round-4 games (same
  split as training: seed 0, 50 games, 3,368 positions) it reports MAE and
  decisive-game sign accuracy per checkpoint and for the stored MCTS root
  values, broken down by plies-to-end, plus a calibration table.
- Checkpoints: teacher-v2, az-r2, az-r3, az-r4.
- Target ablation: retrain round 4 exactly as `runs/az-r4/run.sh` (from
  az-r3, 2,000 x 256, seed 0) with `--value-target outcome` and
  `--value-target search` instead of `blend`. Each takes about 5 minutes of
  CPU. Base commit `4e90fac`; Apple M1 Pro CPU, $0.

## Results

| Value source (held-out r4) | MAE | Decisive sign accuracy |
| --- | --- | --- |
| Constant | 0.609 | - |
| Colour-only (Black/White to move mean) | 0.616 | 52.2% |
| teacher-v2 network | 0.655 | 55.8% |
| az-r2 network | 0.623 | 56.9% |
| az-r3 network | 0.613 | 58.1% |
| az-r4 network (blend target) | 0.605 | 57.9% |
| r4 retrained, outcome target | 0.597 | 57.8% |
| r4 retrained, search target | 0.623 | 58.1% |
| MCTS root value (600 sims) | 0.523 | 64.9% |

Sign accuracy by game ply (decisive games only, az-r4 network vs MCTS root):
ply 4-12 59% vs 61%, 12-20 57% vs 63%, 20-30 61% vs 66%, 30-45 64% vs 72%,
45+ 52% vs 62%.

Draw share (full-board draws labelled 0 at every position):

| Round | Draw games | Draw positions |
| --- | --- | --- |
| r2 (random openings) | 7 / 500 | 10.1% |
| r3 (balanced) | 22 / 500 | 21.7% |
| r4 (balanced) | 27 / 500 | 25.1% |

## Findings

1. **No label bug and no target effect.** Outcome, search and blend targets
   all give 57.8-58.1% decisive sign accuracy. Changing the target will not
   fix the value head.
2. **The headline MAE comparison is misleading.** Balanced openings produce
   long board-filling draws: 5% of games but 25% of positions, all labelled
   0. That mass favours any prediction near 0, so "MAE vs constant" says
   little. Decisive sign accuracy by ply is the better metric, and it has
   improved slowly each round (55.8% teacher-v2 to 58% az-r3/r4).
3. **The network lags its own search.** MCTS root values beat the network by
   2-10 points of sign accuracy at every stage of the game. The gap is a
   network-capacity or data problem, not a labelling problem.
4. **Plies-to-end buckets are selection-biased.** Sign accuracy below 50% for
   positions 40+ plies from the end comes from conditioning on the future:
   only long games have such positions, and White wins most long games, while
   the network (correctly, on average) favours Black early. Use game ply, not
   plies-to-end, for future breakdowns.
5. **Draws waste self-play compute.** Positions cost roughly equal search, so
   about a quarter of round-4 self-play went to games that fill the board.

## Limitations

One held-out split of 50 games; the per-ply buckets hold 330-580 decisive
positions each, so differences under about 5 points are noise. The target
ablation is a single training seed and was not gated in play.

## Next decision

- Adjudicate draws in self-play (stop at a ply cap or when the network and
  search agree the position is dead even) and downweight draw positions in
  training; measure the compute saved and the draw share.
- Track decisive sign accuracy by ply as the value metric in reports.
- Because the network lags search at every stage, the next lever is model
  size (Stage 6 plan: 64 channels, 4-6 blocks) with more games per round,
  gated against az-r2 as before.
