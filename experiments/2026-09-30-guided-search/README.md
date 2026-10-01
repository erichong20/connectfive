# Network-guided search (v2 network inside search)

## Hypothesis

The v2 policy/value network (raw policy 77.5% vs Tactical) can make search
stronger than the classical `pattern` search under the same time per move.

## Variants (`src/connectfive/guided_search.py`)

- `alphabeta`: pattern alpha-beta; quiet moves at plies 0-1 are ranked by
  network policy instead of pattern scores.
- `alphabeta-value`: same, plus quiet leaves scored by the value head,
  mapped with `6000 * atanh(v)`.
- `mcts`: PUCT (c = 1.5, first-play urgency -0.2, top 16 policy moves per quiet
  node). The pattern board decides wins, forced blocks, open-four wins,
  open-three defences, and depth-4 VCF proofs exactly; the network supplies
  priors and leaf values. The move played is the most-visited root child.

Network calls cost about 0.57 ms each (batch 1, M1 Pro CPU) and are cached by
Zobrist hash.

## Protocol

- Opponent: `pattern` with node budget unlimited, depth 12, 0.2 s per move.
- Guided agents: 0.2 s per move, checkpoint `runs/teacher-v2/model`.
- Four-ply random openings, paired colours; seeds below.
- Some matches ran in parallel processes, which affects both sides equally.
- Code: base commit `9770392` plus this change. CPU only, $0.

## Results

| Variant | Games (seeds) | W-L-D | Score | 95% interval | ms/move |
| --- | --- | --- | --- | --- | --- |
| alphabeta | 20 (3000) | 7-13-0 | 35% | 18-57% | 141 |
| alphabeta-value | 20 (3000) | 11-9-0 | 55% | 34-74% | 175 |
| mcts | 20 (3000) | 13-7-0 | 65% | 43-82% | 135 |
| **mcts** | **100 (4000-4049)** | **63-37-0** | **63%** | **53-72%** | ~143 |

All variants solve the 8-position tactical suite. A test checks that they
still do with an untrained network.

## Interpretation

Guided MCTS is stronger than pattern alpha-beta at equal time: the 100-game
interval excludes 50%. It spends fewer milliseconds per move than its budget
because forced positions return immediately. Using the policy only to order
alpha-beta moves did not help. Narrowing to the network's top moves near the
root likely prunes moves that the hand-written ordering keeps. The value head
helped alpha-beta somewhat, but the 20-game sample cannot separate it from
`pattern`.

## Limitations

- Only one time control (0.2 s) was tested, and the PUCT constants were not tuned.
- MCTS runs one simulation per network call, with no batching. Batched leaf
  evaluation could raise throughput about 3x (32 positions take 2.7 ms).
- The value head carries the colour bias of its teacher data.

## Decision

Promote guided MCTS (v2 network) as the strongest agent. Next: produce
MCTS-visit policy targets and self-play data from it (AlphaZero-lite, roadmap
stage 6), and try batched leaf evaluation.

## Reproduce

```bash
./.venv/bin/python scripts/evaluate_guided.py runs/teacher-v2/model --mode mcts --games 20 --seed 4000 --time-limit 0.2
```
