# AlphaZero-lite round 9: Rapfi-labelled candidate + native self-play (promoted)

## Hypothesis

The Rapfi-DAgger candidate (`runs/rapfi-teacher/model-d1`, see
`experiments/2026-10-05-rapfi-teacher-pilot/` and the DAgger section below)
was stronger on the Rapfi ladder but failed its confirmation gate against
az-r7 (60.0% then 53.8%). One self-play round from it, generated with the
faster native core, should consolidate the gain.

## DAgger step (model-d1)

- `scripts/label_with_rapfi.py`: Rapfi (1 thread, 10,000 nodes) labelled
  every post-opening position of rounds 8 and 7 (az-r7 / az-r6 self-play):
  115,543 + 111,709 positions; Rapfi's move as a one-hot policy target,
  `tanh(eval / 300)` as search value, the real game result as outcome.
  Rapfi agreed with the played move 55% of the time (6-game sample).
- Fine-tune az-r7 3,000 x 256 steps, lr 0.0002, on Rapfi-labelled r8 (main,
  10% held out) plus Rapfi-labelled r7 and self-play r8, r7, r6 replay with
  `--hold-out-extra`, soft policy, blend value, per-game value weighting.
- Gate vs az-r7 (seeds 17100-17199): 118-78-4, 60.0% (53.1-66.5%), pass;
  confirmation (seeds 18000-18099): 105-90-5, 53.8% (46.8-60.5%), fail. Not
  promoted. Rapfi ladder (41 balanced openings x 2, 1 s/move): 81.1% / 65.9%
  / 52.4% / 22.0% against 30 / 100 / 300 / 1,000 nodes (az-r7: 79.3 / 54.9 /
  37.8 / 11.6).

## Round 9

- Self-play from model-d1 with the native core (`--native-batch 8`): 4,000
  games, seeds 150000-153999, 600 simulations on full searches, 25% playout
  cap / 100 fast simulations, balanced openings, draw cap 150. 33,439
  recorded positions, Black-White-Draw 2472-1469-59, 4,185 s on 9 workers
  (about 3,440 games/hour; Python rounds 7-8 managed about 2,450). Verifying
  all 4,000 records through the JAX environment then took about 45 minutes;
  self-play now verifies a 200-game sample by default.
- Fine-tune model-d1 3,000 x 256 steps, lr 0.0003, on round 9 (main) plus the
  Rapfi labels of r8/r7 and self-play r8/r7/r6 (`--hold-out-extra`); 1,214 s.
  Held-out round-9 policy accuracy 61.1%, decisive sign accuracy 71.1%.
- Base commit `07208c3` (clean apart from `.claude/`). Script: `run.sh`.

## Results

| Gate vs az-r7 (guided MCTS, 0.2 s/move, 200 paired games) | W-L-D | Score | 95% CI |
| --- | --- | --- | --- |
| Seeds 19000-19099 | 115-81-4 | 58.5% | 51.6-65.1% |
| Confirmation, seeds 19100-19199 | 116-79-5 | 59.2% | 52.3-65.8% |

Both gates pass on their own, so **az-r9 is the new champion**. Wins as
Black / White: 89 / 26 and 85 / 31.

## Interpretation

- Teacher labels on our own positions plus one self-play round gave a
  confirmed gain over az-r7 (about +65 Elo by the gate score), where training
  on Rapfi's own games had made the network worse.
- Gates use the Python search for both sides, so the native core only
  changed how fast the round ran, not how the gate was played.

## Rapfi ladder

41 balanced openings x 2 colours, our side 1 s/move, Rapfi 1 thread
(`ladder-n*.json`; same protocol as `experiments/2026-10-04-rapfi-anchor/`):

| Rapfi | az-r7 | model-d1 | az-r9 | az-r9 95% CI |
| --- | --- | --- | --- | --- |
| 30 nodes | 79.3% | 81.1% | 87.2% | 78-93% |
| 100 nodes | 54.9% | 65.9% | 64.6% | 54-74% |
| 300 nodes | 37.8% | 52.4% | 52.4% | 42-63% |
| 1,000 nodes | 11.6% | 22.0% | 24.4% | 16-35% |

az-r9 is about even with Rapfi at 300 nodes per move, where az-r7 was even
at roughly 100-150: on the order of +100-150 Elo on this external scale,
consistent with the head-to-head gates. Every Rapfi setting moved in our
favour, but adjacent rows overlap, so the per-row differences are not
individually significant. On the rough Gomocup mapping of the anchor
experiment, that moves the estimate from about 1750-2050 to about
1900-2200.

## Next decision

- Export az-r9 to the website after its browser parity tests.
- Continue: Rapfi-label the new round's positions, fine-tune, native
  self-play, gate against az-r9.
