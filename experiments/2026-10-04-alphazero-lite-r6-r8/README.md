# AlphaZero-lite rounds 6-8: fast self-play, overnight loop, az-r7 promoted

## Hypothesis

With self-play about 9x faster (`experiments/2026-10-03-selfplay-speed/`),
several 64-channel rounds fit in one night, and consecutive rounds compound
(as rounds 1-2 did) enough to clear a 200-game gate.

## Configuration (every round)

- Self-play from the previous round's network: 2,800 games, 9 workers, 600
  simulations on full searches, playout cap 25% full / 100 fast simulations,
  balanced openings (threshold 0.3, 64 attempts), draw cap 150 plies. Seeds:
  r6 120000-122799, r7 130000-132799, r8 140000-142799.
- Fine-tune the previous network 2,000 x 256 steps, lr 0.0005, soft policy,
  blend value, value weight 0.5, `--value-weighting game`, replay of all
  earlier rounds and teacher-v2 with `--hold-out-extra`; 10% of the new
  round's games held out (seed 0).
- Gate: `scripts/promotion_gate.py`, 200 paired games, guided MCTS 0.2 s/move,
  against the current champion; promote only if the 95% interval excludes
  50%. Gate seeds: r6 12000-12099, r7 12100-12199, r8 12200-12299.
- Round 6 ran from `round6-run.sh` (commit `dcc787e`); rounds 7-8 from
  `overnight-loop.sh` (commit `9d5d911`, clean apart from `.claude/`), with a
  08:00 deadline (`overnight-watchdog.sh`), no round started with under 2.5 h
  left, and stops on errors, gate errors or NaN. `caffeinate -i -s` was held
  for the loop's lifetime after the Mac's 1-minute idle sleep was found to be
  pausing the run.
- Apple M1 Pro, CPU only, $0. Log: `overnight-loop.log`; gate summaries:
  `gate-r*.json`, `confirm-r7.json`.

## Results

| Round | Self-play | Black-White-Draw | Positions | Held-out policy acc / decisive sign acc | Gate | Result |
| --- | --- | --- | --- | --- | --- | --- |
| r6 | 140 min* | 1709-974-117 | 28,070 | 60.1% / 68.4% | vs az-r2: 105-94-1, 52.8% (45.8-59.6%) | not promoted |
| r7 | 66 min | 1683-1006-111 | ~28k | 61.5% / 69.8% | vs az-r2: 124-73-3, 62.7% (55.9-69.2%) | **promoted** |
| r8 | 71 min | 1675-979-146 | ~28k | 59.6% / 71.5% | vs az-r7: 106-91-3, 53.8% (46.8-60.5%) | not promoted |

\* Round 6 self-play was slowed first by a foreground game and then by idle
sleep; rounds 7-8 show the unimpeded rate (about 40 games/min). Fine-tuning
took 12.5 min per round, each gate about 5 min. Saving each round (record
verification through the JAX environment) took about 20 min, now the largest
fixed cost after self-play.

**Confirmation.** Because rounds 3-7 all gated against az-r2, one lucky pass
was plausible. az-r7 was re-gated against az-r2 on fresh seeds 13000-13099:
**119-79-2, 60.0% (95% CI 53.1-66.5%)**, which passes on its own. az-r7 is the
new champion.

Colour split in the confirmation gate: az-r7 scored 85/100 with Black and
34/100 with White (az-r2 is similarly strong with Black), so Black's
advantage still dominates individual games.

## Interpretation

- Two 64-channel rounds after az-r5 (r6, r7) gave the first promotion since
  az-r2. The pattern matches rounds 1-2: a non-significant round followed by
  a significant one; 52.8% then 62.7%.
- Against az-r2, the chain of 64-channel candidates scored 51.5% (distilled),
  54.5% (r5), 52.8% (r6), then 62.7% and 60.0% (r7, two independent gates).
  r6 is the outlier dip; with ~7-point intervals the r5-r6 differences are
  noise.
- r8 vs az-r7 at 53.8% is a positive but non-significant step from the new
  champion. Held-out metrics are measured on each round's own new games, so
  they are not comparable across rounds.

## Anomalies and limitations

- Records still take ~20 min to verify per round; worth replacing JAX replay
  with the pattern board plus a final environment check on a sample.
- Gates use 0.2 s/move with the shared VCF cache; these numbers are not
  directly comparable with gates before `dcc787e`.
- az-r7 has not been measured in the browser. It has 2.7x the weights of
  az-r2 and roughly 2x slower evaluation; check in-browser speed and strength
  at the website's time budget before shipping it.
- Strength is still only relative to our own bots.

## Next decision

- Keep az-r7 as champion (confirmed). Continue rounds from az-r8 gated against
  az-r7 with the same loop.
- Benchmark az-r7 in the browser at the site's move budget before replacing
  az-r2 there.
- Add an external anchor (an established Gomoku engine at fixed time) so
  strength can be stated in absolute terms.
