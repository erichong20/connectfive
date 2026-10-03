# AlphaZero-lite round 5: 64-channel network, capped draws, per-game value weights

## Hypothesis

The value diagnosis (`experiments/2026-10-01-value-diagnosis/`) found the
32-channel network lags its own search at every stage of the game and that
label changes do not help. A wider network should predict outcomes better
and, after one self-play round, beat the champion az-r2.

## Configuration

Base commit `f3c6745` (clean). Scripts copied here (`run.sh`, `gate.py`,
`launch.sh`); raw outputs in ignored `runs/az-r5/`.

1. **Distil**: a new 64-channel, 4-block network (356,492 parameters vs
   133,388) trained from scratch, 6,000 x 256 steps, lr 0.002 (cosine), soft
   policy, blend value, value weight 0.5, `--value-weighting game`, on round-4
   self-play (10% of games held out, seed 0) plus rounds 1-3 and teacher-v2.
2. **Distil gate** vs az-r2, guided MCTS 0.2 s/move, 100 paired games,
   seeds 9000-9049; automatic stop below 40%.
3. **Self-play**: distilled network, 600 games, seeds 110000-110599, 600
   simulations, balanced openings (threshold 0.3, 64 attempts),
   `--draw-ply-cap 150`.
4. **Fine-tune** 2,000 x 256 steps, lr 0.0005, same targets, on round-5 games
   plus rounds 1-4 and teacher-v2.
5. **Final gate** vs az-r2, seeds 9100-9149, 100 paired games; promotion
   requires the 95% interval to exclude 50%.

Hardware Apple M1 Pro (10 cores), CPU only, $0. Wall-clock limit enforced by a
watchdog (5 h, extended to about 8 h during the run because self-play saves
only at the end; see anomalies).

## Cost

| Stage | Wall clock |
| --- | --- |
| Distil (6,000 steps) | 97 min |
| Distil gate | 3 min |
| Self-play (600 games, 12.0M simulations, 9 workers) | 162 min |
| Fine-tune (2,000 steps) | 36 min |
| Final gate | 2 min |
| **Total** | **about 5 h 17 min** |

A 64-channel single-position evaluation measured 1.14 ms vs 0.61 ms for 32
channels; a training step about 2x slower. Self-play took 3.3x the
round-4 time for 1.2x the games.

## Results

| Measurement | Value |
| --- | --- |
| Distilled net vs az-r2 | 51-48-1, 51.5% (95% CI 42-61%) |
| Self-play Black-White-Draw | 379-179-42 (63% Black; r4 66%) |
| Draws ended by the 150-ply cap / provably dead | 42 / 0 |
| Positions, mean length | 26,999, 49.0 plies |
| Fine-tuned held-out (r5 games): policy acc, decisive sign acc | 58.6%, 70.3% |
| **Final gate vs az-r2** | **54-45-1, 54.5% (95% CI 45-64%) - not promoted** |
| Wins as Black / as White (final gate) | 39 / 15 of 50 each |

Decisive sign accuracy on the same held-out round-4 games (value-diagnosis
metric): az-r2 56.9%, az-r4 57.9%, **distilled 64-channel 62.3%**. The
fine-tuned model scores 63.4% there, but it trained on all round-4 games
including those held out, so that number is contaminated and not used.

## Anomalies

- Search nodes per move at the same 0.2 s budget: distilled net 81, fine-tuned
  net 306, az-r4 (32 channels) 313. Same architecture and timing (~150 ms per
  move) for the two 64-channel nets; not explained. It may reflect how the
  value scale changes proven/forced-move shortcuts in the node count.
  Investigate before trusting node-based speed comparisons.
- Duration estimates were too low (planned 2.5-3 h). The watchdog was moved
  twice; once, killing the old watchdog briefly risked triggering its stop
  branch. Future runs should save self-play incrementally so a limit never
  discards finished games.
- Fine-tuning replay included all round-4 games, leaking the held-out
  round-4 split into the fine-tuned model (see above).

## Interpretation

The wider network clearly improves the value head (62.3% vs 57.9% decisive
sign accuracy on identical held-out games, the first value gain since
az-r2), and it matches az-r2 in play straight from distillation. One
self-play round added a non-significant +3 points (51.5% to 54.5%). As in
rounds 3-4, a single round is not enough to clear the 100-game gate, and the
100-game gate cannot resolve gains much smaller than about 10 points.

## Next decision

- Keep az-r2 as champion; keep `runs/az-r5/model` as the 64-channel
  candidate.
- A second 64-channel round from `runs/az-r5/model` is the natural next step
  (rounds 1 to 2 compounded), but costs about 5 h at current speed. Before
  that: save self-play incrementally, exclude held-out games from replay, and
  consider a 200-game gate so smaller gains are measurable.
- The browser bot would need the larger network's weights (2.7x) and slower
  evaluations; measure its in-browser speed before any promotion ships.

## Follow-up: runner fixes before round 6

- **Resumable self-play.** `scripts/selfplay_round.py` appends each finished
  game to `<out>.games.jsonl` as it completes (`imap_unordered`) and reuses
  games already in the log on restart; a torn final line is ignored and never
  appended onto. The summary records `resumed_games`, and `seconds` covers
  only the current invocation. A wall-clock stop now loses at most the games
  in flight.
- **No replay leakage.** `scripts/train_supervised.py --hold-out-extra`
  drops each replay dataset's own validation games (the same `split_by_game`
  split it gets as a primary dataset), so a model's held-out games stay
  held out in every later round. Off by default to keep earlier commands
  reproducible; use it from round 6 on.
- **200-game gate.** `scripts/promotion_gate.py` runs paired-colour blocks of
  `scripts/evaluate_guided.py` in parallel over one contiguous seed range,
  combines them (`combine_summaries` in `src/connectfive/match.py`), writes
  `gate.json`, and reuses finished blocks on restart. It promotes only if the
  95% interval excludes 50% (`--min-score` turns it into an early-stop
  check). With 200 games the interval half-width near 55% is about 7 points
  instead of about 10.

Tests: `tests/test_guided_search.py` (log resume and torn line),
`tests/test_dataset.py` (replay hold-out), `tests/test_match.py` (combining
blocks). Smoke runs of both scripts resumed correctly.
