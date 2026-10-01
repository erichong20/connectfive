# Negamax v0 evaluation

## Hypothesis

Depth-limited negamax with alpha-beta pruning and a fixed node budget will make
better tactical decisions than the one-ply bot at an affordable local CPU cost.

## Code and dirty state

- Base commit: `91eb99fd4513ab570b8e0e6dcf6c702f42c81c40`.
- Search and suite changes were uncommitted during measurement.

## Configuration

- Maximum depth: 3.
- Candidate width: 10.
- Node budget: 1,500 per move.
- Seed: 0.
- Match protocol: colors alternate every game from an empty board.
- Rules: 15x15, exactly five wins, legal non-winning overlines.
- Compute cost: local CPU only; $0 purchased compute.

## Environment

- Darwin 25.6.0, arm64.
- Python 3.14.7.
- JAX 0.11.2.
- NumPy 2.5.3.

## Tactical suite

Both Tactical and corrected Negamax solved all 8 essential positions. The suite
is intentionally a regression floor, not a broad strength measurement.

## Match results

An initial evaluator that violated negamax sign symmetry scored 2 wins and 8
losses against Tactical over 10 games. This was treated as a correctness signal,
not a valid strength result.

After replacing it with `best_attack(player) - best_attack(opponent)`:

`python scripts/evaluate.py negamax tactical --games 10 --seed 0`

- 7 wins, 1 loss, 2 draws.
- 81.6 average moves per game.
- 182.2 seconds total.
- No illegal moves observed.

An earlier four-game smoke sample was 3-1, consistent with but not independent
of the ten-game run because both used seed 0.

## Anomalies and limitations

- Ten games are far too few for a confident promotion decision.
- Both bots score 8/8 on the compact suite, so it does not distinguish them.
- Runtime varies considerably with game length and search complexity.
- No transposition hits appeared in the representative depth-three fixture;
  the table may become useful only at greater depth or with broader searches.
- The browser has not yet received Negamax because its latency budget needs
  separate measurement.

## Decision

Keep Negamax as an experimental CLI/evaluation agent. Next, review saved losses
and benchmark a smaller browser node budget before considering promotion.

Future match runs can use `--failures-json runs/failures.json` to retain every
loss as a replayable move sequence for fixture review.
