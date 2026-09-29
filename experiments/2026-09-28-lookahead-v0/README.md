# Look-ahead v0 evaluation

## Hypothesis

A fixed two-ply worst-reply search can add tactical foresight without paid
compute or an impractical interactive delay.

## Code and configuration

- Working tree based on commit `e6512689aa2af7e09f49a6d49193770843911f91`
  with uncommitted look-ahead changes.
- Root width: 16 candidate moves.
- Reply width: 12 candidate moves.
- Seed: 0.
- Rules: 15x15, exactly five wins, overlines legal but non-winning.
- Hardware: local CPU; no purchased compute.

## Commands and results

`python scripts/evaluate.py lookahead random --games 20 --seed 0`

- 20 wins, 0 losses, 0 draws.
- 47.0 average moves per game.
- 148.7 seconds.

`python scripts/evaluate.py lookahead tactical --games 2 --seed 0`

- 0 wins, 0 losses, 2 draws.
- 225.0 average moves per game.
- 43.3 seconds.

All 38 Python tests and the browser production build passed.

## Interpretation

The bot is legal, deterministic in Python, responsive enough for human play,
and still beats random. These results do not show that it is stronger than the
one-ply tactical bot: it took much longer to beat random, and the tactical
comparison was two full-board draws. It remains an experimental opponent and
does not pass the roadmap promotion gate.

## Next decision

Create adversarial tactical fixtures that distinguish one-ply from two-ply
reasoning. Then implement proper negamax with alpha-beta pruning and an explicit
node budget, using those fixtures before spending time on larger match samples.
