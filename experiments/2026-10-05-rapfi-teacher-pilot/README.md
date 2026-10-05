# Rapfi as a teacher: pilot (not promoted)

## Hypothesis

Rapfi at 10,000 nodes per move is far stronger than our search (az-r7 is
about even with Rapfi at 100-150 nodes). Fine-tuning az-r7 on Rapfi's moves
and evaluations should transfer some of that strength cheaply.

## Data

`scripts/generate_rapfi_games.py`: Rapfi (`dhbloo/rapfi` 3c94c2a, standard
network, 1 thread, `MAX_NODE=10000`) plays itself from seeded random openings
of 2-8 plies (seeds 200000-204999); our board referees; games still open at
150 plies are draws. Each post-opening position stores Rapfi's move (one-hot
policy), its evaluation for the side to move, and the final outcome. 100
random games were replayed through the JAX environment.

- 5,000 games, 168,787 positions, 683 s on 9 workers (about 26,000
  games/hour, roughly 14x our self-play rate). Black-White-Draw 4091-697-212.
- Evaluation sign agrees with the final result on 97.4% of decisive
  positions; 52% are proven mates. A logistic fit gives p(win) =
  sigmoid(eval / 150), so value labels are `tanh(eval / 300)`
  (`runs/rapfi-teacher/pilot-v2-s300.npz`).
- A first attempt was discarded: `last_eval` mis-parsed Rapfi's mate scores
  (`+M3`) and could return an earlier search's value (see
  `experiments/2026-10-04-rapfi-anchor/` correction).

## Training

Fine-tune az-r7 (64 channels, 4 blocks) 3,000 x 256 steps, lr 0.0005
(cosine), weight decay 1e-4, soft policy target (here one-hot), value =
mean(outcome, Rapfi value), value weight 0.5, per-game value weighting, 10%
of games held out (seed 0). Rapfi data only, no replay. Script:
`train-eval.sh`. Base commit `97a73ce` plus the generator and parser fix.

## Results

| Measure | az-r7 | Rapfi-tuned (p1) |
| --- | --- | --- |
| Rapfi's move, held-out Rapfi positions (20k) | 44.4% | 48.9% |
| Winner sign, held-out Rapfi positions | 90.3% | 91.7% |
| Mean abs value, held-out Rapfi positions | 0.58 | 0.76 |
| Search's move, held-out round-8 self-play | 58.8% | 55.8% |
| Winner sign, held-out round-8 self-play | 71.8% | 69.7% |

Head-to-head (guided MCTS 0.2 s/move, 200 paired games, seeds 17000-17099):
**p1 vs az-r7 79-117-4, 40.5% (95% CI 33.9-47.4%)** - significantly worse.

Rapfi ladder (41 balanced openings x 2 colours, our side 1 s/move):

| Rapfi | az-r7 | p1 |
| --- | --- | --- |
| 30 nodes | 79.3% | 72.0% |
| 100 nodes | 54.9% | 56.7% |
| 300 nodes | 37.8% | 35.4% |
| 1,000 nodes | 11.6% | 14.6% |

No difference outside noise (each row has roughly a 10-point interval).

## Interpretation

- az-r7 already agrees with Rapfi's move 44% of the time; imitation moved
  that only to 49%. The teacher's advantage is in its search, not in a
  first-move guess our network lacks.
- Training on Rapfi games alone shifted the network away from the positions
  our own search reaches (self-play accuracy fell 3 points) and made values
  more confident, and the head-to-head result got worse. Rapfi's games are
  also one-sided (82% Black wins), so many positions are already decided.

## Next decision

Keep az-r7. A better teacher design labels *our* positions instead of
Rapfi's games (DAgger): run Rapfi at 10k nodes on positions from az-r7
self-play, and train on those labels mixed with self-play replay at a lower
learning rate, so the network improves where its own search goes. If that
also fails to move the gate, the evidence points to search speed (a native
engine) as the bigger lever.
