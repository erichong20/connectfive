# Compact policy/value network

The current neural milestone is intentionally smaller than AlphaZero. Its job
is to establish a trustworthy learned evaluator on local hardware before MCTS
adds cost and moving parts.

## Inputs

Each position is encoded as a 15x15x3 tensor:

1. stones belonging to the player about to move;
2. stones belonging to the opponent;
3. a constant plane that is one when Black is to move.

The relative stone planes let one network evaluate either player. The absolute
color plane retains information about first-player advantage. Training targets
and features are transformed together with the square's eight rotations and
reflections.

## Architecture

The default model has 133,100 trainable parameters:

- a 3x3 convolution from 3 to 32 channels;
- four 32-channel residual blocks, each with two 3x3 convolutions;
- a policy head that emits one logit for each of 225 intersections;
- a value head that emits one scalar in `[-1, 1]` from the mover's perspective.

Illegal policy logits are masked before the softmax loss. The supervised loss
is the sum of policy cross-entropy and squared value error:

`loss = -sum(target_policy * log(predicted_policy)) + (value - outcome)^2`

This is deliberately compact enough for CPU experiments. Width and depth should
only increase after measurement shows underfitting on held-out games.

## What is tested

`tests/test_network.py` checks output shapes, parameter scale, mover-relative
encoding, legal masking, all eight feature/policy transformations, parameter
checkpoint reproduction, and actual loss reduction on a tiny batch. The
overfit script repeats that final test with the full default architecture and
the fixed tactical suite.

## Next data contract

The next dataset should contain complete games rather than disconnected
positions. Each example needs encoded state, legal mask, chosen-move policy,
final outcome from the mover's perspective, game identifier, ply, seed, and
generating-agent configuration. Entire games must belong to either training or
validation so adjacent positions cannot leak across the split.

Initially, Tactical-versus-Negamax and self-play games can provide move labels;
the final result supplies real value labels. This is imitation learning, so the
network may inherit the agents' weaknesses. The milestone succeeds only if it
generalizes to held-out games and the raw policy beats Random in paired games.

Only after that result should the project connect the network to MCTS and use
search visit counts as improved policy targets.

## First supervised result

The pipeline now replays complete recorded games into examples, excludes random
opening moves from policy targets, attaches game/ply/player/teacher provenance,
and makes a deterministic game-level split. `scripts/generate_dataset.py`
creates the corpus, `scripts/train_supervised.py` trains and reports held-out
metrics, and `scripts/evaluate_network.py` runs paired-color games from a saved
checkpoint.

The first 32-game corpus contained 3,024 examples from Tactical-versus-Negamax
games. The model achieved 39.9% held-out policy accuracy but unreliable held-out
value predictions. Its raw greedy policy beat Random 20-0 and lost to Tactical
0-20. This is a useful baseline, not a promotion: it demonstrates learning and
legal inference while exposing behavior-cloning distribution shift.
