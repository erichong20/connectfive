# Supervised warm start

## Goal

Turn recorded classical games into a leakage-safe policy/value dataset, train
the compact network with real outcomes, and measure the raw network in games.

## Implementation

- Replays complete `GameRecord` objects and verifies the terminal result.
- Stores features, legal masks, selected actions, mover-relative outcomes, game
  IDs, plies, players, and generating-agent names.
- Excludes randomized opening moves from policy targets.
- Saves numeric arrays without pickle and a readable provenance manifest.
- Splits by entire games and applies independently sampled D4 augmentation.
- Reports train and validation policy loss/accuracy and value MSE/MAE.
- Loads checkpoints into a greedy, legal-masked raw network agent.

## Eight-game smoke run

The v0 corpus had 562 examples. After 500 updates, training policy accuracy was
81.2% but held-out-game accuracy was only 15.5%; validation value MAE was 0.82.
The raw policy beat Random 20-0 and lost to Tactical 0-10. This established the
end-to-end path and exposed overfitting.

## Thirty-two-game run

The v1 corpus used 32 paired-color Tactical-versus-Negamax games with four-ply
randomized openings and a 100-node Negamax budget. It produced 3,024 examples:

- teacher match: Negamax 11 wins, 14 losses, 7 draws;
- generation time: 219.9 seconds;
- training: 1,000 updates, batch size 128, Adam at 0.001;
- training time: 61.9 seconds;
- training policy accuracy: 51.6%; held-out: 39.9%;
- training value MAE: 0.17; held-out: 1.12;
- network versus Random: 20-0, both colors, 21.5 ms per move;
- network versus Tactical: 0-20, both colors, 20.0 ms per move;
- illegal moves: zero;
- purchased compute: $0.

## Interpretation and decision

Four times as many games materially improved held-out policy prediction, but the
value head still memorized game-level outcomes and the raw policy did not match
its Tactical teacher. The perfect Random result shows genuine basic tactical
learning; the perfect Tactical losses show compounding imitation error and
insufficient coverage.

Preserve the model as a baseline. Next, inspect early divergences in Tactical
losses, improve teacher targets/data diversity and value balance, and rerun a
small controlled ablation. Do not increase network size or start a long
AlphaZero run yet.
