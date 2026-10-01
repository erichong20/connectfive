# ThreatSearch v0

## Hypothesis

Extending Negamax leaf nodes through exact wins, forced blocks, and fork moves,
while recognizing broken five-cell patterns, will prevent the double threats
seen in saved Negamax losses.

## Implementation

- New `ThreatSearchAgent`; the original Negamax baseline remains unchanged.
- Nominal maximum depth 3, candidate width 10, node budget 100.
- Up to two extra forcing plies, forcing width 8.
- Fork detection checks only legal replies on the four lines affected by the
  hypothetical move.
- Broken patterns score five-cell windows containing four, three, or two stones.
- Local CPU only; $0 purchased compute.

## Pre-match benchmark

ThreatSearch solved 8/8 essential tactical fixtures. At 100 nodes it averaged
232 ms on the suite with a maximum observed position of 641 ms. This was already
slower and shallower than desired, so the match used a ten-game go/no-go gate.

## Controlled result

`python scripts/evaluate.py threatsearch tactical --games 10 --seed 100 --opening-plies 4 --node-budget 100`

- 3 wins, 4 losses, 3 draws; 4.5/10 match points.
- As Black: 3-0-2; as White: 0-4-1.
- 114.4 average moves per game.
- 558.2 ms average move time.
- 98.6 average nodes and completed depth 1.56.
- 310.5 seconds total runtime.
- No illegal moves.

The previous ordinary Negamax variants both earned 6.5/10 match points on these
same openings. ThreatSearch therefore failed the go/no-go gate, and no 50-game
run was started.

## Failure review

All 145 ThreatSearch decisions in the four losses reached the 100-node budget.
Each loss still ended at a defensive turn with two opponent exact-five moves,
so the extra tactical machinery did not prevent the fork early enough.

## Interpretation

The idea was plausible, but this implementation spent too much work classifying
forks at leaf nodes. That starved the principal search, reduced completed depth,
and increased latency. The next design should add inexpensive incremental threat
features to move ordering and extend only verified forcing sequences. More games
or a larger budget are not justified for this version.

All 69 repository tests pass.
