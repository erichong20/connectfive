# Compact tactical suite and negamax

## Goal

Take the faster classical-search path without spending time on a large manual
test library: create a small essential suite, add proper budgeted negamax, and
use automatically recorded diagnostics to decide whether further work is
justified.

## What changed

- Added eight JSON tactical fixtures covering wins, blocks, priority, exact-five
  overlines, board edges, diagonals, a cross fork, and open-four creation.
- Added a reusable suite loader/evaluator and CLI report.
- Added iterative-deepening negamax with alpha-beta pruning, move ordering, a
  transposition table, a 1,500-node default budget, and a width of ten moves.
- Added a search explanation command with root candidate scores and statistics.
- Added `--failures-json` to save replayable losses from evaluation runs.
- Added regression tests for tactical accuracy, node-budget enforcement,
  diagnostics, and evaluator sign symmetry.

## Fixture corrections

The first defensive fixtures used open fours with two winning endpoints. Those
positions were already forced losses, so accepting either single block was
incorrect. The suite now closes one end to create true one-answer blocks. The
overline fixture also accidentally allowed a separate exact-five win and was
corrected by closing that endpoint. These were test-design bugs, not agent bugs.

## Evaluator correction

The first negamax evaluator combined two scores that each mixed attack and
defense. This violated negamax's sign-symmetry assumption and produced a 2-8
result against Tactical. It was replaced by separate attack-only scores for
each player.

## Results

- Essential suite: Tactical 8/8; Negamax 8/8.
- Search explanation on `prevent-cross-fork`: selected H8 at completed depth 3,
  207 nodes, 27 alpha-beta cutoffs, and about 242 ms in that run.
- Corrected Negamax versus Tactical, 10 paired-color games, seed 0: 7 wins,
  1 loss, 2 draws; 81.6 average moves; 182.2 seconds.
- All 69 tests pass, including paired openings, confidence reporting, search
  diagnostics, and loss-record serialization.

## Interpretation

The corrected search is promising but not promoted. Ten games have wide
uncertainty, both bots solve the same small suite, and match runtime varies with
position complexity. The harness can now save losses automatically; the next
evidence should come from reviewing those replayable failures and from a
browser-specific latency benchmark, not from paid compute.

## Randomized-opening follow-up

The harness now generates a reproducible four-ply local opening per game pair,
reuses that opening with agent colors reversed, reports match-score uncertainty,
separates Black and White results, and aggregates move latency, nodes, and depth.

At a 250-node budget, 50 games from seed 200 produced 16 wins, 21 losses, and 13
draws. Negamax's match score was 45%, with an approximate 95% interval of
32%-59%. It scored 10-8-7 as Black and 6-13-6 as White, averaged 331 ms per move,
227 nodes, and depth 2.32. The run took 767.5 seconds and cost $0.

Twenty-one losses were saved. In the four shortest losses, Negamax's final
defensive turn faced two distinct exact-five threats, so no single move could
save the game. This points to earlier fork prevention rather than missed
one-move blocks.

A controlled ten-game comparison on the same openings found 5-2-3 at 250 nodes
and 6-3-1 at 1,500 nodes. Both configurations scored 6.5/10 match points; the
larger budget averaged 461 ms per move versus 343 ms. Negamax will therefore not
be added to the browser yet.

## Threat-extension follow-up

`ThreatSearchAgent` preserves Negamax and adds broken five-cell pattern scores
plus a two-ply quiescence search for exact wins, forced blocks, fork attacks,
and fork prevention. Fork detection was optimized to inspect only the four
lines changed by a hypothetical move.

The tactical suite remained 8/8, but the controlled ten-game run on seed 100
scored 3 wins, 4 losses, and 3 draws. It averaged 558 ms per move, 99 nodes, and
depth 1.56. All 145 ThreatSearch moves in its four saved losses reached the
100-node cap. Each loss still ended in a double exact-five threat.

The 4.5/10 match score failed the predeclared go/no-go threshold of exceeding
ordinary Negamax's 6.5/10 on the same openings. No 50-game run or browser port
was attempted.
