# Two-ply look-ahead bot

## Hypothesis

A small worst-reply search should avoid more tactical mistakes than the one-ply
bot while remaining responsive on a laptop and in the browser.

## Design

- Preserve `TacticalAgent` as the historical baseline.
- Add `LookaheadAgent` with a 16-move root width and 12-reply width.
- Always order exact wins and forced blocks before heuristic candidates.
- Score each root move using the least favorable opponent reply.
- Recognize wins, forced threats, and double immediate threats at the leaf.
- Mirror the same policy in the browser and make it the default opponent.

## Compute budget

At most 192 reply leaves are examined for a normal decision. This is intended
for local CPU use and requires no paid compute.

## Verification and results

- Ruff passes.
- All 38 Python tests pass.
- The production browser build passes.
- Look-ahead versus random, seed 0: 20 wins, 0 losses, 0 draws; 47.0
  moves per game; 148.7 seconds on the local CPU.
- Look-ahead versus tactical, seed 0: 0 wins, 0 losses, 2 draws; both
  games filled all 225 intersections; 43.3 seconds total.

The random result verifies basic competence, not an improvement: the tactical
baseline previously won its random games much faster. The two tactical draws
are also too small a sample to estimate relative strength. The look-ahead bot
is therefore an experimental playable agent, not a promoted champion.

## Limitations and next decision

This is fixed-width, two-ply search rather than general minimax. Candidate
ordering can hide a quiet best move, and the hand-built leaf evaluator still
has blind spots. The next step is to compare it with the tactical baseline,
collect failure positions, then decide whether full negamax and alpha-beta are
justified.
