# Lessons

This document collects explanations that should remain useful after an
individual experiment is over.

## Start with an opponent you can understand

A random bot verifies the plumbing but teaches almost no strategy. A tactical
rule bot is the next useful step because every choice can be traced to a stated
priority. When it succeeds, we can explain why. When it fails, that failure
becomes a concrete motivation for search.

## Training loss is not playing strength

A policy can better imitate its data while becoming worse against a particular
opponent, and a value model can reduce average error while missing the tactical
positions that decide games. Promotion therefore requires games and fixed
tactical positions, not only loss curves.

## Exact-five changes seemingly obvious heuristics

In freestyle Gomoku, a run of six contains a winning five. Under this project's
standard exact-five rule, an overline is legal but non-winning. Every rule bot,
search terminal test, training label, and browser implementation must preserve
that distinction.

## Look ahead by assuming the opponent chooses the worst reply

The one-ply tactical bot ranks only its own candidate moves. The first search
bot instead scores each move by its worst plausible opponent reply:

`value(move) = immediate_score(move) + min_reply leaf_score(move, reply)`

This is the core minimax idea. The implementation caps the candidate set at 16
moves and each reply set at 12 moves, so a decision considers at most 192 leaf
positions. That fixed budget is cheap and predictable, but it can miss a quiet
move excluded by the heuristic ordering. A later negamax implementation will
make depth and node budgets explicit and use alpha-beta pruning to search more
selectively.
