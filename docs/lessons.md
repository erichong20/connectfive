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

