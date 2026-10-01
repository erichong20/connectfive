# Pattern-search teacher and supervised v2

## Why

A review of the v1 run found that most of its weakness came from the data,
not the network:

- With a 100-node budget, Negamax could not finish depth 2: a full depth-2
  pass costs about 10 + 10 x 10 nodes. It usually returned a depth-1 or
  depth-2 answer and scored 11-14-7 against Tactical.
- Each Negamax move cost about 142 ms. About 84% of that was the leaf
  evaluation, which rescanned roughly 95 candidate cells for both players in
  Python loops.
- Seven of the 32 games filled the whole board. They supplied 51% of the
  value labels, all zeros, and many low-information late-game positions.
- The old transposition table stored fail-low results as exact values. Root
  ties included moves whose scores were only upper bounds.

## What changed

1. An incremental pattern board (`patterns.py`) caches per-direction line
   levels, so a move costs a few table lookups.
2. A new `pattern` search (`pattern_search.py`) adds bound-flagged
   transposition entries, threat-aware move generation, forced-reply
   extensions, VCF proofs, and exact near-best root scores.
3. Parallel teacher self-play (`teacher.py`) labels every post-opening
   position with a hard move, a soft target, a search value, and the outcome.
4. The network gets a constant fourth input plane, a jitted agent, and new
   training options.
5. A DAgger round script labels positions that the student itself reaches.

## Results

See `experiments/2026-09-30-pattern-teacher-v2/README.md`. In summary:
Pattern beats Tactical 40-0 and Negamax 19-1. The v2 raw policy beats Tactical
75-20-5 over 100 games, against 4-94-2 for v1. One DAgger round did not
measurably help.

## Next

Use the v2 network inside search (move ordering, leaf value, or MCTS priors),
or test a wider network. The small train/validation gap now justifies the
latter.
