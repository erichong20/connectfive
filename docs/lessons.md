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

## Negamax requires an antisymmetric evaluator

Negamax relies on the identity that a position's value for one player is the
negative of its value for the other. Our first leaf evaluator subtracted one
defense-weighted tactical score from another. Because each score already mixed
both players' patterns, the attacking terms partly cancelled and good attacks
could receive a negative contribution. The resulting search went 2-8 against
the tactical bot despite being mechanically correct.

The corrected evaluator computes each player's attacking potential separately:

`value(position, player) = best_attack(player) - best_attack(opponent)`

After this correction, the same search configuration went 7-1-2 in a ten-game
diagnostic. That sample is not a promotion result, but the reversal demonstrates
that deeper search magnifies evaluation mistakes rather than automatically
fixing them.

## Iterative deepening makes a node budget safe

The search completes depth one, then depth two, and so on. If its 1,500-node
budget expires halfway through a deeper iteration, it discards that partial
iteration and returns the best move from the last completed depth. This avoids
comparing thoroughly searched early candidates with barely searched later
candidates. Reported nodes, completed depth, alpha-beta cutoffs, cache hits, and
elapsed time make the cost visible.

## A node budget is not a complete time budget

Candidate generation, pattern scoring, and move ordering perform work outside
the recursive node counter. In the 250-node randomized-opening experiment,
Negamax averaged 227 counted nodes but still needed 331 ms per move. A browser
or tournament budget must therefore measure wall-clock latency as well as nodes.

## Pair openings, not just colors

Alternating colors from an empty board is better than a single-color match, but
deterministic bots can still repeat a narrow family of games. The evaluation
harness now generates one reproducible local opening for each pair and plays it
twice with agent colors reversed. This varies the positions while keeping the
comparison fair.

## More search cannot rescue the wrong horizon

On the same ten paired openings, 250-node Negamax scored 5-2-3 and 1,500-node
Negamax scored 6-3-1: both earned 6.5 match points. The larger budget raised
completed depth from 2.29 to 2.96 and increased average move time from 343 ms to
461 ms, but did not improve the match score. Saved losses commonly ended with
two simultaneous opponent winning moves. The useful next change is to value or
extend fork threats earlier, not simply search more nodes with the same leaf
evaluation.

## Tactical extensions need their own cost discipline

Quiescence search is not automatically cheaper just because it follows only
forcing moves. Detecting whether hypothetical Gomoku moves create forks can be
expensive, and extending every iterative-deepening leaf can consume the budget
before the principal search reaches useful depth.

The first ThreatSearch version solved the tactical suite but reached only depth
1.56 on average, exhausted its 100-node allowance on every move in its four
losses, and took 558 ms per move. It scored 3-4-3 on the same openings where
ordinary Negamax earned 6.5/10 match points. A better version should use cheap,
incremental threat features for ordering and reserve extensions for positions
that are already demonstrably forcing. It should not scan candidate forks at
every leaf.

## Make a neural network memorize a tiny dataset first

Before self-play, search integration, or a long training run, a network should
be able to overfit a handful of known examples. This cheap test checks the data
layout, targets, optimizer, gradients, both output heads, legal-action masking,
augmentation, and checkpoint path in isolation. Our 133,100-parameter network
reduced total loss on eight tactical fixtures from 5.8728 to 0.3472 in roughly
5.2 CPU seconds.

The final policy loss was informative. Four of the eight fixtures have two
equally acceptable moves, so their uniform targets have entropy `ln(2)`. The
lowest possible average cross-entropy is therefore approximately
`4 * ln(2) / 8 = 0.3466`, almost exactly the observed 0.3470. A nonzero loss can
be the mathematically correct optimum rather than residual model error.

## Normalization dimensions can silently kill a head

The first value head projected each board point to one channel and then applied
LayerNorm. Normalizing a single value always produces zero, so no board signal
could pass through that layer. The loss still ran, making this easy to miss.
Projecting to four channels before LayerNorm restored variation and let the
value head fit the tiny targets. Tests that require each head to learn are more
valuable than shape tests alone.

## Relative features still need absolute context

The first two network planes describe the mover's stones and the opponent's
stones. This makes the representation naturally follow the side to move and is
compatible with zero-sum value targets. A third, constant plane records whether
Black is to move, because opening advantage and color-dependent data statistics
cannot be recovered from the two relative planes alone.

## Split supervised board data by complete games

Adjacent positions from one game differ by only one stone. A position-level
random split would put near-duplicates in training and validation and make the
reported generalization misleadingly strong. Each example therefore carries a
game ID, and an entire game is assigned to exactly one split. Randomized opening
moves are replayed to reach the teacher's starting state but are not policy
targets, because no teacher selected them.

## Behavior cloning compounds small mistakes

The 32-game supervised model predicted 39.9% of held-out teacher moves exactly
and defeated Random 20-0, but lost 0-20 to Tactical. These results are compatible:
an imitation model is trained on states visited by its teachers, then evaluated
on states produced by its own earlier choices. One different move can move the
game outside the training distribution, and errors compound from there.

This is a central reason AlphaZero improves a policy through search and self-play
rather than treating expert move classification as the final bot. Before adding
MCTS, we should still improve the supervised control: gather broader positions,
represent tied teacher choices more honestly than a single one-hot action where
possible, and inspect the earliest divergence in losses to Tactical.

## More examples helped policy before value

Increasing the corpus from 8 games/562 examples to 32 games/3,024 examples
raised held-out policy accuracy from 15.5% to 39.9%. The v1 value head still had
held-out MAE 1.12 even though training MAE was 0.17. A final game outcome gives
every position in a game the same alternating winner signal, so 32 games provide
far fewer independent value labels than 3,024 position rows suggest. Count games,
outcome balance, and opening families—not only positions—when reasoning about
value-data scale.

## Check that a search budget can finish the depth it advertises

Negamax was configured for depth 3 with width 10 but given 100 nodes. One full
depth-2 iteration already needs about 10 + 10 x 10 nodes, so the depth-3
iteration was almost always abandoned. In the v1 corpus only 58 of 1,511
teacher moves completed depth 3. Record the completed depth, not just the
configured maximum. A "depth-3 teacher" that mostly plays depth 1-2 is not
stronger than a one-ply heuristic, and it scored 11-14-7 against Tactical.

## Transposition entries need bound flags

With alpha-beta, a node that fails low only proves that its value is *at most*
alpha. Storing that number as exact makes later probes with a different window
return wrong values. The pattern search stores exact/lower/upper flags, and a
test compares it with plain minimax at depths 1-3. The same reasoning applies
at the root: after the first move, a score equal to the current best may only
be an upper bound. Random tie-breaking among such "ties" can then choose a
worse move. Searching root moves against a window lowered by a margin makes
every near-best score exact.

## Make positions cheap before making search clever

Gomoku patterns change only along the four lines through the last stone. An
incremental board that caches per-cell line levels, from a memoized table of
10-cell neighbourhoods, made each make/undo about 90 microseconds. The old
code took about 1.4 ms per node plus about 5 ms per leaf. The same CPU then
searched about 13x more nodes per second, and threat pruning and VCF made
those nodes count.

## Teacher quality and label quality beat model size

Without changing the 133k-parameter network, a better teacher and better
labels moved the raw policy from 5% to 77.5% against Tactical (100 games
each). The label changes were: soft near-best targets, search-value labels,
few draws, and no board-filling endgames. Held-out value MAE fell from 1.12 to
0.61, below the 0.89 constant baseline. The train/validation gap closed to
about 3 points. That gap, not loss alone, is the evidence that model capacity
is now worth testing.

## DAgger needs a distribution gap to fix

One DAgger round (400 student self-play games, 9,430 teacher-labelled
positions) did not measurably change v2's results (73% vs 77.5% against
Tactical, overlapping intervals). DAgger fixes compounding drift. Once the
teacher corpus already covers the positions the student reaches, a small
round mostly repeats what the model has seen. Measure the student's teacher
agreement in its own games (56% here) and compare it with held-out agreement
(57%) before investing in more rounds.

## A learned policy helps tree search more than it helps move ordering

At 0.2 s per move, the v2 network inside PUCT search beat `pattern` alpha-beta
63-37 over 100 games. Using the same policy only to rank alpha-beta moves
scored 7-13. MCTS spreads visits by prior *and* value and can recover from a
wrong prior. A hard top-k cut in alpha-beta cannot recover from a pruned move.
Keeping exact tactics (wins, blocks, VCF) outside the network made the hybrid
safe even when the network is wrong.

## Forced positions still need a policy target

MCTS skips simulation when the move is forced, so visit counts are all zero
there. Converting visits to a distribution then silently produced empty
targets for 28% of positions, which contribute zero policy loss. Check that
every policy target sums to one before training. A test now checks this.

## Port by parity fixtures, then optimize

The browser bot reimplements four pieces: patterns, tactics, network, and
MCTS. Python fixtures for each piece make each port testable on its own, so a
mismatch points to one layer instead of to "the bot plays differently". Plain
TypeScript convolutions were correct but 20x too slow (37 simulations/s).
Strength needs at least about 150 simulations per move (42.5% vs `pattern` at
32 simulations, 61% at 128). A 480-byte WebAssembly SIMD kernel raised
throughput to about 300-400 simulations/s, and the parity tests showed the
optimization changed nothing else.
