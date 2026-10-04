# External anchor: az-r7 against Rapfi

## Question

How strong is the bot relative to public Gomoku engines? Gomocup, the annual
computer Gomoku tournament, keeps Elo lists per rule; its "Standard" rule is
our game (15x15, exactly five). Rapfi tops that list (Rapfi 2025: 2791;
middle examples Yixin 2018: 2200, Hewer 2015: 1818). Rapfi is open source and
builds on macOS, so it is the anchor here.

## Setup

- Rapfi: `dhbloo/rapfi` commit `3c94c2a` (GPL-3.0), built locally with CMake
  (`-DUSE_NEON=ON -DUSE_NEON_DOTPROD=ON`, clang, Release) into the ignored
  `runs/engines/`; standard-rule network `mix9svqstandard_bs15.bin.lz4`
  (sha256 `84334f5d...`) and classical `model210901.bin` from
  `dhbloo/rapfi-networks` (CC0). One thread, at most 1 s per move,
  `INFO rule 1` (exactly five), strength limited by `INFO MAX_NODE`.
- Our side: az-r7 guided MCTS, 1 s per move, single-threaded XLA.
- Protocol: `src/connectfive/piskvork.py` speaks the Gomocup (Piskvork)
  stdin/stdout protocol, sending the whole position with `BOARD` each move;
  `scripts/play_external.py` referees with our own board (exact five, illegal
  moves and timeouts lose).
- Openings: random 4-ply openings proved useless (first sweep, 40 games per
  setting): Rapfi rated them about +640 for Black in the median, and our score
  stayed at 22-32% from Rapfi at 1,000 nodes up to full strength, with
  almost every win as Black. `scripts/balanced_openings.py` kept seeded 4-ply
  openings that Rapfi (200k nodes) rates within 150 of even: 41 of 2,715
  (`openings.json`). Each is played twice with colours reversed: 82 games per
  setting. Script: `sweep.sh`. Apple M1 Pro, CPU only, $0.

## Results

| Rapfi | Rapfi ms/move | W-L-D | Score | 95% CI | Wins as B / W | Elo diff (ours) |
| --- | --- | --- | --- | --- | --- | --- |
| 30 nodes | 1 | 65-17-0 | 79.3% | 69.3-86.6% | 32 / 33 | +233 |
| 100 nodes | 1 | 45-37-0 | 54.9% | 44.1-65.2% | 24 / 21 | +34 |
| 300 nodes | 2 | 30-50-2 | 37.8% | 28.1-48.6% | 17 / 13 | -86 |
| 1,000 nodes | 3 | 9-72-1 | 11.6% | 6.3-20.3% | 4 / 5 | -353 |
| full, 1 s | 624 | 1-81-0 | 1.2% | 0.2-6.6% | 1 / 0 | about -760 |

No illegal moves, crashes or timeouts on either side. Elo differences use the
logistic formula on the score; the last row rests on one win and is very
uncertain (roughly -480 to -1000).

## Interpretation

- az-r7 at 1 s per move plays like Rapfi limited to about 100-150 nodes per
  move (well under a millisecond of its search). Balanced openings matter:
  with random openings both bots mostly won with Black regardless of
  strength.
- Against single-threaded Rapfi at 1 s per move, az-r7 is roughly 750 Elo
  weaker (very uncertain).
- Mapping to the Gomocup scale is rough: Rapfi's 2791 was earned at
  tournament time controls with more threads, so 1-second single-thread Rapfi
  is somewhat weaker. If it is around 2500-2800, az-r7 would sit near
  1750-2050: roughly the level of the middle of the Gomocup Standard list
  (around Hewer 2015 at 1818, below Yixin 2018 at 2200). Treat this as an
  order of magnitude, not a rating: it comes from one opponent family, the
  scales are not calibrated against each other, and Elo is not transitive
  across very different engines.

## Next decision

- Use Rapfi at 100 and 300 nodes on these 41 openings as a fixed external
  benchmark for future champions (about 40 minutes per check).
- For a better Gomocup-scale estimate, play mid-list engines directly (most
  are Windows executables and would need Wine) or calibrate limited Rapfi
  against full Rapfi.
