# Connect Five in JAX

A small, JAX-native implementation of standard Gomoku on a 15x15 board. The environment follows the
[PGX](https://github.com/sotetsuk/pgx) API and works with `jax.jit` and
`jax.vmap`.

PGX includes 19x19 Go and Connect Four, but does not currently include Gomoku,
so this repository supplies the missing environment without forking PGX.

## Rules

- Black (player 0) moves first; players alternate placing one stone.
- An action is a row-major board index: `row * 15 + column`.
- Exactly five adjacent stones horizontally, vertically, or diagonally win.
- Overlines (six or more adjacent stones) are legal but do not win.
- A full board without a winner is a draw.
- There are no Renju forbidden-move or capture rules.
- Playing on an occupied intersection is illegal. Following the PGX convention,
  an illegal action ends the game immediately and the player who made it loses.

## Install and play

Requires Python 3.10 or newer. CI runs on Python 3.10 and 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
connectfive                 # two humans
connectfive --bot random    # play Black against a random bot
connectfive --bot tactical  # play against the one-ply tactical bot
connectfive --bot random --human-color white  # bot is Black and moves first
```

`python -m connectfive` works the same way as the `connectfive` command.

The board is printed with row 15 at the top; `X` is Black and `O` is White.
Enter Go coordinates such as `K10` (columns run A-P and skip the letter I), or
`quit` to stop. The terminal won't accept a move on an occupied intersection.
Use `--seed 42` (or any integer) when you want the random bot to replay the
same sequence of choices; the seed only affects the bot.

## Evaluate bots

The evaluation harness alternates colors and records wins, losses, draws,
illegal moves, game length, and elapsed time. Runs are deterministic for a
fixed seed.

```bash
python scripts/evaluate.py random random --games 1000 --seed 0
python scripts/evaluate.py tactical random --games 100 --seed 0
```

Pass `--json runs/evaluation.json` to retain a machine-readable summary. The
tactical bot is intentionally limited to one-ply pattern recognition so its
decisions and shortcomings remain understandable before search is introduced.

## Browser interface

The playable web interface lives in `web/`:

```bash
cd web
npm install
npm run dev
```

Open the local URL shown in the terminal. Choose the Tactical or Random
opponent, and choose White if you want the bot to make the first move.

## Use as a JAX/PGX environment

```python
import jax
from connectfive import ConnectFive

env = ConnectFive()
state = env.init(jax.random.PRNGKey(0))
state = jax.jit(env.step)(state, 7 * 15 + 7)

print(state.observation.shape)   # (15, 15, 2)
print(state.legal_action_mask.shape)  # (225,)
```

The two observation planes are the stones belonging to the observing player
and the opponent's stones, respectively. Rewards are `[1, -1]` for a black
win, `[-1, 1]` for a white win, and `[0, 0]` otherwise.

Batching follows the usual PGX pattern:

```python
keys = jax.random.split(jax.random.PRNGKey(0), 256)
states = jax.jit(jax.vmap(env.init))(keys)
actions = jax.numpy.full(256, 112, dtype=jax.numpy.int32)
states = jax.jit(jax.vmap(env.step))(states, actions)
```

## Train the compact network

Install the optional training dependencies, then run the fast end-to-end
overfit check:

```bash
pip install -e '.[train]'
python scripts/overfit_network.py
```

This trains the default 4-block, 32-channel policy/value network on eight fixed
tactical positions and saves a small parameter checkpoint under the ignored
`runs/` directory. It is a wiring test, not a claim about playing strength. See
`docs/neural-network.md` for the architecture and the next training milestone.

The supervised pipeline generates complete classical-agent games, splits them
by game, trains with real outcomes, and evaluates the raw policy:

```bash
python scripts/generate_dataset.py --games 32 --out runs/supervised-v1/games.npz
python scripts/train_supervised.py runs/supervised-v1/games.npz \
  --steps 1000 --batch-size 128 --out runs/supervised-v1/model
python scripts/evaluate_network.py runs/supervised-v1/model --opponent random
```

Generated games, reports, and checkpoints stay under ignored `runs/`. The first
32-game experiment beat Random but not Tactical, so it is not yet a browser bot.

## AlphaZero self-play prototype

`scripts/train_alphazero.py` is an AlphaZero-style training loop: a policy/value
ResNet, Gumbel MuZero search from [mctx](https://github.com/google-deepmind/mctx),
randomized openings, a replay buffer, and symmetry augmentation. Self-play and
search run entirely on the accelerator.

```bash
pip install -e '.[train]'
python scripts/train_alphazero.py --tiny               # quick CPU smoke test
python scripts/train_alphazero.py --out runs/az        # full run; use a GPU
```

This script remains a prototype and is not the current recommended training
path. Before a serious run it must be moved onto the tested network/checkpoint
path and fixed to preserve complete training state and trajectories across
iterations. Every flag in its `Config` can be overridden, but do not start a
long run merely because its loss decreases.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
```
