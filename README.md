# Connect Five in JAX

A small, JAX-native implementation of freestyle Connect Five (Gomoku) on the
standard 19x19 Go board. The environment follows the
[PGX](https://github.com/sotetsuk/pgx) API and works with `jax.jit` and
`jax.vmap`.

PGX includes 19x19 Go and Connect Four, but does not currently include Gomoku,
so this repository supplies the missing environment without forking PGX.

## Rules

- Black (player 0) moves first; players alternate placing one stone.
- An action is a row-major board index: `row * 19 + column`.
- Five **or more** adjacent stones horizontally, vertically, or diagonally win.
- A full board without a winner is a draw.
- There are no Renju forbidden-move or capture rules.

## Install and play

Python 3.10-3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
connectfive                 # two humans
connectfive --bot random    # play Black against a random bot
connectfive --bot random --human-color white  # bot is Black and moves first
```

Enter Go coordinates such as `K10` (the letter I is skipped), or `quit`. Use
`--seed 42` (or any integer) when you want the random bot to replay the same
sequence of choices.

## Use as a JAX/PGX environment

```python
import jax
from connectfive import ConnectFive

env = ConnectFive()
state = env.init(jax.random.PRNGKey(0))
state = jax.jit(env.step)(state, 9 * 19 + 9)

print(state.observation.shape)   # (19, 19, 2)
print(state.legal_action_mask.shape)  # (361,)
```

The two observation planes are the stones belonging to the observing player
and the opponent's stones, respectively. Rewards are `[1, -1]` for a black
win, `[-1, 1]` for a white win, and `[0, 0]` otherwise.

Batching follows the usual PGX pattern:

```python
keys = jax.random.split(jax.random.PRNGKey(0), 256)
states = jax.jit(jax.vmap(env.init))(keys)
actions = jax.numpy.full(256, 180, dtype=jax.numpy.int32)
states = jax.jit(jax.vmap(env.step))(states, actions)
```

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
```
