"""AlphaZero-style self-play training for Connect Five.

A single-accelerator sketch: a fully convolutional policy/value ResNet, Gumbel
MuZero search from ``mctx``, randomized openings (gomoku is a first-player win,
so fixed openings collapse into "Black always wins"), a FIFO replay buffer,
and 8-way dihedral symmetry augmentation.

    pip install -e '.[train]'
    python scripts/train_alphazero.py --tiny          # CPU smoke test
    python scripts/train_alphazero.py --out runs/az   # real run on a GPU
"""

import argparse
import collections
import dataclasses
import pickle
import time
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
from pgx.experimental import auto_reset

from connectfive import BOARD_SIZE, NUM_ACTIONS, ConnectFive

env = ConnectFive()


@dataclasses.dataclass(frozen=True)
class Config:
    num_blocks: int = 10
    channels: int = 128
    selfplay_batch: int = 1024  # games played in parallel
    rollout_steps: int = 256  # moves per game slot per iteration
    num_simulations: int = 64
    opening_radius: int = 4  # random opening stones land within this distance of center
    max_opening_moves: int = 6  # must stay below 9 so an opening can never win
    buffer_iterations: int = 4  # keep samples from this many recent iterations
    train_batch: int = 1024
    updates_per_iteration: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 5
    eval_games: int = 256
    iterations: int = 1000
    seed: int = 0


TINY = {
    "num_blocks": 2,
    "channels": 32,
    "selfplay_batch": 8,
    "rollout_steps": 200,
    "num_simulations": 8,
    "train_batch": 64,
    "updates_per_iteration": 4,
    "eval_every": 1,
    "eval_games": 8,
    "iterations": 3,
}


# --- Network -----------------------------------------------------------------


class ResBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.channels, (3, 3))(nn.relu(nn.LayerNorm()(x)))
        y = nn.Conv(self.channels, (3, 3))(nn.relu(nn.LayerNorm()(y)))
        return x + y


class PolicyValueNet(nn.Module):
    """Fully convolutional, so the same weights run on any board size."""

    num_blocks: int
    channels: int

    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32)
        # A constant plane lets zero-padded convolutions tell the edge from empty points.
        x = jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)
        x = nn.Conv(self.channels, (3, 3))(x)
        for _ in range(self.num_blocks):
            x = ResBlock(self.channels)(x)
        x = nn.relu(nn.LayerNorm()(x))

        logits = nn.Conv(1, (1, 1))(x).reshape(x.shape[0], -1)
        # Max-pooling lets a single threat anywhere on the board swing the value.
        pooled = jnp.concatenate([x.mean(axis=(1, 2)), x.max(axis=(1, 2))], axis=-1)
        value = nn.Dense(1)(nn.relu(nn.Dense(self.channels)(pooled)))
        return logits, jnp.tanh(value[..., 0])


# --- Openings ------------------------------------------------------------------


def make_opening_init(cfg: Config):
    """Start each game from 0..max_opening_moves random stones near the center."""
    assert cfg.max_opening_moves < 9
    near = jnp.abs(jnp.arange(BOARD_SIZE) - BOARD_SIZE // 2) <= cfg.opening_radius
    center = (near[:, None] & near[None, :]).reshape(-1)

    def init(key):
        key, count_key = jax.random.split(key)
        state = env.init(key)
        num_moves = jax.random.randint(count_key, (), 0, cfg.max_opening_moves + 1)

        def body(i, carry):
            state, key = carry
            key, sub = jax.random.split(key)
            allowed = state.legal_action_mask & center
            action = jax.random.categorical(sub, jnp.where(allowed, 0.0, -jnp.inf))
            stepped = env.step(state, action)
            state = jax.tree.map(lambda a, b: jnp.where(i < num_moves, a, b), stepped, state)
            return state, key

        state, _ = jax.lax.fori_loop(0, cfg.max_opening_moves, body, (state, key))
        return state

    return init


# --- Self-play -----------------------------------------------------------------


class Samples(NamedTuple):
    obs: np.ndarray  # (N, size, size, 2) bool, from the side to move
    policy: np.ndarray  # (N, size * size) search visit distribution
    value: np.ndarray  # (N,) final result from the side to move


def make_selfplay(cfg: Config, net: PolicyValueNet):
    opening_init = make_opening_init(cfg)
    step_with_reset = jax.vmap(auto_reset(env.step, opening_init))
    batch = jnp.arange(cfg.selfplay_batch)

    def recurrent_fn(params, key, action, state):
        del key
        player = state.current_player
        state = jax.vmap(env.step)(state, action)
        logits, value = net.apply(params, state.observation)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        output = mctx.RecurrentFnOutput(
            reward=state.rewards[batch, player],
            discount=jnp.where(state.terminated, 0.0, -1.0),  # zero-sum: flip sign each ply
            prior_logits=logits,
            value=jnp.where(state.terminated, 0.0, value),
        )
        return output, state

    def move(params, state, key):
        search_key, reset_key = jax.random.split(key)
        # auto_reset leaves `terminated` set on a freshly reset game; clear it so the
        # search doesn't treat the new root as already finished.
        root_state = state.replace(terminated=jnp.zeros_like(state.terminated))
        logits, value = net.apply(params, root_state.observation)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=root_state)
        out = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=cfg.num_simulations,
            invalid_actions=~root_state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
        )
        player = root_state.current_player
        keys = jax.random.split(reset_key, cfg.selfplay_batch)
        next_state = step_with_reset(root_state, out.action, keys)
        record = (
            root_state.observation,
            out.action_weights,
            next_state.rewards[batch, player],
            next_state.terminated,
            player,
        )
        return next_state, record

    @jax.jit
    def selfplay(params, state, key):
        keys = jax.random.split(key, cfg.rollout_steps)
        state, (obs, policy, reward, terminated, player) = jax.lax.scan(
            lambda s, k: move(params, s, k), state, keys
        )

        # Back up final results: v_t = r_t, or -v_{t+1} if the game continued.
        def backup(next_value, xs):
            r, done = xs
            v = r + jnp.where(done, 0.0, -next_value)
            return v, v

        _, value = jax.lax.scan(
            backup, jnp.zeros(cfg.selfplay_batch), (reward, terminated), reverse=True
        )
        # Moves after a slot's last terminal belong to a game still in progress.
        finished = jnp.cumsum(terminated[::-1], axis=0)[::-1] >= 1
        stats = {
            "games": terminated.sum(),
            "black_wins": (terminated & (reward > 0) & (player == 0)).sum(),
            "draws": (terminated & (reward == 0)).sum(),
        }
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        return state, (flat(obs), flat(policy), flat(value), flat(finished)), stats

    return opening_init, selfplay


# --- Training ------------------------------------------------------------------

# The 8 symmetries of the square, applied to arrays shaped (batch, row, col, ...).
_SYMMETRIES = [
    lambda x, k=k, t=t: jnp.rot90(jnp.swapaxes(x, 1, 2) if t else x, k, axes=(1, 2))
    for t in (False, True)
    for k in range(4)
]


def augment(key, obs, policy):
    i = jax.random.randint(key, (), 0, len(_SYMMETRIES))
    obs = jax.lax.switch(i, _SYMMETRIES, obs)
    policy = jax.lax.switch(i, _SYMMETRIES, policy.reshape(-1, BOARD_SIZE, BOARD_SIZE))
    return obs, policy.reshape(-1, NUM_ACTIONS)


def make_train_step(net: PolicyValueNet, optimizer):
    def loss_fn(params, obs, policy, value, key):
        obs, policy = augment(key, obs, policy)
        logits, pred = net.apply(params, obs)
        policy_loss = optax.softmax_cross_entropy(logits, policy).mean()
        value_loss = optax.l2_loss(pred, value).mean()
        return policy_loss + value_loss, (policy_loss, value_loss)

    @jax.jit
    def train_step(params, opt_state, obs, policy, value, key):
        grads, losses = jax.grad(loss_fn, has_aux=True)(params, obs, policy, value, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, losses

    return train_step


def make_evaluate(cfg: Config, net: PolicyValueNet, opening_init):
    """Greedy (no search) match: new params vs baseline, each side half the time."""
    n = cfg.eval_games
    games = jnp.arange(n)
    new_player = (games >= n // 2).astype(jnp.int32)

    @jax.jit
    def evaluate(params, baseline, key):
        state = jax.vmap(opening_init)(jax.random.split(key, n))

        def body(carry, _):
            state, score = carry
            new_logits, _ = net.apply(params, state.observation)
            old_logits, _ = net.apply(baseline, state.observation)
            mine = (state.current_player == new_player)[:, None]
            logits = jnp.where(mine, new_logits, old_logits)
            action = jnp.argmax(jnp.where(state.legal_action_mask, logits, -jnp.inf), axis=-1)
            state = jax.vmap(env.step)(state, action)
            return (state, score + state.rewards[games, new_player]), None

        (_, score), _ = jax.lax.scan(body, (state, jnp.zeros(n)), None, length=NUM_ACTIONS)
        return score  # +1 win, 0 draw, -1 loss for the new params

    return evaluate


def train(cfg: Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    net = PolicyValueNet(cfg.num_blocks, cfg.channels)
    optimizer = optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay)
    opening_init, selfplay = make_selfplay(cfg, net)
    train_step = make_train_step(net, optimizer)
    evaluate = make_evaluate(cfg, net, opening_init)

    key, init_key, env_key = jax.random.split(jax.random.PRNGKey(cfg.seed), 3)
    params = net.init(init_key, jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, 2), jnp.bool_))
    opt_state = optimizer.init(params)
    baseline = params
    state = jax.vmap(opening_init)(jax.random.split(env_key, cfg.selfplay_batch))
    buffer: collections.deque[Samples] = collections.deque(maxlen=cfg.buffer_iterations)
    rng = np.random.default_rng(cfg.seed)

    for iteration in range(1, cfg.iterations + 1):
        start = time.time()
        key, play_key = jax.random.split(key)
        state, (obs, policy, value, finished), stats = selfplay(params, state, play_key)
        keep = np.asarray(finished)
        buffer.append(
            Samples(np.asarray(obs)[keep], np.asarray(policy)[keep], np.asarray(value)[keep])
        )
        data = Samples(*(np.concatenate(parts) for parts in zip(*buffer)))
        play_time = time.time() - start

        losses = []
        # Early on, games can outlast a whole rollout, leaving nothing to train on yet.
        for _ in range(cfg.updates_per_iteration if len(data.value) else 0):
            idx = rng.integers(0, len(data.value), cfg.train_batch)
            key, step_key = jax.random.split(key)
            params, opt_state, loss = train_step(
                params, opt_state, data.obs[idx], data.policy[idx], data.value[idx], step_key
            )
            losses.append(loss)
        policy_loss, value_loss = (
            np.mean(jax.device_get(losses), axis=0) if losses else (np.nan, np.nan)
        )

        games = int(stats["games"])
        print(
            f"iter {iteration:4d} | games {games:6d} "
            f"black {int(stats['black_wins']) / max(games, 1):.0%} "
            f"draws {int(stats['draws'])} | buffer {len(data.value):8d} "
            f"| policy {policy_loss:.3f} value {value_loss:.3f} "
            f"| play {play_time:.0f}s total {time.time() - start:.0f}s",
            flush=True,
        )

        if iteration % cfg.eval_every == 0:
            key, eval_key = jax.random.split(key)
            score = np.asarray(evaluate(params, baseline, eval_key))
            print(
                f"  eval vs iter {iteration - cfg.eval_every}: "
                f"W {(score > 0).mean():.0%} D {(score == 0).mean():.0%} "
                f"L {(score < 0).mean():.0%}",
                flush=True,
            )
            baseline = params
            with open(out_dir / f"params_{iteration:05d}.pkl", "wb") as f:
                pickle.dump(jax.device_get(params), f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/alphazero"))
    parser.add_argument("--tiny", action="store_true", help="small settings for a CPU smoke test")
    for field in dataclasses.fields(Config):
        flag = "--" + field.name.replace("_", "-")
        parser.add_argument(flag, type=field.type, default=None)
    args = parser.parse_args()

    overrides = dict(TINY) if args.tiny else {}
    overrides |= {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(Config)
        if getattr(args, f.name) is not None
    }
    cfg = Config(**overrides)
    print(cfg, f"devices={jax.devices()}", flush=True)
    train(cfg, args.out)


if __name__ == "__main__":
    main()
