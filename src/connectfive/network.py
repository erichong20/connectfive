"""Small, independently testable policy-value network for Connect Five."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from connectfive.env import BOARD_SIZE, NUM_ACTIONS, State


@dataclass(frozen=True)
class NetworkConfig:
    """Architecture small enough for local educational experiments."""

    residual_blocks: int = 4
    channels: int = 32
    value_hidden: int = 64
    # 3 = own, opponent, black-to-move. 4 adds a constant plane of ones so the
    # zero-padded convolutions can tell the board edge from an empty cell even
    # when white is to move (the black-to-move plane is then all zeros).
    input_planes: int = 3


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        residual = inputs
        x = nn.Conv(self.channels, (3, 3), use_bias=False)(inputs)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), use_bias=False)(x)
        x = nn.LayerNorm()(x)
        return nn.relu(residual + x)


class PolicyValueNetwork(nn.Module):
    """Residual trunk with separate policy and scalar value heads."""

    config: NetworkConfig = NetworkConfig()

    @nn.compact
    def __call__(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = features.astype(jnp.float32)
        x = nn.Conv(self.config.channels, (3, 3), use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        for _ in range(self.config.residual_blocks):
            x = ResidualBlock(self.config.channels)(x)

        policy = nn.Conv(2, (1, 1), use_bias=False, name="policy_conv")(x)
        policy = nn.relu(policy)
        logits = nn.Dense(1, name="policy_logits")(policy).reshape(
            features.shape[0], NUM_ACTIONS
        )

        value = nn.Conv(4, (1, 1), use_bias=False, name="value_conv")(x)
        value = nn.relu(nn.LayerNorm(name="value_norm")(value))
        value = value.reshape(features.shape[0], 4 * NUM_ACTIONS)
        value = nn.relu(nn.Dense(self.config.value_hidden, name="value_hidden")(value))
        value = nn.Dense(1, name="value_output")(value)
        return logits, jnp.tanh(value[:, 0])


def encode_state(state: State, planes: int = 3) -> jax.Array:
    """Encode stones relative to the mover plus an absolute color plane."""

    player = state.current_player
    own = state._board == player
    opponent = state._board == (1 - player)
    black_to_move = jnp.full_like(own, player == 0)
    stacked = [own, opponent, black_to_move]
    if planes == 4:
        stacked.append(jnp.ones_like(own))
    return jnp.stack(stacked, axis=-1).astype(jnp.float32)


def encode_board(board: np.ndarray, player: int, planes: int = 3) -> np.ndarray:
    """NumPy twin of ``encode_state`` for an environment board array."""

    own = board == player
    opponent = board == (1 - player)
    stacked = [own, opponent, np.full_like(own, player == 0)]
    if planes == 4:
        stacked.append(np.ones_like(own))
    return np.stack(stacked, axis=-1).astype(np.float32)


def with_input_planes(features: np.ndarray, planes: int) -> np.ndarray:
    """Adapt stored three-plane features to a network's input plane count."""

    if features.shape[-1] == planes:
        return features
    if features.shape[-1] == 3 and planes == 4:
        ones = np.ones(features.shape[:-1] + (1,), dtype=features.dtype)
        return np.concatenate((features, ones), axis=-1)
    raise ValueError(f"cannot convert {features.shape[-1]} planes to {planes}")


def mask_policy_logits(logits: jax.Array, legal_action_mask: jax.Array) -> jax.Array:
    """Give illegal moves effectively zero softmax probability."""

    minimum = jnp.finfo(logits.dtype).min
    return jnp.where(legal_action_mask, logits, minimum)


def policy_value_loss(
    model: PolicyValueNetwork,
    params: Any,
    features: jax.Array,
    legal_action_mask: jax.Array,
    policy_targets: jax.Array,
    value_targets: jax.Array,
    value_weight: float = 1.0,
    value_sample_weights: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits, values = model.apply(params, features)
    logits = mask_policy_logits(logits, legal_action_mask)
    policy_loss = optax.softmax_cross_entropy(logits, policy_targets).mean()
    squared = jnp.square(values - value_targets)
    if value_sample_weights is None:
        value_loss = squared.mean()
    else:
        # Weights reshape only the value loss; every position still trains the policy.
        value_loss = (squared * value_sample_weights).sum() / value_sample_weights.sum()
    total = policy_loss + value_weight * value_loss
    return total, {
        "loss": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }


def make_train_step(
    model: PolicyValueNetwork,
    optimizer: optax.GradientTransformation,
    value_weight: float = 1.0,
):
    """Create one compiled supervised policy/value update."""

    @jax.jit
    def train_step(
        params: Any,
        opt_state: Any,
        features: jax.Array,
        legal_action_mask: jax.Array,
        policy_targets: jax.Array,
        value_targets: jax.Array,
        value_sample_weights: jax.Array | None = None,
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        def objective(current_params):
            return policy_value_loss(
                model,
                current_params,
                features,
                legal_action_mask,
                policy_targets,
                value_targets,
                value_weight,
                value_sample_weights,
            )

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    return train_step


def transform_features(features: jax.Array, symmetry: int) -> jax.Array:
    """Apply one of the square's eight dihedral symmetries."""

    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be between 0 and 7")
    transformed = jnp.swapaxes(features, -3, -2) if symmetry >= 4 else features
    return jnp.rot90(transformed, symmetry % 4, axes=(-3, -2))


def transform_policy(policy: jax.Array, symmetry: int) -> jax.Array:
    """Apply the same board symmetry to flat policy or legal-mask vectors."""

    if policy.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"policy's final dimension must be {NUM_ACTIONS}")
    board = policy.reshape(*policy.shape[:-1], BOARD_SIZE, BOARD_SIZE)
    if symmetry >= 4:
        board = jnp.swapaxes(board, -2, -1)
    board = jnp.rot90(board, symmetry % 4, axes=(-2, -1))
    return board.reshape(policy.shape)


def count_parameters(params: Any) -> int:
    return sum(value.size for value in jax.tree.leaves(params))


@dataclass(frozen=True)
class LoadedCheckpoint:
    config: NetworkConfig
    params: Any
    step: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class NetworkAgent:
    """Greedy raw-policy agent, without search."""

    params: Any
    config: NetworkConfig
    name: str = "network"

    def select_action(self, state: State, key: jax.Array) -> int:
        del key
        logits = _policy_logits(self.config, self.params, state)
        return int(np.asarray(logits).argmax())

    @classmethod
    def from_checkpoint(cls, path: Path, key: jax.Array) -> NetworkAgent:
        loaded = load_checkpoint(path, key)
        return cls(params=loaded.params, config=loaded.config)


@partial(jax.jit, static_argnums=0)
def _policy_logits(config: NetworkConfig, params: Any, state: State) -> jax.Array:
    """Compiled masked policy logits for one environment state."""

    features = encode_state(state, config.input_planes)[None, ...]
    logits, _ = PolicyValueNetwork(config).apply(params, features)
    return mask_policy_logits(logits, state.legal_action_mask[None, ...])[0]


def save_checkpoint(
    path: Path,
    params: Any,
    config: NetworkConfig,
    step: int,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save architecture metadata and Flax parameter bytes via file replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".json")
    params_path = path.with_suffix(".msgpack")
    metadata_temp = metadata_path.with_suffix(".json.tmp")
    params_temp = params_path.with_suffix(".msgpack.tmp")
    metadata = {
        "format_version": 1,
        "config": asdict(config),
        "step": int(step),
        "metrics": metrics or {},
    }
    metadata_temp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    params_temp.write_bytes(serialization.to_bytes(params))
    os.replace(params_temp, params_path)
    os.replace(metadata_temp, metadata_path)


def load_checkpoint(path: Path, key: jax.Array) -> LoadedCheckpoint:
    """Load parameters using the recorded architecture as the template."""

    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config = NetworkConfig(**metadata["config"])
    model = PolicyValueNetwork(config)
    template = model.init(
        key,
        jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, config.input_planes), dtype=jnp.float32),
    )
    params = serialization.from_bytes(template, path.with_suffix(".msgpack").read_bytes())
    return LoadedCheckpoint(
        config=config,
        params=params,
        step=int(metadata["step"]),
        metrics={key: float(value) for key, value in metadata["metrics"].items()},
    )
