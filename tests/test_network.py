from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

pytest.importorskip("flax")

from connectfive import BOARD_SIZE, NUM_ACTIONS, ConnectFive
from connectfive.network import (
    NetworkAgent,
    NetworkConfig,
    PolicyValueNetwork,
    count_parameters,
    encode_state,
    load_checkpoint,
    make_train_step,
    mask_policy_logits,
    policy_value_loss,
    save_checkpoint,
    transform_features,
    transform_policy,
)


def make_model(blocks=1, channels=8):
    config = NetworkConfig(residual_blocks=blocks, channels=channels, value_hidden=16)
    return config, PolicyValueNetwork(config)


def test_network_shapes_and_small_parameter_count():
    config = NetworkConfig()
    model = PolicyValueNetwork(config)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, BOARD_SIZE, BOARD_SIZE, 3), dtype=jnp.float32),
    )
    logits, values = model.apply(
        params, jnp.zeros((2, BOARD_SIZE, BOARD_SIZE, 3), dtype=jnp.float32)
    )
    assert logits.shape == (2, NUM_ACTIONS)
    assert values.shape == (2,)
    assert count_parameters(params) < 200_000
    assert config == NetworkConfig()


def test_encoding_is_from_current_players_perspective():
    env = ConnectFive()
    state = env.init(jax.random.PRNGKey(0))
    state = env.step(state, 7 * BOARD_SIZE + 7)
    features = encode_state(state)
    assert features[7, 7, 0] == 0
    assert features[7, 7, 1] == 1
    assert jnp.all(features[..., 2] == 0)


def test_illegal_action_mask_removes_softmax_probability():
    logits = jnp.zeros((1, NUM_ACTIONS))
    legal = jnp.ones((1, NUM_ACTIONS), dtype=jnp.bool_).at[0, 3].set(False)
    probabilities = jax.nn.softmax(mask_policy_logits(logits, legal))
    assert probabilities[0, 3] == 0
    assert jnp.isclose(probabilities.sum(), 1.0)


def test_network_agent_always_selects_a_legal_move():
    env = ConnectFive()
    state = env.init(jax.random.PRNGKey(0))
    state = env.step(state, 0)
    config, model = make_model()
    params = model.init(jax.random.PRNGKey(1), encode_state(state)[None, ...])
    action = NetworkAgent(params, config).select_action(state, jax.random.PRNGKey(2))
    assert action != 0
    assert bool(state.legal_action_mask[action])


@pytest.mark.parametrize("symmetry", range(8))
def test_policy_and_features_use_the_same_symmetry(symmetry):
    action = 2 * BOARD_SIZE + 5
    features = jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, 3)).at[0, 2, 5, 0].set(1)
    policy = jnp.zeros((1, NUM_ACTIONS)).at[0, action].set(1)
    transformed_features = transform_features(features, symmetry)
    transformed_policy = transform_policy(policy, symmetry).reshape(
        1, BOARD_SIZE, BOARD_SIZE
    )
    assert jnp.array_equal(transformed_features[0, ..., 0], transformed_policy[0])


def test_checkpoint_round_trip(tmp_path: Path):
    config, model = make_model()
    key = jax.random.PRNGKey(4)
    params = model.init(key, jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, 3)))
    path = tmp_path / "tiny_network"
    save_checkpoint(path, params, config, step=12, metrics={"loss": 0.25})
    loaded = load_checkpoint(path, key)
    assert loaded.config == config
    assert loaded.step == 12
    assert loaded.metrics == {"loss": 0.25}
    assert all(jax.tree.leaves(jax.tree.map(np.array_equal, params, loaded.params)))


def test_tiny_batch_training_reduces_loss():
    _, model = make_model()
    key = jax.random.PRNGKey(8)
    actions = jnp.array([0, 20, 100, 224])
    features = jnp.zeros((4, BOARD_SIZE, BOARD_SIZE, 3), dtype=jnp.float32)
    features = features.at[jnp.arange(4), actions // BOARD_SIZE, actions % BOARD_SIZE, 0].set(1)
    legal = jnp.ones((4, NUM_ACTIONS), dtype=jnp.bool_)
    targets = jax.nn.one_hot(actions, NUM_ACTIONS)
    values = jnp.array([-1.0, -0.5, 0.5, 1.0])
    params = model.init(key, features)
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer)
    first = None
    last = None
    for _ in range(80):
        params, opt_state, metrics = train_step(
            params, opt_state, features, legal, targets, values
        )
        loss = float(metrics["loss"])
        first = loss if first is None else first
        last = loss
    assert last is not None and first is not None
    assert last < first * 0.35


def test_numpy_encoding_matches_jax_encoding_and_edge_plane():
    from connectfive.network import encode_board, with_input_planes

    env = ConnectFive()
    state = env.init(jax.random.PRNGKey(0))
    for action in (112, 0, 50):
        state = env.step(state, action)
    board = np.asarray(state._board)
    player = int(state.current_player)
    for planes in (3, 4):
        assert np.array_equal(
            encode_board(board, player, planes), np.asarray(encode_state(state, planes))
        )
    four = with_input_planes(encode_board(board, player)[None], 4)
    assert np.all(four[..., 3] == 1)


def test_four_plane_checkpoint_round_trip(tmp_path: Path):
    env = ConnectFive()
    state = env.step(env.init(jax.random.PRNGKey(0)), 112)
    config = NetworkConfig(residual_blocks=1, channels=8, value_hidden=16, input_planes=4)
    model = PolicyValueNetwork(config)
    params = model.init(jax.random.PRNGKey(1), encode_state(state, 4)[None, ...])
    save_checkpoint(tmp_path / "model", params, config, step=1)
    agent = NetworkAgent.from_checkpoint(tmp_path / "model", jax.random.PRNGKey(0))
    assert agent.config.input_planes == 4
    assert bool(state.legal_action_mask[agent.select_action(state, jax.random.PRNGKey(0))])


def test_value_sample_weights_change_only_the_value_loss():
    config = NetworkConfig(residual_blocks=1, channels=8, value_hidden=16)
    model = PolicyValueNetwork(config)
    features = jnp.zeros((2, BOARD_SIZE, BOARD_SIZE, 3))
    params = model.init(jax.random.PRNGKey(0), features)
    legal = jnp.ones((2, NUM_ACTIONS), dtype=bool)
    policy = jnp.full((2, NUM_ACTIONS), 1 / NUM_ACTIONS)
    targets = jnp.array([1.0, -1.0])
    _, plain = policy_value_loss(model, params, features, legal, policy, targets)
    _, ones = policy_value_loss(model, params, features, legal, policy, targets,
                                value_sample_weights=jnp.ones(2))
    _, first = policy_value_loss(model, params, features, legal, policy, targets,
                                 value_sample_weights=jnp.array([1.0, 0.0]))
    assert jnp.allclose(plain["value_loss"], ones["value_loss"])
    assert jnp.allclose(plain["policy_loss"], first["policy_loss"])
    value = model.apply(params, features)[1][0]
    assert jnp.allclose(first["value_loss"], (value - 1.0) ** 2)
