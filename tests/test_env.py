import jax
import jax.numpy as jnp
import pytest

from connectfive import BOARD_SIZE, NUM_ACTIONS, ConnectFive


@pytest.fixture
def env():
    return ConnectFive()


def play(env, actions):
    state = env.init(jax.random.PRNGKey(0))
    for action in actions:
        state = env.step(state, jnp.int32(action))
    return state


def test_initial_state(env):
    state = env.init(jax.random.PRNGKey(0))
    assert env.id == "connect_five"
    assert env.num_actions == NUM_ACTIONS
    assert env.observation_shape == (BOARD_SIZE, BOARD_SIZE, 2)
    assert int(state.current_player) == 0
    assert bool(state.legal_action_mask.all())
    assert not bool(state.observation.any())


@pytest.mark.parametrize(
    "black_moves",
    [
        [20, 21, 22, 23, 24],
        [20, 39, 58, 77, 96],
        [20, 40, 60, 80, 100],
        [24, 42, 60, 78, 96],
    ],
)
def test_five_in_every_direction_wins(env, black_moves):
    white_moves = [300, 302, 304, 306]
    actions = [move for pair in zip(black_moves[:-1], white_moves) for move in pair]
    actions.append(black_moves[-1])
    state = play(env, actions)
    assert bool(state.terminated)
    assert state.rewards.tolist() == [1.0, -1.0]


def test_turn_and_observation_are_current_player_relative(env):
    state = play(env, [0])
    assert int(state.current_player) == 1
    assert bool(state.observation[0, 0, 1])
    assert not bool(state.legal_action_mask[0])


def test_illegal_move_loses(env):
    state = play(env, [0, 0])
    assert bool(state.terminated)
    assert state.rewards.tolist() == [1.0, -1.0]


def test_jit_and_vmap(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    states = jax.jit(jax.vmap(env.init))(keys)
    actions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    states = jax.jit(jax.vmap(env.step))(states, actions)
    assert states.observation.shape == (4, BOARD_SIZE, BOARD_SIZE, 2)
    assert states._step_count.tolist() == [1, 1, 1, 1]

