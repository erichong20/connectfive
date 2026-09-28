import jax
import jax.numpy as jnp
import pytest

from connectfive import ConnectFive
from connectfive.cli import parse_move, random_action


@pytest.mark.parametrize(("text", "action"), [("A1", 0), ("K10", 144), ("P15", 224)])
def test_parse_move(text, action):
    assert parse_move(text) == action


@pytest.mark.parametrize("text", ["I5", "A0", "T15", "P16", "wat"])
def test_parse_move_rejects_invalid_coordinates(text):
    with pytest.raises(ValueError):
        parse_move(text)


def test_random_action_is_legal():
    env = ConnectFive()
    state = env.init(jax.random.PRNGKey(0))
    state = env.step(state, jnp.int32(112))
    action = random_action(state, jax.random.PRNGKey(1))
    assert action != 112
    assert bool(state.legal_action_mask[action])
