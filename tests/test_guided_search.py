import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("flax")

from connectfive import BOARD_SIZE, ConnectFive
from connectfive.guided_search import GuidedAgent, NetworkEvaluator
from connectfive.network import NetworkConfig, PolicyValueNetwork
from connectfive.tactical_suite import load_tactical_suite, position_state


@pytest.fixture(scope="module")
def evaluator():
    config = NetworkConfig(residual_blocks=1, channels=8, value_hidden=16, input_planes=4)
    params = PolicyValueNetwork(config).init(
        jax.random.PRNGKey(0), jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, 4))
    )
    return NetworkEvaluator(params, config)


@pytest.mark.parametrize("mode", ["mcts", "alphabeta", "alphabeta-value"])
def test_guided_search_keeps_exact_tactics_with_an_untrained_network(evaluator, mode):
    agent = GuidedAgent(evaluator, mode, time_limit=0.05)
    for index, position in enumerate(load_tactical_suite()):
        action = agent.select_action(position_state(position), jax.random.PRNGKey(index))
        assert action in position.acceptable_actions, position.name


def test_guided_mcts_plays_legal_moves_from_the_start(evaluator):
    env = ConnectFive()
    state = env.init(jax.random.PRNGKey(0))
    agent = GuidedAgent(evaluator, "mcts", time_limit=0.02)
    for ply in range(6):
        action = agent.select_action(state, jax.random.PRNGKey(ply))
        assert bool(state.legal_action_mask[action])
        state = env.step(state, action)
