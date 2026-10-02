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


def test_selfplay_records_visit_targets_and_replays(tmp_path):
    from connectfive.dataset import SupervisedDataset, verify_record
    from connectfive.network import save_checkpoint
    from connectfive.selfplay import SelfPlayConfig, generate_selfplay_games
    from connectfive.teacher import game_record, teacher_games_to_arrays

    config = NetworkConfig(residual_blocks=1, channels=8, value_hidden=16, input_planes=4)
    params = PolicyValueNetwork(config).init(
        jax.random.PRNGKey(0), jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, 4))
    )
    save_checkpoint(tmp_path / "model", params, config, step=0)
    selfplay = SelfPlayConfig(checkpoint=str(tmp_path / "model"), simulations=8)
    games = generate_selfplay_games([1], selfplay, workers=1)
    verify_record(game_record(games[0], selfplay))
    dataset = SupervisedDataset(**teacher_games_to_arrays(games, selfplay))
    sums = dataset.policy_targets.astype("float32").sum(axis=1)
    assert abs(sums - 1).max() < 1e-2
    assert (dataset.legal_action_masks | (dataset.policy_targets == 0)).all()


def test_balanced_opening_is_deterministic_and_respects_threshold():
    from connectfive.patterns import ACTION_TO_INDEX, PatternBoard
    from connectfive.selfplay import SelfPlayConfig, choose_opening

    def fake(board):
        return None, 0.9 if len(board.moves) and board.moves[0] % 2 else 0.1

    loose = SelfPlayConfig(checkpoint="x")
    strict = SelfPlayConfig(checkpoint="x", balance_threshold=0.2, balance_attempts=64)
    assert choose_opening(5, loose, fake) == choose_opening(5, loose, fake)
    opening = choose_opening(5, strict, fake)
    assert opening == choose_opening(5, strict, fake)
    board = PatternBoard()
    for action in opening:
        board.play(ACTION_TO_INDEX[action])
    assert fake(board)[1] <= 0.2
