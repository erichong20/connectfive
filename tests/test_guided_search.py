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


def test_selfplay_draw_ply_cap_records_a_replayable_draw(tmp_path):
    from connectfive.dataset import verify_record
    from connectfive.network import save_checkpoint
    from connectfive.selfplay import SelfPlayConfig, generate_selfplay_games
    from connectfive.teacher import game_record

    config = NetworkConfig(residual_blocks=1, channels=8, value_hidden=16, input_planes=4)
    params = PolicyValueNetwork(config).init(
        jax.random.PRNGKey(0), jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, 4))
    )
    save_checkpoint(tmp_path / "model", params, config, step=0)
    selfplay = SelfPlayConfig(checkpoint=str(tmp_path / "model"), simulations=4, draw_ply_cap=6)
    game = generate_selfplay_games([3], selfplay, workers=1)[0]
    assert len(game.moves) == 6 and game.winner is None and game.adjudicated == "ply_cap"
    assert len(game.positions) == 2
    verify_record(game_record(game, selfplay))


def test_selfplay_log_resumes_without_replaying_finished_games(tmp_path):
    from connectfive.network import save_checkpoint
    from connectfive.selfplay import SelfPlayConfig, generate_selfplay_games, load_game_log

    config = NetworkConfig(residual_blocks=1, channels=8, value_hidden=16, input_planes=4)
    params = PolicyValueNetwork(config).init(
        jax.random.PRNGKey(0), jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, 4))
    )
    save_checkpoint(tmp_path / "model", params, config, step=0)
    selfplay = SelfPlayConfig(checkpoint=str(tmp_path / "model"), simulations=4, draw_ply_cap=10)
    log = tmp_path / "games.jsonl"
    first = generate_selfplay_games([1, 2], selfplay, log=log)
    with log.open("a") as handle:
        handle.write('{"torn": ')  # an interrupted write
    again = generate_selfplay_games([1, 2, 3], selfplay, log=log)
    assert again[:2] == first
    assert sorted(load_game_log(log)) == [1, 2, 3]
    assert len(log.read_text().splitlines()) == 4  # only seed 3 was played again


def test_playout_cap_records_only_full_searches(tmp_path):
    from connectfive.dataset import verify_record
    from connectfive.network import save_checkpoint
    from connectfive.selfplay import SelfPlayConfig, generate_selfplay_games
    from connectfive.teacher import game_record

    config = NetworkConfig(residual_blocks=1, channels=8, value_hidden=16, input_planes=4)
    params = PolicyValueNetwork(config).init(
        jax.random.PRNGKey(0), jnp.zeros((1, BOARD_SIZE, BOARD_SIZE, 4))
    )
    save_checkpoint(tmp_path / "model", params, config, step=0)
    base = {"checkpoint": str(tmp_path / "model"), "simulations": 6, "draw_ply_cap": 30}
    full = generate_selfplay_games([5], SelfPlayConfig(**base))[0]
    capped = generate_selfplay_games(
        [5], SelfPlayConfig(**base, full_search_fraction=0.3, fast_simulations=2)
    )[0]
    assert len(full.positions) == len(full.moves) - len(full.opening_moves)
    assert 0 < len(capped.positions) < len(capped.moves) - len(capped.opening_moves)
    verify_record(game_record(capped, SelfPlayConfig(**base)))


def test_batched_search_is_legal_counts_exactly_and_clears_virtual_loss(evaluator):
    from connectfive.guided_search import GuidedMCTS
    from connectfive.match import generate_opening
    from connectfive.patterns import ACTION_TO_INDEX, PatternBoard

    board = PatternBoard()
    for action in generate_opening(11, 6):
        board.play(ACTION_TO_INDEX[action])
    before = (board.hash, list(board.moves))
    search = GuidedMCTS(board, evaluator, time_limit=1e9, max_simulations=50, batch_size=4)
    result = search.run()
    assert (board.hash, list(board.moves)) == before
    assert search.simulations == 50
    # Root children's visits add up to completed simulations: no virtual loss left.
    assert sum(visits for _, visits in result.root_scores) == 50
    assert result.actions and all(board.cells[ACTION_TO_INDEX[a]] == 0 for a in result.actions)


def test_native_core_matches_python_batched_search(evaluator):
    from connectfive.guided_search import GuidedMCTS
    from connectfive.match import generate_opening
    from connectfive.native import BatchEvaluator, NativeCore, NativeMCTS
    from connectfive.patterns import ACTION_TO_INDEX, PatternBoard

    core = NativeCore(planes=evaluator.config.input_planes)
    for batch in (1, 4):
        batched = BatchEvaluator.from_evaluator(evaluator, batch)
        for seed in range(4):
            board = PatternBoard()
            for action in generate_opening(300 + seed, 6):
                board.play(ACTION_TO_INDEX[action])
            python = GuidedMCTS(board, evaluator, time_limit=1e9, max_simulations=60,
                                batch_size=batch).run()
            native = NativeMCTS(board, batched, time_limit=1e9, max_simulations=60, core=core).run()
            assert native.actions[:1] == python.actions[:1]
            assert dict(native.root_scores) == dict(python.root_scores)
