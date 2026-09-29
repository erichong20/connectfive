import jax
import jax.numpy as jnp
import numpy as np

from connectfive import BOARD_SIZE, ConnectFive, LookaheadAgent, RandomAgent, TacticalAgent


def make_state(stones, player=0):
    env = ConnectFive()
    state = env.init(jax.random.PRNGKey(0))
    board = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)
    for row, col, color in stones:
        board[row, col] = color
    return state.replace(
        current_player=jnp.int32(player),
        _board=jnp.asarray(board),
        legal_action_mask=jnp.asarray(board.reshape(-1) == -1),
        _step_count=jnp.int32(len(stones)),
    )


def action(row, col):
    return row * BOARD_SIZE + col


def test_random_agent_is_seeded_and_legal():
    state = make_state([(7, 7, 0)])
    agent = RandomAgent()
    first = agent.select_action(state, jax.random.PRNGKey(42))
    second = agent.select_action(state, jax.random.PRNGKey(42))
    assert first == second
    assert bool(state.legal_action_mask[first])


def test_tactical_agent_takes_an_immediate_exact_five():
    state = make_state([(7, col, 0) for col in range(3, 7)], player=0)
    selected = TacticalAgent().select_action(state, jax.random.PRNGKey(0))
    assert selected in {action(7, 2), action(7, 7)}


def test_tactical_agent_blocks_an_immediate_exact_five():
    state = make_state([(7, col, 1) for col in range(3, 7)], player=0)
    selected = TacticalAgent().select_action(state, jax.random.PRNGKey(0))
    assert selected in {action(7, 2), action(7, 7)}


def test_tactical_agent_wins_instead_of_blocking():
    stones = [(3, col, 0) for col in range(3, 7)]
    stones += [(10, col, 1) for col in range(3, 7)]
    state = make_state(stones, player=0)
    selected = TacticalAgent().select_action(state, jax.random.PRNGKey(0))
    assert selected in {action(3, 2), action(3, 7)}


def test_tactical_agent_does_not_mistake_an_overline_for_a_win():
    stones = [(7, col, 0) for col in (1, 2, 3, 4, 6)]
    stones += [(3, col, 0) for col in range(1, 5)]
    state = make_state(stones, player=0)
    selected = TacticalAgent().select_action(state, jax.random.PRNGKey(0))
    assert selected in {action(3, 0), action(3, 5)}
    assert selected != action(7, 5)


def test_lookahead_agent_takes_an_immediate_exact_five():
    state = make_state([(7, col, 0) for col in range(3, 7)], player=0)
    selected = LookaheadAgent().select_action(state, jax.random.PRNGKey(0))
    assert selected in {action(7, 2), action(7, 7)}


def test_lookahead_agent_is_seeded_and_legal():
    state = make_state([(7, 7, 0), (7, 8, 1), (8, 8, 0)], player=1)
    agent = LookaheadAgent()
    first = agent.select_action(state, jax.random.PRNGKey(42))
    second = agent.select_action(state, jax.random.PRNGKey(42))
    assert first == second
    assert bool(state.legal_action_mask[first])


def test_lookahead_agent_blocks_two_way_immediate_threat():
    stones = [(7, col, 1) for col in (3, 4, 5, 6)]
    state = make_state(stones, player=0)
    selected = LookaheadAgent().select_action(state, jax.random.PRNGKey(0))
    assert selected in {action(7, 2), action(7, 7)}
