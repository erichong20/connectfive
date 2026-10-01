import jax
import numpy as np
import pytest

from connectfive import NegamaxAgent, TacticalAgent, ThreatSearchAgent
from connectfive.search import NegamaxSearch, ThreatNegamaxSearch, _threat_attack_score
from connectfive.tactical_suite import load_tactical_suite, position_state


@pytest.mark.parametrize("position", load_tactical_suite(), ids=lambda case: case.name)
@pytest.mark.parametrize(
    "agent", [TacticalAgent(), NegamaxAgent(), ThreatSearchAgent()]
)
def test_agent_solves_essential_tactical_position(position, agent):
    selected = agent.select_action(position_state(position), jax.random.PRNGKey(0))
    assert selected in position.acceptable_actions


def test_negamax_search_respects_node_budget_and_reports_work():
    position = load_tactical_suite()[6]
    agent = NegamaxAgent(max_depth=4, candidate_width=10, node_budget=80)
    selected = agent.select_action(position_state(position), jax.random.PRNGKey(0))
    assert selected in position.acceptable_actions
    assert agent.last_search is not None
    assert agent.last_search.stats.nodes <= agent.node_budget
    assert agent.last_search.stats.completed_depth >= 1
    assert agent.last_search.root_scores


def test_negamax_leaf_evaluation_is_antisymmetric():
    position = load_tactical_suite()[7]
    state = position_state(position)
    board = np.asarray(state._board)
    legal = np.flatnonzero(np.asarray(state.legal_action_mask))
    search = NegamaxSearch(max_depth=1, candidate_width=10, node_budget=100)
    black_value = search._evaluate(board, 0, legal)
    white_value = search._evaluate(board, 1, legal)
    assert black_value == -white_value


def test_threat_search_identifies_cross_fork_as_forcing_defense():
    position = load_tactical_suite()[6]
    state = position_state(position)
    board = np.asarray(state._board)
    legal = np.flatnonzero(np.asarray(state.legal_action_mask))
    search = ThreatNegamaxSearch(3, 10, 250, extension_depth=2, forcing_width=8)
    moves, must_respond = search._forcing_moves(board, position.player, legal)
    assert not must_respond
    assert position.acceptable_actions[0] in moves


def test_broken_pattern_scoring_rewards_noncontiguous_threat():
    position = load_tactical_suite()[7]
    state = position_state(position)
    board = np.asarray(state._board)
    action = position.acceptable_actions[0]
    assert _threat_attack_score(board, action, position.player) > 0
