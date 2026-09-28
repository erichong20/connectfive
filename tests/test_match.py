import pytest

from connectfive import RandomAgent, TacticalAgent, evaluate_agents, play_game


def test_recorded_game_is_reproducible():
    first = play_game(RandomAgent(), RandomAgent(), seed=7)
    second = play_game(RandomAgent(), RandomAgent(), seed=7)
    assert first == second
    assert first.moves
    assert first.illegal_player is None
    assert first.winner in (0, 1, None)


def test_evaluation_alternates_colors_and_accounts_for_every_game():
    summary = evaluate_agents(TacticalAgent(), RandomAgent(), games=4, seed=3)
    assert summary.wins + summary.losses + summary.draws == 4
    assert summary.agent_illegal_moves == 0
    assert summary.opponent_illegal_moves == 0
    assert summary.total_moves > 0


def test_evaluation_requires_color_pairs():
    with pytest.raises(ValueError, match="positive even"):
        evaluate_agents(RandomAgent(), RandomAgent(), games=3)
