import pytest

from connectfive import (
    RandomAgent,
    TacticalAgent,
    evaluate_agents,
    evaluate_agents_with_records,
    generate_opening,
    play_game,
)


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


def test_evaluation_can_return_replayable_records():
    summary, records = evaluate_agents_with_records(
        RandomAgent(), RandomAgent(), games=2, seed=4
    )
    assert summary.games == 2
    assert len(records) == 2
    assert [record.seed for record in records] == [4, 4]
    assert all(record.moves for record in records)


def test_randomized_openings_are_reproducible_and_paired():
    opening = generate_opening(seed=12, plies=4)
    assert opening == generate_opening(seed=12, plies=4)
    assert len(opening) == len(set(opening)) == 4
    summary, records = evaluate_agents_with_records(
        RandomAgent(), RandomAgent(), games=4, seed=12, opening_plies=4
    )
    assert summary.games == 4
    assert records[0].opening_moves == records[1].opening_moves
    assert records[2].opening_moves == records[3].opening_moves
    assert records[0].opening_moves != records[2].opening_moves


def test_match_summary_reports_color_timing_and_confidence():
    summary = evaluate_agents(RandomAgent(), RandomAgent(), games=2, seed=5)
    low, high = summary.score_rate_95_ci
    assert 0 <= low <= summary.score_rate <= high <= 1
    assert summary.agent_black_games == summary.agent_white_games == 1
    assert summary.agent_move_count > 0
    assert summary.average_agent_move_ms > 0


def test_combine_summaries_adds_blocks_and_rejects_mixed_agents():

    from connectfive.match import combine_summaries, evaluate_agents

    first = evaluate_agents(RandomAgent(), RandomAgent(), 4, seed=1).as_dict()
    second = evaluate_agents(RandomAgent(), RandomAgent(), 4, seed=3).as_dict()
    total = combine_summaries([first, second])
    assert total.games == 8
    assert total.wins + total.losses + total.draws == 8
    assert total.total_moves == first["total_moves"] + second["total_moves"]
    with pytest.raises(ValueError):
        combine_summaries([first, {**second, "opponent": "other"}])
