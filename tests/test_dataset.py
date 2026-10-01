from pathlib import Path

import numpy as np

from connectfive import RandomAgent, play_game
from connectfive.dataset import (
    load_dataset,
    records_to_dataset,
    save_dataset,
    split_by_game,
)


def make_records():
    return tuple(
        play_game(RandomAgent(), RandomAgent(), seed=seed) for seed in (20, 21, 22)
    )


def test_record_labels_use_mover_perspective_and_replay():
    records = make_records()
    dataset = records_to_dataset(records)
    assert len(dataset) == sum(len(record.moves) for record in records)
    for game_id, record in enumerate(records):
        selected = dataset.game_ids == game_id
        expected = np.array(
            [record.rewards[ply % 2] for ply in range(len(record.moves))]
        )
        assert np.array_equal(dataset.values[selected], expected)
        assert np.all(dataset.legal_action_masks[np.arange(len(dataset)), dataset.actions])


def test_random_openings_are_not_policy_targets():
    record = play_game(RandomAgent(), RandomAgent(), seed=30, opening_moves=(112, 113))
    dataset = records_to_dataset((record,))
    assert len(dataset) == len(record.moves) - 2
    assert dataset.plies[0] == 2


def test_split_keeps_games_disjoint():
    dataset = records_to_dataset(make_records())
    train, validation = split_by_game(dataset, validation_fraction=0.34, seed=4)
    assert set(dataset.game_ids[train]).isdisjoint(dataset.game_ids[validation])
    assert set(train) | set(validation) == set(range(len(dataset)))


def test_dataset_round_trip(tmp_path: Path):
    records = make_records()
    dataset = records_to_dataset(records)
    path = tmp_path / "games.npz"
    save_dataset(path, dataset, records, {"seed": 20})
    loaded = load_dataset(path)
    for field in dataset.__dataclass_fields__:
        assert np.array_equal(getattr(dataset, field), getattr(loaded, field))
    assert path.with_suffix(".json").exists()


def test_teacher_games_label_every_position_and_replay(tmp_path: Path):
    from connectfive.dataset import SupervisedDataset
    from connectfive.teacher import (
        TeacherConfig,
        game_record,
        generate_teacher_games,
        teacher_games_to_arrays,
    )

    config = TeacherConfig(node_budget=200, max_depth=3, sample_plies=2)
    games = generate_teacher_games([5, 6], config, workers=1)
    dataset = SupervisedDataset(**teacher_games_to_arrays(games, config))
    records = tuple(game_record(game, config) for game in games)
    # Openings are replayed but never labelled.
    assert len(dataset) == sum(len(g.moves) - len(g.opening_moves) for g in games)
    assert np.all(dataset.legal_action_masks[np.arange(len(dataset)), dataset.actions])
    sums = dataset.policy_targets.astype(np.float32).sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-2)
    assert np.all(dataset.legal_action_masks | (dataset.policy_targets == 0))
    for game_id, game in enumerate(games):
        selected = dataset.game_ids == game_id
        if game.winner is not None:
            expected = np.where(dataset.players[selected] == game.winner, 1.0, -1.0)
            assert np.array_equal(dataset.values[selected], expected)
    replayed = records_to_dataset(records)
    assert replayed.features.shape[0] == len(dataset)
    assert np.array_equal(replayed.features, dataset.features)
    path = tmp_path / "teacher.npz"
    save_dataset(path, dataset, records, {"teacher": config.name})
    loaded = load_dataset(path)
    assert np.array_equal(loaded.policy_targets, dataset.policy_targets)
    assert np.array_equal(loaded.search_values, dataset.search_values)
