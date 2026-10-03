"""Game-level supervised data for policy/value learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from connectfive.env import NUM_ACTIONS, ConnectFive, is_dead_draw
from connectfive.match import GameRecord
from connectfive.network import encode_state


@dataclass(frozen=True)
class SupervisedDataset:
    """Positions with enough provenance to prevent game-level leakage."""

    features: np.ndarray
    legal_action_masks: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    game_ids: np.ndarray
    plies: np.ndarray
    players: np.ndarray
    generators: np.ndarray
    # Optional teacher-search labels (format version 2). ``policy_targets`` is a
    # soft distribution over the 225 actions; ``search_values`` is the teacher's
    # score for the mover mapped to [-1, 1].
    policy_targets: np.ndarray | None = None
    search_values: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.actions)


def records_to_dataset(records: tuple[GameRecord, ...]) -> SupervisedDataset:
    """Replay complete games and create labels for teacher-selected moves."""

    env = ConnectFive()
    step = jax.jit(env.step)
    features = []
    legal_masks = []
    actions = []
    values = []
    game_ids = []
    plies = []
    players = []
    generators = []

    for game_id, record in enumerate(records):
        if record.illegal_player is not None:
            raise ValueError("cannot train from a game containing an illegal move")
        state = env.init(jax.random.PRNGKey(record.seed))
        opening_length = len(record.opening_moves)
        for ply, action in enumerate(record.moves):
            if bool(state.terminated):
                raise ValueError("record contains moves after termination")
            if not 0 <= action < NUM_ACTIONS or not bool(state.legal_action_mask[action]):
                raise ValueError("record contains an illegal action")
            player = int(state.current_player)
            if ply >= opening_length:
                features.append(np.asarray(encode_state(state), dtype=np.float32))
                legal_masks.append(np.asarray(state.legal_action_mask, dtype=np.bool_))
                actions.append(action)
                values.append(record.rewards[player])
                game_ids.append(game_id)
                plies.append(ply)
                players.append(player)
                generators.append(record.black if player == 0 else record.white)
            state = step(state, jnp.int32(action))

        actual_rewards = tuple(float(value) for value in state.rewards)
        if actual_rewards != record.rewards or not bool(state.terminated):
            raise ValueError("record replay does not reproduce its terminal result")

    if not actions:
        raise ValueError("records produced no teacher-selected training examples")
    return SupervisedDataset(
        features=np.stack(features),
        legal_action_masks=np.stack(legal_masks),
        actions=np.asarray(actions, dtype=np.int16),
        values=np.asarray(values, dtype=np.float32),
        game_ids=np.asarray(game_ids, dtype=np.int32),
        plies=np.asarray(plies, dtype=np.int16),
        players=np.asarray(players, dtype=np.int8),
        generators=np.asarray(generators, dtype=np.str_),
    )


def concatenate_datasets(first: SupervisedDataset, second: SupervisedDataset) -> SupervisedDataset:
    """Join datasets, renumbering the second one's games so ids stay unique."""

    offset = int(first.game_ids.max()) + 1 if len(first) else 0
    fields = {}
    for name in SupervisedDataset.__dataclass_fields__:
        a, b = getattr(first, name), getattr(second, name)
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError(f"only one dataset has optional field {name!r}")
            fields[name] = None
            continue
        if name == "game_ids":
            b = b + offset
        if name == "generators":
            a, b = a.astype(np.str_), b.astype(np.str_)
        fields[name] = np.concatenate((a, b))
    return SupervisedDataset(**fields)


def verify_record(record: GameRecord) -> None:
    """Replay a record through the JAX environment and check its final result."""

    env = ConnectFive()
    step = jax.jit(env.step)
    state = env.init(jax.random.PRNGKey(record.seed))
    for action in record.moves:
        if bool(state.terminated) or not bool(state.legal_action_mask[action]):
            raise ValueError(f"game {record.seed} is illegal under the environment rules")
        state = step(state, jnp.int32(action))
    rewards = tuple(float(value) for value in state.rewards)
    if record.adjudicated is not None:
        if (bool(state.terminated) or record.winner is not None
                or record.rewards != (0.0, 0.0)
                or record.adjudicated not in ("dead", "ply_cap")
                or (record.adjudicated == "dead"
                    and not is_dead_draw(np.asarray(state._board)))):
            raise ValueError(f"game {record.seed} is not a valid adjudicated draw")
        return
    if rewards != record.rewards or not bool(state.terminated):
        raise ValueError(f"game {record.seed} does not replay to its recorded result")


def split_by_game(
    dataset: SupervisedDataset,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return example indices with entire games assigned to one split."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    game_ids = np.unique(dataset.game_ids)
    if len(game_ids) < 2:
        raise ValueError("at least two games are required for a split")
    shuffled = np.random.default_rng(seed).permutation(game_ids)
    validation_games = max(1, round(len(game_ids) * validation_fraction))
    validation_games = min(validation_games, len(game_ids) - 1)
    validation_ids = shuffled[:validation_games]
    validation_mask = np.isin(dataset.game_ids, validation_ids)
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def save_dataset(
    path: Path,
    dataset: SupervisedDataset,
    records: tuple[GameRecord, ...],
    metadata: dict,
) -> None:
    """Save numeric examples and human-readable game provenance."""

    path.parent.mkdir(parents=True, exist_ok=True)
    optional = {
        name: getattr(dataset, name)
        for name in ("policy_targets", "search_values")
        if getattr(dataset, name) is not None
    }
    np.savez_compressed(
        path,
        features=dataset.features,
        legal_action_masks=dataset.legal_action_masks,
        actions=dataset.actions,
        values=dataset.values,
        game_ids=dataset.game_ids,
        plies=dataset.plies,
        players=dataset.players,
        generators=dataset.generators,
        **optional,
    )
    manifest = {
        "format_version": 2 if optional else 1,
        "examples": len(dataset),
        "games": [record.as_dict() for record in records],
        **metadata,
    }
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def load_dataset(path: Path) -> SupervisedDataset:
    """Load a dataset without permitting pickled object arrays."""

    with np.load(path, allow_pickle=False) as arrays:
        return SupervisedDataset(
            features=arrays["features"],
            legal_action_masks=arrays["legal_action_masks"],
            actions=arrays["actions"],
            values=arrays["values"],
            game_ids=arrays["game_ids"],
            plies=arrays["plies"],
            players=arrays["players"],
            generators=arrays["generators"],
            policy_targets=arrays.get("policy_targets"),
            search_values=arrays.get("search_values"),
        )
