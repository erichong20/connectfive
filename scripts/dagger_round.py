"""One DAgger round: the network plays itself, the teacher labels its positions.

Imitation learning only shows the student positions the teacher reaches. Once
the student makes one unfamiliar move, later positions drift away from the
training data and errors compound. DAgger collects the positions the *student*
actually reaches and asks the teacher what to do there.

Value labels for these games use the teacher's search value, because the final
outcome of a weak student-versus-student game says little about the position.
"""

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from connectfive.dataset import SupervisedDataset, save_dataset
from connectfive.match import generate_opening
from connectfive.network import (
    PolicyValueNetwork,
    encode_board,
    load_checkpoint,
    mask_policy_logits,
)
from connectfive.patterns import ACTION_TO_INDEX, PatternBoard
from connectfive.teacher import (
    TeacherConfig,
    config_dict,
    game_record,
    label_games,
    teacher_games_to_arrays,
)


def self_play(checkpoint: Path, seeds: list[int], opening_plies: int,
              sample_plies: int, temperature: float) -> list[tuple]:
    """Play batched student self-play games; returns (seed, moves, opening, winner)."""

    loaded = load_checkpoint(checkpoint, jax.random.PRNGKey(0))
    model = PolicyValueNetwork(loaded.config)
    planes = loaded.config.input_planes

    @jax.jit
    def policy(features, legal):
        logits, _ = model.apply(loaded.params, features)
        return mask_policy_logits(logits, legal)

    rng = np.random.default_rng(seeds[0])
    boards, openings, winners = [], [], []
    for seed in seeds:
        board = PatternBoard()
        opening = generate_opening(seed, opening_plies)
        for action in opening:
            board.play(ACTION_TO_INDEX[action])
        boards.append(board)
        openings.append(opening)
        winners.append(None)
    active = list(range(len(seeds)))
    while active:
        arrays = [boards[i].to_array() for i in active]
        features = np.stack([
            encode_board(array, boards[i].player, planes) for array, i in zip(arrays, active)
        ])
        legal = np.stack([array.reshape(-1) == -1 for array in arrays])
        logits = np.asarray(policy(jnp.asarray(features), jnp.asarray(legal)), dtype=np.float64)
        still_active = []
        for row, i in enumerate(active):
            board = boards[i]
            if len(board.moves) - len(openings[i]) < sample_plies:
                scaled = (logits[row] - logits[row].max()) / temperature
                probabilities = np.where(legal[row], np.exp(scaled), 0.0)
                action = int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
            else:
                action = int(logits[row].argmax())
            index = ACTION_TO_INDEX[action]
            mover = board.turn
            won = board.summary[mover][index][3]
            board.play(index)
            if won:
                winners[i] = mover - 1
            elif not board.is_full():
                still_active.append(i)
        active = still_active
    return [
        (seed, tuple(board.moves), opening, winner)
        for seed, board, opening, winner in zip(seeds, boards, openings, winners)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--node-budget", type=int, default=2_000)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--sample-plies", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.games))
    started = time.perf_counter()
    played = self_play(
        args.checkpoint, seeds, args.opening_plies, args.sample_plies, args.temperature
    )
    play_seconds = time.perf_counter() - started
    config = TeacherConfig(node_budget=args.node_budget, opening_plies=args.opening_plies)
    labelled = label_games(played, config, workers=args.workers)
    label_seconds = time.perf_counter() - started - play_seconds
    arrays = teacher_games_to_arrays(labelled, config)
    arrays["values"] = arrays["search_values"].copy()
    arrays["generators"] = np.asarray(
        [f"dagger:{args.checkpoint.name}/{config.name}"] * len(arrays["actions"]), dtype=np.str_
    )
    dataset = SupervisedDataset(**arrays)
    agreement = float(np.mean([
        game.moves[position.ply] == position.best_action
        or dict(position.policy).get(game.moves[position.ply], 0.0) > 0.05
        for game in labelled for position in game.positions
    ]))
    lengths = [len(game.moves) for game in labelled]
    summary = {
        "student_teacher_agreement": agreement,
        "mean_length": float(np.mean(lengths)),
        "play_seconds": play_seconds,
        "label_seconds": label_seconds,
        "value_label": "teacher search value (outcome not used)",
    }
    records = tuple(game_record(game, config) for game in labelled)
    save_dataset(
        args.out, dataset, records,
        metadata={"student": str(args.checkpoint), "teacher": config_dict(config),
                  "seeds": [seeds[0], seeds[-1]], "summary": summary},
    )
    print(f"saved {len(dataset)} student positions to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
