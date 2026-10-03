"""Train and validate the compact policy/value network on complete games."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from connectfive import NUM_ACTIONS
from connectfive.dataset import concatenate_datasets, load_dataset, split_by_game
from connectfive.network import (
    NetworkConfig,
    PolicyValueNetwork,
    count_parameters,
    load_checkpoint,
    make_train_step,
    mask_policy_logits,
    save_checkpoint,
    transform_features,
    transform_policy,
    with_input_planes,
)


def augment(
    features: np.ndarray,
    legal: np.ndarray,
    policies: np.ndarray,
    symmetries: np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply independently sampled D4 transforms to a batch."""

    policies = policies.astype(np.float32)
    output_features = np.empty_like(features)
    output_legal = np.empty_like(legal)
    output_policies = np.empty_like(policies)
    for symmetry in range(8):
        selected = symmetries == symmetry
        if not np.any(selected):
            continue
        output_features[selected] = np.asarray(
            transform_features(jnp.asarray(features[selected]), symmetry)
        )
        output_legal[selected] = np.asarray(
            transform_policy(jnp.asarray(legal[selected]), symmetry)
        )
        output_policies[selected] = np.asarray(
            transform_policy(jnp.asarray(policies[selected]), symmetry)
        )
    return (
        jnp.asarray(output_features),
        jnp.asarray(output_legal),
        jnp.asarray(output_policies),
    )


def value_targets(dataset, indices, target: str) -> np.ndarray:
    outcome = dataset.values[indices]
    if target == "outcome":
        return outcome
    if dataset.search_values is None:
        raise ValueError(f"value target {target!r} needs a dataset with search values")
    search = dataset.search_values[indices]
    if target == "search":
        return search
    return 0.5 * (outcome + search)


def replay_indices(extra, hold_out: bool, fraction: float, seed: int) -> np.ndarray:
    """Training indices of a replay dataset.

    With ``hold_out`` the dataset's own validation games (the split it gets as
    a primary dataset) are dropped, so games a previous model was validated on
    never leak into later training.
    """

    if not hold_out:
        return np.arange(len(extra))
    return split_by_game(extra, fraction, seed)[0]


def game_value_weights(game_ids: np.ndarray) -> np.ndarray:
    """Per-position weights giving each game the same total value-loss weight.

    Long games (board-filling draws contribute ~220 positions) would otherwise
    dominate the value target. Weights average 1, so the loss scale is kept.
    """

    _, inverse, counts = np.unique(game_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return (weights / weights.mean()).astype(np.float32)


def policy_targets(dataset, indices, target: str) -> np.ndarray:
    if target == "soft":
        if dataset.policy_targets is None:
            raise ValueError("soft policy targets need a dataset with policy_targets")
        return dataset.policy_targets[indices].astype(np.float32)
    return np.eye(NUM_ACTIONS, dtype=np.float32)[dataset.actions[indices].astype(np.int32)]


def evaluate(model, params, dataset, indices, planes: int, chunk: int = 2_048) -> dict[str, float]:
    """Held-out metrics against the hard move, outcome, and (if present) teacher labels."""

    logits_parts, value_parts = [], []
    for start in range(0, len(indices), chunk):
        part = indices[start:start + chunk]
        features = jnp.asarray(with_input_planes(dataset.features[part], planes))
        logits, values = model.apply(params, features)
        logits_parts.append(np.asarray(
            mask_policy_logits(logits, jnp.asarray(dataset.legal_action_masks[part]))
        ))
        value_parts.append(np.asarray(values))
    logits = jnp.asarray(np.concatenate(logits_parts))
    values = np.concatenate(value_parts)
    actions = jnp.asarray(dataset.actions[indices], dtype=jnp.int32)
    outcome = dataset.values[indices]
    policy_loss = optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()
    predicted = np.asarray(jnp.argmax(logits, axis=1))
    metrics = {
        "policy_loss": float(policy_loss),
        "policy_accuracy": float((predicted == np.asarray(actions)).mean()),
        "value_mse": float(np.square(values - outcome).mean()),
        "value_mae": float(np.abs(values - outcome).mean()),
        # Reference point: always predicting the mean training outcome.
        "value_mae_constant_baseline": float(np.abs(outcome - outcome.mean()).mean()),
        "value_sign_accuracy_decisive": float(
            (np.sign(values[outcome != 0]) == outcome[outcome != 0]).mean()
        ) if np.any(outcome != 0) else float("nan"),
    }
    if dataset.policy_targets is not None:
        soft = dataset.policy_targets[indices].astype(np.float32)
        metrics["soft_policy_loss"] = float(
            optax.softmax_cross_entropy(logits, jnp.asarray(soft)).mean()
        )
        # A move the teacher scored as (near-)best counts as correct.
        metrics["teacher_set_accuracy"] = float(
            (soft[np.arange(len(predicted)), predicted] > 0.05).mean()
        )
    if dataset.search_values is not None:
        search = dataset.search_values[indices]
        metrics["search_value_mae"] = float(np.abs(values - search).mean())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/supervised-v0/model"))
    parser.add_argument("--policy-target", choices=("move", "soft"), default="move")
    parser.add_argument(
        "--value-target", choices=("outcome", "search", "blend"), default="outcome"
    )
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument(
        "--hold-out-extra", action="store_true",
        help="also drop each --extra-train dataset's validation games (same split)",
    )
    parser.add_argument(
        "--value-weighting", choices=("position", "game"), default="position",
        help="'game' gives every game equal total weight in the value loss",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--input-planes", type=int, choices=(3, 4), default=3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument(
        "--extra-train", type=Path, nargs="*", default=[],
        help="additional datasets used only for training (e.g. DAgger rounds)",
    )
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="fine-tune from these weights (architecture flags are taken from it)",
    )
    parser.add_argument(
        "--validation-dataset", type=Path,
        help="score this dataset's held-out games too (same split seed and fraction)",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    train_indices, validation_indices = split_by_game(
        dataset, args.validation_fraction, args.seed
    )
    for extra_path in args.extra_train:
        before = len(dataset)
        extra = load_dataset(extra_path)
        extra_train = replay_indices(
            extra, args.hold_out_extra, args.validation_fraction, args.seed
        )
        dataset = concatenate_datasets(dataset, extra)
        train_indices = np.concatenate((train_indices, before + extra_train))
    key = jax.random.PRNGKey(args.seed)
    if args.init_checkpoint:
        loaded = load_checkpoint(args.init_checkpoint, key)
        config, params = loaded.config, loaded.params
        model = PolicyValueNetwork(config)
    else:
        config = NetworkConfig(
            residual_blocks=args.blocks,
            channels=args.channels,
            input_planes=args.input_planes,
        )
        model = PolicyValueNetwork(config)
        params = model.init(
            key, jnp.asarray(with_input_planes(dataset.features[:1], config.input_planes))
        )
    planes = config.input_planes
    schedule = optax.cosine_decay_schedule(args.learning_rate, args.steps, alpha=0.05)
    optimizer = (
        optax.adamw(schedule, weight_decay=args.weight_decay)
        if args.weight_decay
        else optax.adam(schedule)
    )
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer, args.value_weight)
    value_sample_weights = (
        game_value_weights(dataset.game_ids) if args.value_weighting == "game" else None
    )
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        batch = rng.choice(train_indices, size=args.batch_size, replace=True)
        symmetries = rng.integers(0, 8, size=args.batch_size)
        features, legal, policies = augment(
            with_input_planes(dataset.features[batch], planes),
            dataset.legal_action_masks[batch],
            policy_targets(dataset, batch, args.policy_target),
            symmetries,
        )
        params, opt_state, metrics = train_step(
            params,
            opt_state,
            features,
            legal,
            policies,
            jnp.asarray(value_targets(dataset, batch, args.value_target)),
            None if value_sample_weights is None
            else jnp.asarray(value_sample_weights[batch]),
        )
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step {step:4d} | train loss {float(metrics['loss']):.4f}")

    train_metrics = evaluate(model, params, dataset, train_indices, planes)
    validation_metrics = evaluate(model, params, dataset, validation_indices, planes)
    extra_validation = None
    if args.validation_dataset:
        other = load_dataset(args.validation_dataset)
        _, other_validation = split_by_game(other, args.validation_fraction, args.seed)
        extra_validation = evaluate(model, params, other, other_validation, planes)
        print(f"extra validation ({args.validation_dataset}): {extra_validation}")
    elapsed = time.perf_counter() - started
    checkpoint_metrics = {
        f"train_{name}": value for name, value in train_metrics.items()
    }
    checkpoint_metrics.update(
        {f"validation_{name}": value for name, value in validation_metrics.items()}
    )
    save_checkpoint(args.out, params, config, args.steps, checkpoint_metrics)
    report = {
        "dataset": str(args.dataset),
        "extra_train": [str(path) for path in args.extra_train],
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "extra_validation": extra_validation,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "policy_target": args.policy_target,
        "value_target": args.value_target,
        "value_weighting": args.value_weighting,
        "hold_out_extra": args.hold_out_extra,
        "value_weight": args.value_weight,
        "weight_decay": args.weight_decay,
        "config": {
            "residual_blocks": config.residual_blocks,
            "channels": config.channels,
            "value_hidden": config.value_hidden,
            "input_planes": config.input_planes,
        },
        "learning_rate_schedule": "cosine to 5%",
        "train_games": sorted(set(dataset.game_ids[train_indices].tolist())),
        "validation_games": sorted(set(dataset.game_ids[validation_indices].tolist())),
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "parameters": count_parameters(params),
        "elapsed_seconds": elapsed,
        "train": train_metrics,
        "validation": validation_metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"train: {train_metrics}")
    print(f"validation: {validation_metrics}")
    print(f"saved checkpoint {args.out}; {elapsed:.1f}s")


if __name__ == "__main__":
    main()
