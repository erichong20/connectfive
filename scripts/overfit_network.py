"""Overfit the compact policy-value network on eight fixed positions."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from connectfive import NUM_ACTIONS
from connectfive.network import (
    NetworkConfig,
    PolicyValueNetwork,
    count_parameters,
    encode_state,
    make_train_step,
    policy_value_loss,
    save_checkpoint,
)
from connectfive.tactical_suite import load_tactical_suite, position_state


def make_tiny_dataset():
    positions = load_tactical_suite()
    features = jnp.stack([encode_state(position_state(position)) for position in positions])
    legal = jnp.stack(
        [position_state(position).legal_action_mask for position in positions]
    )
    policies = []
    for position in positions:
        target = jnp.zeros(NUM_ACTIONS)
        target = target.at[jnp.asarray(position.acceptable_actions)].set(
            1 / len(position.acceptable_actions)
        )
        policies.append(target)
    # Synthetic values deliberately exercise the value head. They are not game labels.
    values = jnp.linspace(-1.0, 1.0, len(positions))
    return features, legal, jnp.stack(policies), values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/network-overfit/model"))
    args = parser.parse_args()

    config = NetworkConfig()
    model = PolicyValueNetwork(config)
    features, legal, policies, values = make_tiny_dataset()
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, features)
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer)
    initial, _ = policy_value_loss(model, params, features, legal, policies, values)

    metrics = None
    for step in range(1, args.steps + 1):
        params, opt_state, metrics = train_step(
            params, opt_state, features, legal, policies, values
        )
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(
                f"step {step:4d} | loss {float(metrics['loss']):.4f} "
                f"policy {float(metrics['policy_loss']):.4f} "
                f"value {float(metrics['value_loss']):.4f}",
                flush=True,
            )

    assert metrics is not None
    final = float(metrics["loss"])
    save_checkpoint(
        args.out,
        params,
        config,
        step=args.steps,
        metrics={"initial_loss": float(initial), "final_loss": final},
    )
    print(
        f"parameters {count_parameters(params):,} | "
        f"loss {float(initial):.4f} -> {final:.4f} | checkpoint {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
