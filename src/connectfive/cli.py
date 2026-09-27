"""Terminal interface for Connect Five."""

import argparse

import jax
import jax.numpy as jnp

from connectfive.env import BOARD_SIZE, ConnectFive, State

COLUMNS = "ABCDEFGHJKLMNOPQRST"


def render(state: State) -> str:
    symbols = {-1: ".", 0: "X", 1: "O"}
    header = "   " + " ".join(COLUMNS)
    rows = [header]
    board = state._board.tolist()
    for index in range(BOARD_SIZE - 1, -1, -1):
        number = index + 1
        stones = " ".join(symbols[int(value)] for value in board[index])
        rows.append(f"{number:>2} {stones} {number:>2}")
    rows.append(header)
    return "\n".join(rows)


def parse_move(text: str) -> int:
    move = text.strip().upper()
    if len(move) < 2 or move[0] not in COLUMNS:
        raise ValueError("use a Go coordinate such as K10")
    try:
        row = int(move[1:]) - 1
    except ValueError as error:
        raise ValueError("the row must be a number from 1 to 19") from error
    if not 0 <= row < BOARD_SIZE:
        raise ValueError("the row must be a number from 1 to 19")
    return row * BOARD_SIZE + COLUMNS.index(move[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Connect Five on a 19x19 board")
    parser.add_argument(
        "--random-white", action="store_true", help="play black against a random white agent"
    )
    parser.add_argument("--seed", type=int, default=0, help="random-agent seed")
    args = parser.parse_args()

    env = ConnectFive()
    key = jax.random.PRNGKey(args.seed)
    state = env.init(key)

    while not bool(state.terminated):
        print("\n" + render(state))
        player = int(state.current_player)
        if args.random_white and player == 1:
            key, subkey = jax.random.split(key)
            probabilities = state.legal_action_mask / jnp.sum(state.legal_action_mask)
            action = int(jax.random.choice(subkey, env.num_actions, p=probabilities))
            row, col = divmod(action, BOARD_SIZE)
            print(f"White plays {COLUMNS[col]}{row + 1}.")
        else:
            label = "Black (X)" if player == 0 else "White (O)"
            try:
                entered = input(f"{label} move: ")
                if entered.strip().lower() in {"q", "quit", "exit"}:
                    print("Game ended.")
                    return
                action = parse_move(entered)
            except (EOFError, KeyboardInterrupt):
                print("\nGame ended.")
                return
            except ValueError as error:
                print(f"Invalid move: {error}.")
                continue
            if not bool(state.legal_action_mask[action]):
                print("That intersection is occupied.")
                continue
        state = env.step(state, action)

    print("\n" + render(state))
    if float(state.rewards[0]) > 0:
        print("Black wins!")
    elif float(state.rewards[1]) > 0:
        print("White wins!")
    else:
        print("Draw.")


if __name__ == "__main__":
    main()

