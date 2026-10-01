"""Terminal interface for Connect Five."""

import argparse

import jax

from connectfive.agents import RandomAgent, make_agent
from connectfive.env import BOARD_SIZE, ConnectFive, State

COLUMNS = "ABCDEFGHJKLMNOPQRST"[:BOARD_SIZE]
PLAYER_NAMES = ("Black", "White")


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
        raise ValueError(f"the row must be a number from 1 to {BOARD_SIZE}") from error
    if not 0 <= row < BOARD_SIZE:
        raise ValueError(f"the row must be a number from 1 to {BOARD_SIZE}")
    return row * BOARD_SIZE + COLUMNS.index(move[0])


def random_action(state: State, key: jax.Array) -> int:
    """Backward-compatible wrapper around the reusable random agent."""

    return RandomAgent().select_action(state, key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play standard Gomoku on a 15x15 board")
    parser.add_argument(
        "--bot",
        choices=("random", "tactical", "lookahead", "negamax", "threatsearch", "pattern"),
        help="play against a bot using the selected policy",
    )
    parser.add_argument(
        "--human-color",
        choices=("black", "white"),
        default="black",
        help="your color when playing a bot; White makes the bot move first (default: black)",
    )
    parser.add_argument(
        "--random-white",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=0, help="random-agent seed")
    args = parser.parse_args()

    env = ConnectFive()
    key = jax.random.PRNGKey(args.seed)
    state = env.init(key)
    bot_player = None
    bot_agent = None
    if args.bot or args.random_white:
        bot_player = 0 if args.human_color == "white" and not args.random_white else 1
        bot_agent = make_agent(args.bot or "random")

    while not bool(state.terminated):
        print("\n" + render(state))
        player = int(state.current_player)
        if player == bot_player:
            key, subkey = jax.random.split(key)
            assert bot_agent is not None
            action = bot_agent.select_action(state, subkey)
            row, col = divmod(action, BOARD_SIZE)
            print(
                f"{PLAYER_NAMES[player]} {bot_agent.name} bot plays "
                f"{COLUMNS[col]}{row + 1}."
            )
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
