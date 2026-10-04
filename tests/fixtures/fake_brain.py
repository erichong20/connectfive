"""Minimal Piskvork brain for tests: plays the lowest-numbered empty cell."""
import sys

size, stones = 15, set()
reading = False
for line in sys.stdin:
    command = line.strip()
    if reading:
        if command == "DONE":
            reading = False
            free = next((r, c) for r in range(size) for c in range(size) if (c, r) not in stones)
            print(f"MESSAGE thinking\n{free[1]},{free[0]}", flush=True)
        else:
            x, y, _ = command.split(",")
            stones.add((int(x), int(y)))
    elif command.startswith(("START", "RESTART")):
        stones.clear()
        print("OK", flush=True)
    elif command == "BOARD":
        stones.clear()
        reading = True
    elif command == "END":
        break
