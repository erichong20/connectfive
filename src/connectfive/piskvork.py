"""Play external Gomocup engines through the Piskvork stdin/stdout protocol.

Gomocup engines ("brains") read text commands and print moves as ``x,y`` with
x the column and y the row, both 0-based. This client keeps the referee on our
side: every external move is checked against our own board, so the exact-five
rule, legality and results never depend on the other engine.

Only the commands this project needs are used: ``START``, ``INFO``,
``BOARD``/``DONE`` (the whole position, so the engine needs no history),
``RESTART`` and ``END``. Lines starting with ``MESSAGE``, ``DEBUG``,
``SUGGEST``, ``OK`` or ``UNKNOWN`` are not moves.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

SIZE = 15
NON_MOVE_PREFIXES = ("MESSAGE", "DEBUG", "SUGGEST", "OK", "UNKNOWN", "INFO", "ERROR")


class EngineError(RuntimeError):
    """The external engine crashed, timed out or sent something unusable."""


def action_to_xy(action: int) -> str:
    row, col = divmod(action, SIZE)
    return f"{col},{row}"


def parse_move(line: str) -> int | None:
    """Return the action for an ``x,y`` line, or None if it is not a move."""

    parts = line.strip().split(",")
    if len(parts) != 2 or not all(part.strip().lstrip("-").isdigit() for part in parts):
        return None
    col, row = (int(part) for part in parts)
    if not (0 <= col < SIZE and 0 <= row < SIZE):
        return -1  # a move, but off the board: illegal
    return row * SIZE + col


@dataclass
class PiskvorkEngine:
    """One running engine process; ``info`` lines are sent after every START."""

    command: list[str]
    cwd: Path | None = None
    info: dict[str, int | str] = field(default_factory=dict)
    name: str = "external"
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _lines: queue.Queue = field(default_factory=queue.Queue, repr=False)
    log: list[str] = field(default_factory=list, repr=False)

    def start(self) -> None:
        self._process = subprocess.Popen(
            self.command, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        threading.Thread(target=self._read, daemon=True).start()
        self.send(f"START {SIZE}")
        self._expect_ok(timeout=30)
        for key, value in self.info.items():
            self.send(f"INFO {key} {value}")

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line.rstrip("\n"))
        self._lines.put(None)

    def send(self, line: str) -> None:
        if self._process is None or self._process.poll() is not None:
            raise EngineError(f"{self.name} is not running")
        assert self._process.stdin is not None
        self.log.append(f"> {line}")
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _next_line(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EngineError(f"{self.name} timed out")
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty as error:
            raise EngineError(f"{self.name} timed out") from error
        if line is None:
            raise EngineError(f"{self.name} exited")
        self.log.append(f"< {line}")
        return line

    def _expect_ok(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            line = self._next_line(deadline).strip()
            if line.upper().startswith("OK"):
                return
            if line.upper().startswith("ERROR"):
                raise EngineError(f"{self.name}: {line}")

    def restart(self) -> None:
        self.send("RESTART")
        self._expect_ok(timeout=30)
        for key, value in self.info.items():
            self.send(f"INFO {key} {value}")

    def move(self, history: list[int], timeout: float) -> tuple[int, float]:
        """Send the position (moves in play order) and return (action, seconds).

        ``history[i]`` was played by black for even i; the engine is the side to
        move, so its own stones are those with the parity of ``len(history)``.
        """

        mine = len(history) % 2
        self.send("BOARD")
        for ply, action in enumerate(history):
            self.send(f"{action_to_xy(action)},{1 if ply % 2 == mine else 2}")
        started = time.monotonic()
        self.send("DONE")
        deadline = started + timeout
        while True:
            line = self._next_line(deadline)
            if line.strip().upper().startswith(NON_MOVE_PREFIXES):
                continue
            action = parse_move(line)
            if action is not None:
                return action, time.monotonic() - started

    def last_eval(self, mate: int = 20_000) -> int | None:
        """The last ``Eval`` reported (Rapfi/Yixin MESSAGE lines) for the latest move.

        Only lines after the most recent ``DONE`` count, so a value is never taken
        from an earlier search. ``+M3`` / ``-M5`` (forced win/loss in N moves) map
        to ``±(mate - N)``.
        """

        for line in reversed(self.log):
            if line == "> DONE":
                return None
            if line.startswith("< MESSAGE") and "Eval" in line:
                token = line.split("Eval", 1)[1].split("|")[0].strip().split()
                if not token:
                    continue
                text = token[0]
                sign = -1 if text.startswith("-") else 1
                body = text.lstrip("+-")
                if body.startswith("M") and body[1:].isdigit():
                    return sign * (mate - int(body[1:]))
                if body.isdigit():
                    return sign * int(body)
        return None

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self.send("END")
            self._process.wait(timeout=5)
        except (EngineError, OSError, subprocess.TimeoutExpired):
            self._process.kill()
        self._process = None
