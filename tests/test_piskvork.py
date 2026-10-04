import sys
from pathlib import Path

from connectfive.piskvork import PiskvorkEngine, action_to_xy, parse_move

FAKE = Path(__file__).parent / "fixtures" / "fake_brain.py"


def test_coordinates_are_column_row():
    assert action_to_xy(1 * 15 + 2) == "2,1"
    assert parse_move("2,1") == 17
    assert parse_move("15,0") == -1
    assert parse_move("MESSAGE 1,2") is None


def test_engine_plays_from_a_board_and_restarts():
    engine = PiskvorkEngine([sys.executable, str(FAKE)], info={"rule": 1}, name="fake")
    engine.start()
    try:
        action, _ = engine.move([0, 1, 2], timeout=10)
        assert action == 3  # lowest empty cell, after the MESSAGE line is skipped
        engine.restart()
        assert engine.move([], timeout=10)[0] == 0
        assert "> INFO rule 1" in engine.log
    finally:
        engine.close()


def test_last_eval_reads_the_latest_message():
    engine = PiskvorkEngine(["true"])
    engine.log += ["< MESSAGE Depth 3 | Eval 849 | Time 1ms", "< MESSAGE Speed 244K | Depth 5-6 | Eval -71 | Node 489",
                   "< 6,7"]
    assert engine.last_eval() == -71
    assert PiskvorkEngine(["true"]).last_eval() is None
