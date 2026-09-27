import pytest

from connectfive.cli import parse_move


@pytest.mark.parametrize(("text", "action"), [("A1", 0), ("K10", 180), ("T19", 360)])
def test_parse_move(text, action):
    assert parse_move(text) == action


@pytest.mark.parametrize("text", ["I5", "A0", "T20", "wat"])
def test_parse_move_rejects_invalid_coordinates(text):
    with pytest.raises(ValueError):
        parse_move(text)

