import pytest

from jarvis.voice.input import HotkeyTracker, key_name, parse_hotkey


def test_parse_simple_and_combo():
    assert parse_hotkey("f8") == {"f8"}
    assert parse_hotkey("Ctrl + Space") == {"ctrl", "space"}


def test_parse_aliases():
    assert parse_hotkey("right_ctrl") == {"ctrl_r"}
    assert parse_hotkey("windows+j") == {"cmd", "j"}


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_hotkey("  ")


def test_single_key_start_stop_and_key_repeat():
    tracker = HotkeyTracker("right_ctrl")
    assert tracker.press("ctrl_r") == "start"
    assert tracker.press("ctrl_r") is None  # Windows repeats key-down while held
    assert tracker.active
    assert tracker.release("ctrl_r") == "stop"
    assert not tracker.active


def test_right_ctrl_ignores_left_ctrl():
    tracker = HotkeyTracker("right_ctrl")
    assert tracker.press("ctrl_l") is None


def test_generic_ctrl_accepts_either_side():
    for side in ("ctrl_l", "ctrl_r"):
        tracker = HotkeyTracker("ctrl+space")
        assert tracker.press(side) is None
        assert tracker.press("space") == "start"
        assert tracker.release(side) == "stop"


def test_combo_stops_when_either_key_released():
    tracker = HotkeyTracker("ctrl+alt+j")
    tracker.press("ctrl_l")
    tracker.press("alt_l")
    assert tracker.press("j") == "start"
    assert tracker.release("j") == "stop"
    assert tracker.press("j") == "start"


def test_unrelated_keys_do_nothing():
    tracker = HotkeyTracker("f8")
    assert tracker.press("a") is None
    assert tracker.release("a") is None
    assert not tracker.active


class _Special:
    name = "ctrl_r"


class _Char:
    def __init__(self, char):
        self.char = char


class _Unknown:
    pass


def test_key_name_special_and_char():
    assert key_name(_Special()) == "ctrl_r"
    assert key_name(_Char("J")) == "j"


def test_key_name_control_code_maps_back_to_letter():
    assert key_name(_Char("\n")) == "j"  # Ctrl+J arrives as '\n'


def test_key_name_unknown_is_none():
    assert key_name(_Unknown()) is None
