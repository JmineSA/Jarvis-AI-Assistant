import threading
import time

import numpy as np

from jarvis.voice.wakeword import (
    CHUNK,
    MODEL_FILES,
    WakeWordDetector,
    WakeWordTrigger,
    ensure_wake_models,
    make_chime,
    wake_models_present,
)


class FakeSession:
    """Stands in for an onnx model: returns arrays of the right shape."""

    def __init__(self, kind, scores=None):
        self.kind, self.scores, self.calls = kind, scores or [], 0

    def get_inputs(self):
        return [type("I", (), {"name": "x"})()]

    def run(self, _outputs, feed):
        self.calls += 1
        data = next(iter(feed.values()))
        if self.kind == "mel":
            frames = max(1, (data.shape[1] - 400) // 160 + 1)
            return [np.zeros((frames, 1, 1, 32), dtype=np.float32)]
        if self.kind == "embed":
            return [np.zeros((data.shape[0], 1, 1, 96), dtype=np.float32)]
        score = self.scores[min(self.calls - 1, len(self.scores) - 1)] if self.scores else 0.0
        return [np.array([[score]], dtype=np.float32)]


def make_detector(tmp_path, scores=None):
    kinds = {"melspectrogram.onnx": "mel", "embedding_model.onnx": "embed", "hey_jarvis_v0.1.onnx": "wake"}
    return WakeWordDetector(tmp_path, session_factory=lambda p: FakeSession(kinds[p.name], scores))


def test_detector_processes_chunks_and_keeps_the_remainder(tmp_path):
    det = make_detector(tmp_path)
    assert det.feed(np.zeros(CHUNK * 2 + 100, dtype=np.float32)) == 0.0
    assert det.chunks_seen == 2
    det.feed(np.zeros(CHUNK - 100, dtype=np.float32))
    assert det.chunks_seen == 3  # the leftover 100 samples joined the next audio


def test_detector_returns_the_models_score(tmp_path):
    det = make_detector(tmp_path, scores=[0.1, 0.2, 0.9])
    det.chunks_seen = 0
    scores = [det.feed(np.zeros(CHUNK, dtype=np.float32)) for _ in range(3)]
    assert scores[-1] > 0.5 and scores[0] < 0.5


def test_ensure_models_downloads_only_what_is_missing(tmp_path):
    fetched = []

    def fake(url, target):
        fetched.append(url.rsplit("/", 1)[1])
        target.write_bytes(b"x")

    ensure_wake_models(tmp_path, fake, say=lambda m: None)
    assert sorted(fetched) == sorted(MODEL_FILES) and wake_models_present(tmp_path)
    fetched.clear()
    ensure_wake_models(tmp_path, fake, say=lambda m: None)
    assert fetched == []


def test_failed_download_leaves_no_partial_file(tmp_path):
    def boom(url, target):
        target.write_bytes(b"half")
        raise OSError("offline")

    try:
        ensure_wake_models(tmp_path, boom, say=lambda m: None)
    except Exception as exc:
        assert "Could not download" in str(exc)
    assert not any(tmp_path.rglob("*.part")) and not wake_models_present(tmp_path)


class FakeHotkey:
    description = "Right Ctrl"

    def __init__(self):
        self.pressed = False
        self.started = self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def wait_for_press(self, stop, timeout=None):
        time.sleep(min(timeout or 0.01, 0.01))
        if self.pressed:
            self.pressed = False
            return True
        return None

    def wait_for_release(self, stop, timeout=None):
        self.released = True

    def interrupt_requested(self):
        return False

    def clear(self):
        pass


class FakeStream:
    def __init__(self):
        self.closed = False

    def read(self, n):
        time.sleep(0.005)
        return np.zeros((n, 1), dtype=np.float32), False

    def stop(self):
        pass

    def close(self):
        self.closed = True


def make_trigger(tmp_path, scores, hotkey=None, chimes=None, level=lambda s: 0.0):
    streams = []

    def factory():
        streams.append(FakeStream())
        return streams[-1]

    det = make_detector(tmp_path, scores)
    trig = WakeWordTrigger(
        hotkey or FakeHotkey(), det, None, level, threshold=0.5, silence_seconds=0.2, first_word_seconds=0.3,
        chime=(lambda: chimes.append(1)) if chimes is not None else None, stream_factory=factory,
    )
    return trig, streams


def test_saying_the_wake_word_starts_a_turn_and_closes_the_microphone(tmp_path):
    chimes = []
    trig, streams = make_trigger(tmp_path, scores=[0.0] * 20 + [0.95], chimes=chimes)
    trig.start()
    try:
        assert trig.wait_for_press(threading.Event(), 5) is True
        assert streams and all(s.closed for s in streams)  # the mic is free for the recorder
        assert chimes == [1] and trig._by_voice is True
    finally:
        trig.close()


def test_push_to_talk_still_works_and_nothing_fires_without_the_phrase(tmp_path):
    hotkey = FakeHotkey()
    trig, streams = make_trigger(tmp_path, scores=[0.0], hotkey=hotkey)
    trig.start()
    try:
        assert trig.wait_for_press(threading.Event(), 0.3) is None  # nothing heard, nothing pressed
        hotkey.pressed = True
        assert trig.wait_for_press(threading.Event(), 2) is True and trig._by_voice is False
        assert all(s.closed for s in streams)
        trig.wait_for_release(threading.Event(), 1)
        assert hotkey.released
    finally:
        trig.close()


def test_spoken_wake_records_until_you_go_quiet(tmp_path):
    levels = iter([0.0] * 12 + [0.2] * 6 + [0.0] * 200)
    trig, _ = make_trigger(tmp_path, scores=[0.0], level=lambda s: next(levels))
    trig._by_voice = True
    started = time.monotonic()
    trig.wait_for_release(threading.Event(), 10)
    took = time.monotonic() - started
    assert 0.2 < took < 3  # waited for speech, then for the silence, not the 10 s limit


def test_woke_but_said_nothing_gives_up(tmp_path):
    trig, _ = make_trigger(tmp_path, scores=[0.0])
    trig._by_voice = True
    started = time.monotonic()
    trig.wait_for_release(threading.Event(), 10)
    assert time.monotonic() - started < 2


def test_chime_is_a_short_clip():
    clip = make_chime()
    assert 0.1 < clip.duration < 0.5 and float(np.max(np.abs(clip.samples))) <= 0.3


def test_wake_setup_command_downloads_and_switches_the_wake_word_on(tmp_path, monkeypatch):
    from jarvis import diagnostics
    from jarvis.config import Settings
    from jarvis.voice import wakeword

    config = tmp_path / "settings.yaml"
    config.write_text("voice:\n  wake_word: false\n", encoding="utf-8")
    settings = Settings.model_validate({"voice": {"models_dir": str(tmp_path / "models")}})

    def fake(models_dir, say=print):
        folder = wakeword.wake_dir(models_dir)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    monkeypatch.setattr(wakeword, "ensure_wake_models", fake)
    lines = []
    assert diagnostics.run_wake_setup(settings, config, say=lines.append) == 0
    assert "wake_word: true" in config.read_text(encoding="utf-8")
    assert any("Hey JARVIS" in line for line in lines)


def test_main_knows_the_wake_commands():
    from jarvis import main

    assert main.resolve_config_path  # imported, so `wake-setup` cannot hit a NameError


def test_trigger_reports_how_the_turn_started_and_can_listen_for_a_follow_up(tmp_path):
    chimes = []
    trig, _ = make_trigger(tmp_path, scores=[0.0] * 20 + [0.95], chimes=chimes)
    assert trig.woke_by_voice is False
    trig.start()
    try:
        assert trig.wait_for_press(threading.Event(), 5) is True and trig.woke_by_voice is True
        trig.begin_follow_up()
        assert trig.woke_by_voice is True  # a follow-up listens by voice, with no key and no wake word
    finally:
        trig.close()


def test_key_press_is_not_a_voice_turn(tmp_path):
    hotkey = FakeHotkey()
    trig, _ = make_trigger(tmp_path, scores=[0.0], hotkey=hotkey)
    trig.start()
    try:
        hotkey.pressed = True
        assert trig.wait_for_press(threading.Event(), 2) is True and trig.woke_by_voice is False
    finally:
        trig.close()
