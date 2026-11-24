import numpy as np

from jarvis.voice.audio_utils import clean_for_speech, peak_level, resample, rms_level


def test_resample_same_rate_is_unchanged():
    data = np.linspace(-1, 1, 100, dtype=np.float32)
    assert np.array_equal(resample(data, 16000, 16000), data)


def test_resample_integer_ratio_length_and_dtype():
    out = resample(np.ones(48000, dtype=np.float32), 48000, 16000)
    assert out.shape == (16000,)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)


def test_resample_non_integer_ratio_length():
    out = resample(np.zeros(44100, dtype=np.float32), 44100, 16000)
    assert abs(out.size - 16000) <= 1


def test_resample_empty():
    assert resample(np.zeros(0, dtype=np.float32), 44100, 16000).size == 0


def test_levels():
    data = np.array([0.0, 0.5, -0.8, 0.2], dtype=np.float32)
    assert peak_level(data) == np.float32(0.8)
    assert 0 < rms_level(data) < 0.8
    assert peak_level(np.zeros(0, dtype=np.float32)) == 0.0


def test_clean_removes_markdown_and_links():
    text = "**Done.** See [the docs](https://example.com/x) or https://example.com now."
    assert clean_for_speech(text) == "Done. See the docs or a link now."


def test_clean_flattens_bullets_and_headings():
    text = "# Plan\n- first item\n- second item\n1. third"
    assert clean_for_speech(text) == "Plan first item second item third"


def test_clean_removes_emoji_and_backticks():
    assert clean_for_speech("Use `pip` \U0001F600 today") == "Use pip today"


def test_clean_keeps_underscores_in_names():
    assert clean_for_speech("open j_mine") == "open j_mine"
