"""Pure helper functions for audio and text. No hardware needed."""

from __future__ import annotations

import re

import numpy as np


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Convert mono audio to another sample rate (good enough for speech)."""
    samples = np.asarray(samples, dtype=np.float32)
    if from_rate == to_rate or samples.size == 0:
        return samples
    if from_rate % to_rate == 0:  # e.g. 48000 -> 16000: average groups (acts as a low-pass)
        factor = from_rate // to_rate
        usable = (samples.size // factor) * factor
        return samples[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)
    duration = samples.size / from_rate
    out_len = max(1, round(duration * to_rate))
    old_t = np.linspace(0.0, duration, samples.size, endpoint=False)
    new_t = np.linspace(0.0, duration, out_len, endpoint=False)
    return np.interp(new_t, old_t, samples).astype(np.float32)


def peak_level(samples: np.ndarray) -> float:
    """Loudest absolute sample, 0.0 to 1.0."""
    return float(np.max(np.abs(samples))) if samples.size else 0.0


def rms_level(samples: np.ndarray) -> float:
    """Average loudness, 0.0 to 1.0."""
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def level_speech(
    samples: np.ndarray, target_peak: float = 0.7, max_gain: float = 25.0, floor: float = 0.003
) -> np.ndarray:
    """Raise a quiet recording so a calm voice is as clear to Whisper as a loud one.

    Recordings with almost no sound (below `floor`) are returned unchanged, so background
    hiss is not boosted into words. Loud recordings are never made louder.
    """
    samples = np.asarray(samples, dtype=np.float32)
    peak = peak_level(samples)
    if peak < floor or peak >= target_peak:
        return samples
    gain = min(target_peak / peak, max_gain)
    return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)


_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")
_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)?[^)]*\)")
_URL = re.compile(r"https?://\S+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


def clean_for_speech(text: str) -> str:
    """Strip markdown, links and emoji so the voice does not read symbols aloud."""
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _BULLET.sub("", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("`", "").replace("*", "")
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_leading_silence(samples: np.ndarray, rate: int, keep_seconds: float = 0.3) -> np.ndarray:
    """Cut the quiet lead-in (the wait for you to start talking), keeping a little before the first word."""
    frame = max(1, int(rate * 0.02))
    if samples.size < frame * 2:
        return samples
    count = samples.size // frame
    peaks = np.abs(samples[: count * frame]).reshape(count, frame).max(axis=1)
    loud = np.flatnonzero(peaks >= max(0.012, 0.2 * float(peaks.max())))
    if loud.size == 0:
        return samples
    return samples[max(0, int(loud[0] * frame - keep_seconds * rate)):]
