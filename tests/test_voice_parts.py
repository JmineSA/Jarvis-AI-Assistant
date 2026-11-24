from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.voice.interfaces import AudioClip, VoiceError
from jarvis.voice.models import ensure_piper_voice, piper_voice_path
from jarvis.voice.stt import FasterWhisperTranscriber, build_prompt, is_prompt_echo
from jarvis.voice.tts import PiperSynthesizer


# ------------------------------------------------------------------------------ STT


class FakeWhisper:
    def __init__(self, texts):
        self.texts = texts
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return iter(SimpleNamespace(text=t) for t in self.texts), object()


def test_transcriber_joins_segments_and_uses_fast_settings():
    model = FakeWhisper([" Open ", "VS Code. "])
    stt = FasterWhisperTranscriber(hint_words=["VS Code"], model_factory=lambda: model)
    text = stt.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "Open VS Code."
    assert model.kwargs["beam_size"] == 1
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["language"] == "en"
    assert model.kwargs["initial_prompt"] == "Vocabulary: VS Code."


def test_transcriber_discards_prompt_echo():
    model = FakeWhisper(["Vocabulary: JARVIS, Power BI."])
    stt = FasterWhisperTranscriber(hint_words=["JARVIS", "Power BI"], model_factory=lambda: model)
    assert stt.transcribe(np.zeros(16000, dtype=np.float32)) == ""


def test_prompt_helpers():
    assert build_prompt([]) is None
    assert is_prompt_echo("vocabulary: a, b.", "Vocabulary: a, b.")
    assert not is_prompt_echo("open chrome", "Vocabulary: a, b.")
    assert not is_prompt_echo("anything", None)


def test_model_load_failure_becomes_voice_error():
    def broken():
        raise OSError("offline")

    stt = FasterWhisperTranscriber(model_factory=broken)
    with pytest.raises(VoiceError, match="speech model"):
        stt.warm_up()


# ------------------------------------------------------------------------------ TTS


class FakePiper:
    def synthesize(self, text, **kwargs):
        for sentence in text.split("."):
            if sentence.strip():
                yield SimpleNamespace(
                    audio_float_array=np.zeros(220, dtype=np.float64), sample_rate=22050
                )


def test_synthesizer_yields_one_clip_per_sentence():
    tts = PiperSynthesizer(Path("voice.onnx"), voice_factory=lambda: FakePiper())
    clips = list(tts.synthesize("One. Two. Three."))
    assert len(clips) == 3
    assert all(isinstance(c, AudioClip) and c.sample_rate == 22050 for c in clips)
    assert clips[0].samples.dtype == np.float32


def test_voice_load_failure_becomes_voice_error():
    def broken():
        raise OSError("corrupt")

    tts = PiperSynthesizer(Path("voice.onnx"), voice_factory=broken)
    with pytest.raises(VoiceError, match="voice.onnx"):
        tts.warm_up()


# --------------------------------------------------------------------------- models


def test_existing_voice_is_not_downloaded_again(tmp_path):
    onnx = piper_voice_path(tmp_path, "en_GB-alan-medium")
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"x")
    onnx.with_name(onnx.name + ".json").write_text("{}")

    def downloader(voice, target):
        raise AssertionError("should not download")

    assert ensure_piper_voice(tmp_path, "en_GB-alan-medium", downloader) == onnx


def test_missing_voice_is_downloaded(tmp_path):
    def downloader(voice, target):
        (target / f"{voice}.onnx").write_bytes(b"model")
        (target / f"{voice}.onnx.json").write_text("{}")

    path = ensure_piper_voice(tmp_path, "en_GB-alan-medium", downloader)
    assert path.exists()


def test_failed_download_leaves_no_partial_files(tmp_path):
    def downloader(voice, target):
        (target / f"{voice}.onnx").write_bytes(b"half")
        raise OSError("connection reset")

    with pytest.raises(VoiceError, match="download"):
        ensure_piper_voice(tmp_path, "en_GB-alan-medium", downloader)
    assert not piper_voice_path(tmp_path, "en_GB-alan-medium").exists()


def test_interrupted_download_also_cleans_up(tmp_path):
    def downloader(voice, target):
        (target / f"{voice}.onnx").write_bytes(b"half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ensure_piper_voice(tmp_path, "en_GB-alan-medium", downloader)
    assert not piper_voice_path(tmp_path, "en_GB-alan-medium").exists()
