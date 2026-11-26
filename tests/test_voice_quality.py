import numpy as np

from jarvis.voice.audio_utils import level_speech, peak_level
from jarvis.voice.correct import VocabularyCorrector
from jarvis.voice.stt import FasterWhisperTranscriber


def test_quiet_speech_is_raised_but_hiss_and_loud_speech_are_not():
    quiet = np.full(1000, 0.04, dtype=np.float32)
    assert abs(peak_level(level_speech(quiet)) - 0.7) < 0.01
    hiss = np.full(1000, 0.001, dtype=np.float32)
    assert np.array_equal(level_speech(hiss), hiss)
    loud = np.full(1000, 0.9, dtype=np.float32)
    assert np.array_equal(level_speech(loud), loud)
    assert level_speech(np.array([], dtype=np.float32)).size == 0


def test_gain_is_capped():
    faint = np.full(1000, 0.004, dtype=np.float32)
    assert peak_level(level_speech(faint)) <= 0.004 * 25 + 1e-6


def test_corrector_fixes_near_misses_only():
    c = VocabularyCorrector(["Power BI", "Jupyter", "Databricks", "VS Code", "LightsOn"])
    assert c.correct("open power by") == "open Power BI"
    assert c.correct("open Jupiter please") == "open Jupyter please"
    assert c.correct("open data bricks") == "open Databricks"
    assert c.correct("open lights on folder") == "open LightsOn folder"
    assert c.correct("this is important") == "this is important"
    assert c.correct("open Power BI") == "open Power BI"
    assert c.correct("") == ""


def test_corrector_never_touches_short_words_and_empty_vocab():
    c = VocabularyCorrector(["BI", "Go"])
    assert not c
    assert c.correct("go to bi") == "go to bi"


class _Seg:
    def __init__(self, text):
        self.text = text


class _Model:
    def __init__(self):
        self.audio = None
        self.beam = None

    def transcribe(self, audio, **kw):
        self.audio, self.beam = audio, kw["beam_size"]
        return [_Seg("open power by")], None


def test_transcriber_levels_audio_corrects_words_and_uses_beam():
    model = _Model()
    t = FasterWhisperTranscriber(model_factory=lambda: model, beam_size=3, correction_terms=["Power BI"])
    text = t.transcribe(np.full(1600, 0.04, dtype=np.float32))
    assert text == "open Power BI" and model.beam == 3 and peak_level(model.audio) > 0.5


def test_demo_speaks_each_voice(tmp_path, monkeypatch):
    from jarvis import diagnostics
    from jarvis.config import Settings

    spoken, lines = [], []

    class Synth:
        def __init__(self, path, scale):
            spoken.append(path.name)

        def synthesize(self, text):
            yield "clip"

    class Player:
        def __init__(self, device):
            pass

        def play(self, clip, stop):
            return True

    monkeypatch.setattr("jarvis.voice.models.ensure_piper_voice", lambda d, n: tmp_path / f"{n}.onnx")
    monkeypatch.setattr("jarvis.voice.tts.PiperSynthesizer", Synth)
    monkeypatch.setattr("jarvis.voice.output.SoundDevicePlayer", Player)
    assert diagnostics.run_voice_demo(Settings(), ["a", "b"], say=lines.append) == 0
    assert spoken == ["a.onnx", "b.onnx"]
