"""Say "Hey JARVIS" instead of holding the push-to-talk key.

Uses the open-source openWakeWord "hey_jarvis" model through onnxruntime (already installed
with Piper). Three small files (about 3.7 MB in all) are downloaded once by
`python -m jarvis wake-setup`. Everything runs on your PC: the microphone is only listened to
for that phrase and the sound is thrown away. After the phrase, JARVIS records until you stop
talking, then carries on exactly as it does after the push-to-talk key.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.voice.interfaces import SAMPLE_RATE, AudioClip, Trigger, VoiceError

logger = logging.getLogger(__name__)

CHUNK = 1280  # 80 ms at 16 kHz: the size the model works in
RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
MODEL_FILES = ("melspectrogram.onnx", "embedding_model.onnx", "hey_jarvis_v0.1.onnx")


def wake_dir(models_dir: Path) -> Path:
    return models_dir / "wakeword"


def wake_models_present(models_dir: Path) -> bool:
    return all((wake_dir(models_dir) / name).is_file() for name in MODEL_FILES)


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as out:
        out.write(response.read())


def ensure_wake_models(
    models_dir: Path, downloader: Callable[[str, Path], None] = _download, say: Callable[[str], None] = print
) -> Path:
    """Download the three model files (about 3.7 MB) if they are missing."""
    folder = wake_dir(models_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        target = folder / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        say(f"Downloading {name} ...")
        temp = target.with_suffix(".part")
        try:
            downloader(RELEASE + name, temp)
            temp.replace(target)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise VoiceError(f"Could not download {name}: {exc}. Check your internet connection.") from exc
    return folder


# --------------------------------------------------------------------------- the detector


class WakeWordDetector:
    """Feed it 80 ms chunks of microphone audio; it returns how sure it is that "hey jarvis" was said."""

    def __init__(self, folder: Path, session_factory: Callable[[Path], Any] | None = None) -> None:
        make = session_factory or self._session
        self._mel = make(folder / "melspectrogram.onnx")
        self._embed = make(folder / "embedding_model.onnx")
        self._wake = make(folder / "hey_jarvis_v0.1.onnx")
        self.reset()

    @staticmethod
    def _session(path: Path) -> Any:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1  # one quiet core: this runs all the time
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def reset(self) -> None:
        self._raw: deque[int] = deque(maxlen=SAMPLE_RATE * 10)
        self._mel_buffer = np.ones((76, 32), dtype=np.float32)
        noise = np.random.default_rng(0).integers(-1000, 1000, SAMPLE_RATE * 4).astype(np.int16)
        spec = self._melspec(noise)
        windows = [spec[i: i + 76] for i in range(0, spec.shape[0], 8) if spec[i: i + 76].shape[0] == 76]
        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        self._features = np.atleast_2d(self._embed.run(None, {"input_1": batch})[0].squeeze())
        self._pending = np.zeros(0, dtype=np.int16)
        self.chunks_seen = 0

    def _melspec(self, samples_int16: np.ndarray) -> np.ndarray:
        x = samples_int16.astype(np.float32)[None, :]
        out = self._mel.run(None, {"input": x})[0]
        return np.squeeze(out) / 10 + 2

    def feed(self, audio: np.ndarray) -> float:
        """`audio`: mono float32 in [-1, 1] at 16 kHz, any length. Returns the latest score (0 to 1)."""
        pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
        pcm = np.concatenate([self._pending, pcm])
        score = 0.0
        while pcm.size >= CHUNK:
            chunk, pcm = pcm[:CHUNK], pcm[CHUNK:]
            score = self._chunk(chunk)
        self._pending = pcm
        return score

    def _chunk(self, chunk: np.ndarray) -> float:
        self._raw.extend(chunk.tolist())
        recent = np.array(list(self._raw)[-(CHUNK + 160 * 3):], dtype=np.int16)
        spec = self._melspec(recent).reshape(-1, 32)
        self._mel_buffer = np.vstack((self._mel_buffer, spec))[-970:]
        window = self._mel_buffer[-76:].astype(np.float32)[None, :, :, None]
        embedding = np.atleast_2d(self._embed.run(None, {"input_1": window})[0].squeeze())
        self._features = np.vstack((self._features, embedding))[-120:]
        self.chunks_seen += 1
        frames = self._features[-16:][None, :, :].astype(np.float32)
        name = self._wake.get_inputs()[0].name
        return float(np.squeeze(self._wake.run(None, {name: frames})[0]))


# --------------------------------------------------------------------------- the trigger


def make_chime(rate: int = 22_050) -> AudioClip:
    """A short rising two-note beep that says "I heard you"."""
    t = np.linspace(0, 0.09, int(rate * 0.09), endpoint=False)
    fade = np.minimum(1, np.minimum(t / 0.01, (0.09 - t) / 0.02))
    notes = [np.sin(2 * np.pi * f * t) * fade * 0.25 for f in (660, 990)]
    return AudioClip(np.concatenate(notes).astype(np.float32), rate)


def make_ack(rate: int = 22_050) -> AudioClip:
    """One soft low tick that says "got it, working on it" the moment you stop talking."""
    t = np.linspace(0, 0.07, int(rate * 0.07), endpoint=False)
    fade = np.minimum(1, np.minimum(t / 0.01, (0.07 - t) / 0.03))
    return AudioClip((np.sin(2 * np.pi * 520 * t) * fade * 0.15).astype(np.float32), rate)


class WakeWordTrigger:
    """Push-to-talk plus "Hey JARVIS": either one starts a turn.

    The microphone is only open for the wake word while JARVIS is waiting. It is closed before
    recording starts, so the recorder never fights it for the device.
    """

    def __init__(
        self,
        hotkey: Trigger,
        detector: WakeWordDetector,
        device: int | str | None,
        level: Callable[[float], float],
        threshold: float = 0.5,
        silence_seconds: float = 1.2,
        first_word_seconds: float = 6.0,
        chime: Callable[[], None] | None = None,
        stream_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._hotkey = hotkey
        self._detector = detector
        self._device = device
        self._level = level                # level(seconds) -> peak of the most recent audio
        self._threshold = threshold
        self._silence = silence_seconds
        self._first_word = first_word_seconds
        self._chime = chime
        self._stream_factory = stream_factory
        self._clock = clock
        self._armed = threading.Event()    # True while JARVIS is waiting to be woken
        self._session = threading.Lock()   # held while the wake microphone is open
        self._woken = threading.Event()
        self._closing = threading.Event()
        self._by_voice = False
        self._thread: threading.Thread | None = None
        self.last_score = 0.0

    @property
    def description(self) -> str:
        return f"{getattr(self._hotkey, 'description', 'the talk key')} or say 'Hey JARVIS'"

    @property
    def woke_by_voice(self) -> bool:
        """True if the turn that just started was begun by saying "Hey JARVIS" (not the key)."""
        return self._by_voice

    def begin_follow_up(self) -> None:
        """Listen for an answer without the wake word (JARVIS just asked you a question)."""
        self._stand_down()
        self._woken.clear()
        self._by_voice = True

    # ------------------------------------------------------------------ listening thread

    def _open(self) -> Any:
        if self._stream_factory is not None:
            return self._stream_factory()
        import sounddevice as sd

        from jarvis.voice.input import resolve_device

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=resolve_device(self._device), blocksize=CHUNK)
        stream.start()
        return stream

    def _listen(self) -> None:
        while not self._closing.is_set():
            if not self._armed.wait(0.1):
                continue
            with self._session:
                if not self._armed.is_set():
                    continue
                stream = None
                try:
                    stream = self._open()
                    self._detector.reset()
                    while self._armed.is_set() and not self._closing.is_set():
                        block, _overflow = stream.read(CHUNK)
                        score = self._detector.feed(np.asarray(block).reshape(-1))
                        self.last_score = score
                        if self._detector.chunks_seen > 12 and score >= self._threshold:  # skip the first second
                            logger.info("Wake word heard (score %.2f)", score)
                            self._by_voice = True
                            self._armed.clear()
                            self._woken.set()
                except Exception:
                    logger.exception("Wake word listening stopped; push-to-talk still works.")
                    self._armed.clear()
                finally:
                    if stream is not None:
                        try:
                            stream.stop()
                            stream.close()
                        except Exception:
                            logger.debug("Closing the wake word microphone failed", exc_info=True)

    # ------------------------------------------------------------------ Trigger interface

    def start(self) -> None:
        self._hotkey.start()
        self._thread = threading.Thread(target=self._listen, name="jarvis-wakeword", daemon=True)
        self._thread.start()
        logger.info("Wake word on: say 'Hey JARVIS' (threshold %.2f).", self._threshold)

    def close(self) -> None:
        self._closing.set()
        self._armed.clear()
        self._hotkey.close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _stand_down(self) -> None:
        """Close the wake microphone (and wait until it is closed) before anything else records."""
        self._armed.clear()
        if self._session.acquire(timeout=2):  # waits until the listening thread has closed the microphone
            self._session.release()

    def wait_for_press(self, stop: threading.Event, timeout: float | None = None) -> bool | None:
        deadline = None if timeout is None else self._clock() + timeout
        self._by_voice = False
        if not self._woken.is_set():
            self._armed.set()
        while not stop.is_set():
            pressed = self._hotkey.wait_for_press(stop, 0.05)
            if pressed:
                self._by_voice = False
                self._stand_down()
                return True
            if pressed is False:
                self._stand_down()
                return False
            if self._woken.is_set():
                self._woken.clear()
                self._stand_down()
                if self._chime is not None:
                    try:
                        self._chime()
                    except Exception:
                        logger.debug("Wake chime failed", exc_info=True)
                return True
            if deadline is not None and self._clock() >= deadline:
                return None
        self._stand_down()
        return False

    def wait_for_release(self, stop: threading.Event, timeout: float | None = None) -> None:
        if not self._by_voice:
            self._hotkey.wait_for_release(stop, timeout)
            return
        # Spoken wake: record until you have said something and then gone quiet.
        start = self._clock()
        floor_peaks: list[float] = []
        spoke_at: float | None = None
        last_loud = start
        while not stop.is_set():
            now = self._clock()
            if timeout is not None and now - start >= timeout:
                return
            peak = self._level(0.15)
            if now - start < 0.4:
                floor_peaks.append(peak)           # the first moments set the background noise level
            else:
                floor = float(np.median(floor_peaks)) if floor_peaks else 0.0
                loud = peak > max(0.012, floor * 3)
                if loud:
                    spoke_at = spoke_at or now
                    last_loud = now
                elif spoke_at is not None and now - last_loud >= self._silence:
                    return
                elif spoke_at is None and now - start >= self._first_word:
                    return                         # woke up but never said anything
            time.sleep(0.05)

    def interrupt_requested(self) -> bool:
        return self._hotkey.interrupt_requested()

    def clear(self) -> None:
        self._hotkey.clear()
