"""The voice conversation loop.

One turn: hold push-to-talk -> record -> transcribe -> ask the brain -> speak.
All hardware sits behind the small interfaces in jarvis.voice.interfaces, so the
loop itself is tested with fakes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum

import numpy as np

from jarvis.ai.ollama_client import OllamaError
from jarvis.voice.audio_utils import clean_for_speech, peak_level
from jarvis.voice.interfaces import (
    SAMPLE_RATE,
    Player,
    Recorder,
    Synthesizer,
    Transcriber,
    Trigger,
    VoiceError,
)

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


StateCallback = Callable[[State], None]
MessageCallback = Callable[[str, str], None]  # (role, text); role: user | assistant | system

LLM_DOWN_SPEECH = "I'm having trouble reaching my language model."
GENERIC_ERROR_SPEECH = "I'm afraid something went wrong with that request."


class VoiceAssistant:
    def __init__(
        self,
        *,
        respond: Callable[[str], str],
        recorder: Recorder,
        transcriber: Transcriber,
        synthesizer: Synthesizer,
        player: Player,
        trigger: Trigger,
        on_state: StateCallback | None = None,
        on_message: MessageCallback | None = None,
        min_speech_seconds: float = 0.3,
        max_record_seconds: float = 30.0,
        silence_peak: float = 0.01,
    ) -> None:
        self._respond = respond
        self.recorder = recorder
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.player = player
        self.trigger = trigger
        self._on_state = on_state
        self._on_message = on_message
        self._min_samples = int(min_speech_seconds * SAMPLE_RATE)
        self._max_record_seconds = max_record_seconds
        self._silence_peak = silence_peak
        self._stop = threading.Event()
        self.state = State.IDLE

    # ------------------------------------------------------------------ lifecycle

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Run until stop() is called. Blocks."""
        self._notify("system", "Loading speech models...")
        self.transcriber.warm_up()
        self.synthesizer.warm_up()
        self.trigger.start()
        try:
            self._set_state(State.IDLE)
            while not self._stop.is_set():
                if not self.trigger.wait_for_press(self._stop):
                    break
                self.run_turn()
                self._set_state(State.IDLE)
        finally:
            self.trigger.close()
            self._set_state(State.IDLE)

    def run_turn(self) -> None:
        """Handle one spoken exchange. Never raises for ordinary failures."""
        audio = self._listen()
        if audio is None:
            return
        text = self._transcribe(audio)
        if not text:
            self._notify("system", "I didn't catch that.")
            return
        self._notify("user", text)
        reply = self._think(text)
        if reply is None:
            return
        self._notify("assistant", reply)
        self._speak(reply)

    # ------------------------------------------------------------------ stages

    def _listen(self) -> np.ndarray | None:
        self._set_state(State.LISTENING)
        try:
            self.recorder.start()
        except VoiceError as exc:
            self._notify("system", str(exc))
            return None
        self.trigger.wait_for_release(self._stop, self._max_record_seconds)
        try:
            audio = self.recorder.stop()
        except Exception as exc:
            logger.exception("Recorder failed to stop")
            self._notify("system", f"Microphone error: {exc}")
            return None

        if audio.size < self._min_samples:
            self._notify("system", "That was too short. Hold the key while you speak.")
            return None
        if peak_level(audio) < self._silence_peak:
            self._notify(
                "system",
                "I heard no sound. Check the microphone (run: python -m jarvis voice-check).",
            )
            return None
        return audio

    def _transcribe(self, audio: np.ndarray) -> str:
        self._set_state(State.PROCESSING)
        started = time.monotonic()
        try:
            text = self.transcriber.transcribe(audio)
        except Exception as exc:
            logger.exception("Transcription failed")
            self._notify("system", f"Speech recognition failed: {exc}")
            return ""
        logger.info("STT %.1fs for %.1fs of audio", time.monotonic() - started, audio.size / SAMPLE_RATE)
        return text.strip()

    def _think(self, text: str) -> str | None:
        started = time.monotonic()
        try:
            reply = self._respond(text)
        except OllamaError as exc:
            logger.error("Model error: %s", exc)
            self._notify("system", str(exc))
            self._speak(LLM_DOWN_SPEECH)
            return None
        except Exception:
            logger.exception("Brain failed")
            self._speak(GENERIC_ERROR_SPEECH)
            return None
        logger.info("LLM %.1fs", time.monotonic() - started)
        return reply

    def _speak(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if not spoken:
            return
        self._set_state(State.SPEAKING)
        started = time.monotonic()
        first = True
        try:
            for clip in self.synthesizer.synthesize(spoken):
                if self.trigger.interrupt_requested():
                    break  # the user pressed talk: stop and listen
                if first:
                    logger.info("TTS first audio after %.1fs", time.monotonic() - started)
                    first = False
                if not self.player.play(clip, self.trigger.interrupt_requested):
                    break
        except Exception as exc:
            logger.exception("Speech output failed")
            self._notify("system", f"Voice output failed ({exc}). The reply is shown as text.")

    # ------------------------------------------------------------------ callbacks

    def _set_state(self, state: State) -> None:
        self.state = state
        if self._on_state:
            try:
                self._on_state(state)
            except Exception:
                logger.exception("State callback failed")

    def _notify(self, role: str, text: str) -> None:
        logger.info("[%s] %s", role, text)
        if self._on_message:
            try:
                self._on_message(role, text)
            except Exception:
                logger.exception("Message callback failed")
