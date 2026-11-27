"""The voice conversation loop.

One turn: hold push-to-talk -> record -> transcribe -> ask the brain -> speak.
All hardware sits behind the small interfaces in jarvis.voice.interfaces, so the
loop itself is tested with fakes.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from enum import Enum

import numpy as np

from jarvis.ai.ollama_client import OllamaError
from jarvis.voice.audio_utils import clean_for_speech, peak_level, trim_leading_silence
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

MAX_FOLLOW_UPS = 3
ANNOUNCEMENT_POLL_SECONDS = 0.25  # how often an idle JARVIS checks for reminders to speak

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
        annotate: Callable[..., None] | None = None,
        ack: Callable[[], object] | None = None,
        wake_reply: str = "",
        follow_up: bool = False,
    ) -> None:
        self._wake_reply = wake_reply
        self._follow_up = follow_up
        self._by_voice = False
        self._last_reply = ""
        self._ack = ack
        self._annotate = annotate
        self._stt_seconds = 0.0
        self._tts_first = 0.0
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
        self._announcements: queue.Queue[str] = queue.Queue()
        self.state = State.IDLE

    # ------------------------------------------------------------------ lifecycle

    def stop(self) -> None:
        self._stop.set()

    def say(self, text: str) -> None:
        """Queue something to say (a reminder). Safe to call from any thread.

        It is spoken the next time JARVIS is idle, never in the middle of a conversation.
        """
        self._announcements.put(text)

    def run(self) -> None:
        """Run until stop() is called. Blocks."""
        self._notify("system", "Loading speech models...")
        self.transcriber.warm_up()
        self.synthesizer.warm_up()
        self.trigger.start()
        try:
            self._set_state(State.IDLE)
            while not self._stop.is_set():
                self._flush_announcements()
                pressed = self.trigger.wait_for_press(self._stop, ANNOUNCEMENT_POLL_SECONDS)
                if pressed is None:  # nothing pressed yet: look for reminders again
                    continue
                if not pressed:      # stop requested
                    break
                self._conversation()
                self._set_state(State.IDLE)
        finally:
            self.trigger.close()
            self._set_state(State.IDLE)

    def _conversation(self) -> None:
        """One start-to-finish exchange. After "Hey JARVIS" it says it is listening; if JARVIS then asks you
        a question, it listens for your answer without the wake word (a few times at most), like a car assistant."""
        by_voice = bool(getattr(self.trigger, "woke_by_voice", False))
        if by_voice and self._wake_reply:
            self._speak(self._wake_reply)
            time.sleep(0.25)  # let the speaker go quiet so the microphone does not hear JARVIS itself
        self.run_turn(by_voice=by_voice)
        follow_ups = 0
        while (
            by_voice and self._follow_up and follow_ups < MAX_FOLLOW_UPS and not self._stop.is_set()
            and self._last_reply.rstrip().endswith("?") and hasattr(self.trigger, "begin_follow_up")
        ):
            follow_ups += 1
            self.trigger.begin_follow_up()
            self.run_turn(by_voice=True, follow_up=True)

    def _flush_announcements(self) -> None:
        while True:
            try:
                text = self._announcements.get_nowait()
            except queue.Empty:
                return
            self._notify("assistant", text)
            self._speak(text)
            self._set_state(State.IDLE)

    def run_turn(self, by_voice: bool = False, follow_up: bool = False) -> None:
        """Handle one spoken exchange. Never raises for ordinary failures."""
        self._by_voice = by_voice
        self._last_reply = ""
        audio = self._listen(follow_up)
        if audio is None:
            return
        if self._ack is not None:
            threading.Thread(target=self._play_ack, daemon=True).start()  # plays while speech recognition runs
        text = self._transcribe(audio)
        if not text:
            self._notify("system", "I didn't catch that.")
            if by_voice:
                self._speak("Sorry, I didn't catch that.")
            return
        self._notify("user", text)
        reply = self._think(text)
        if reply is None:
            return
        self._notify("assistant", reply)
        self._last_reply = reply
        self._tts_first = 0.0
        self._speak(reply)
        if self._annotate is not None:
            try:
                self._annotate(via="voice", stt_s=self._stt_seconds, tts_first_s=self._tts_first, audio_s=audio.size / SAMPLE_RATE)
            except Exception:
                logger.exception("Could not annotate the turn log")

    # ------------------------------------------------------------------ stages

    def _listen(self, follow_up: bool = False) -> np.ndarray | None:
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

        if self._by_voice and audio.size >= self._min_samples and peak_level(audio) >= self._silence_peak:
            audio = trim_leading_silence(audio, SAMPLE_RATE)  # drop the wait before you started talking
        if self._by_voice and (audio.size < self._min_samples or peak_level(audio) < self._silence_peak):
            if not follow_up:
                self._notify("system", "Standing by. Say 'Hey JARVIS' when you need me.")
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

    def _play_ack(self) -> None:
        try:
            if self._ack is not None:
                self._ack()
        except Exception:
            logger.debug("Could not play the acknowledgement sound", exc_info=True)

    def _transcribe(self, audio: np.ndarray) -> str:
        self._set_state(State.PROCESSING)
        started = time.monotonic()
        try:
            text = self.transcriber.transcribe(audio)
        except Exception as exc:
            logger.exception("Transcription failed")
            self._notify("system", f"Speech recognition failed: {exc}")
            return ""
        self._stt_seconds = time.monotonic() - started
        logger.info("STT %.1fs for %.1fs of audio", self._stt_seconds, audio.size / SAMPLE_RATE)
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
                    self._tts_first = time.monotonic() - started
                    logger.info("TTS first audio after %.1fs", self._tts_first)
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
