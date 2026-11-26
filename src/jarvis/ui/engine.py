"""The bridge between the window and JARVIS's brain, voice and reminders.

The window only ever talks to this object, and this object only ever talks back with
Qt signals. Everything slow (the language model, speech, the weather) runs on worker
threads, so the window never freezes. The brain is still the only thing that decides
what to do, and the executor is still the only thing that does it: this adds no new way
to act, it only lets you see and drive the same pipeline from a window.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from datetime import datetime
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal

from jarvis.ai.ollama_client import OllamaClient, OllamaError
from jarvis.briefing import reminder_briefing
from jarvis.greeting import make_greeting
from jarvis.config import Settings
from jarvis.memory.store import MemoryStore
from jarvis.orchestrator import LLM_DOWN_SPEECH, State, VoiceAssistant
from jarvis.profile import Profile
from jarvis.reminders import ReminderScheduler
from jarvis.tools.media import hint_names
from jarvis.tools.drives import hint_names as drive_hint_names
from jarvis.tools.registry import ToolResult
from jarvis.tools.weather import current_chip, fetch_forecast
from jarvis.voice.interfaces import Trigger, VoiceError
from jarvis.wiring import build_brain, load_user_profile, open_store

logger = logging.getLogger(__name__)

CONFIRM_TIMEOUT_SECONDS = 60.0
WEATHER_UNAVAILABLE = "Weather unavailable"

# The window's names for the assistant's state.
STATE_NAMES = {
    State.IDLE: "idle",
    State.LISTENING: "listening",
    State.PROCESSING: "thinking",
    State.SPEAKING: "speaking",
}


class ButtonTrigger:
    """Push-to-talk from the on-screen microphone button, alongside the real hotkey.

    Tapping the button starts recording and tapping it again stops, so you don't have to
    hold the mouse down. The hotkey (if any) keeps working exactly as before.
    """

    description = "button"

    def __init__(self, inner: Trigger | None = None) -> None:
        self._inner = inner
        self._pressed = threading.Event()
        self._held = threading.Event()
        self._source = "button"

    # called from the window
    def button_down(self) -> None:
        self._held.set()
        self._pressed.set()

    def button_up(self) -> None:
        self._held.clear()

    @property
    def recording(self) -> bool:
        return self._held.is_set()

    # Trigger interface
    def start(self) -> None:
        if self._inner is not None:
            self._inner.start()

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()

    def wait_for_press(self, stop: threading.Event, timeout: float | None = None) -> bool | None:
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        while not stop.is_set():
            if self._pressed.is_set():
                self._pressed.clear()
                self._source = "button"
                return True
            slice_ = 0.05
            if deadline is not None:
                slice_ = min(slice_, max(0.0, deadline - time.monotonic()))
            if self._inner is not None:
                result = self._inner.wait_for_press(stop, slice_)
                if result is True:
                    self._source = "inner"
                    return True
                if result is False:
                    return False
            else:
                self._pressed.wait(slice_)
            if deadline is not None and time.monotonic() >= deadline:
                return None
        return False

    def wait_for_release(self, stop: threading.Event, timeout: float | None = None) -> None:
        import time

        if self._source == "inner" and self._inner is not None:
            self._inner.wait_for_release(stop, timeout)
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._held.is_set() and not stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                self._held.clear()
                return
            time.sleep(0.02)

    def interrupt_requested(self) -> bool:
        return self._pressed.is_set() or (self._inner is not None and self._inner.interrupt_requested())

    def clear(self) -> None:
        self._pressed.clear()
        if self._inner is not None:
            self._inner.clear()


class Engine(QObject):
    """Owns the brain, memory, voice and reminders, and reports everything with signals."""

    stateChanged = Signal(str)               # idle | listening | thinking | speaking
    messageAdded = Signal(str, str)          # role (user | assistant | system), text
    toolFinished = Signal(str, str, str)     # tool name, status, spoken message
    confirmRequested = Signal(int, str)      # request id, what JARVIS wants to do
    memoryChanged = Signal()
    healthChanged = Signal(bool, str)        # language model ready?, detail
    voiceChanged = Signal(bool, str)         # voice ready?, detail
    weatherChanged = Signal(str)             # e.g. "24° Partly cloudy"
    modelInfoChanged = Signal(str)           # e.g. "GPU 100% · last reply 2.1 s"

    def __init__(
        self,
        settings: Settings,
        *,
        use_voice: bool = True,
        client: Any = None,
        store: MemoryStore | None | bool = False,
        profile: Profile | None = None,
        voice_builder: Callable[[Settings, list[str]], Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.profile = profile if profile is not None else load_user_profile(settings)
        self.store: MemoryStore | None = open_store(settings) if store is False else store  # type: ignore[assignment]
        self.client = client or OllamaClient.from_settings(settings.ollama)
        self.brain = build_brain(settings, self.client, self.profile, self.store)
        if self.brain.executor is not None:
            self.brain.executor.add_listener(self._on_tool)
        self.owner_name = self.profile.owner_name or "Operator"

        self._use_voice = use_voice
        self._voice_builder = voice_builder
        self._assistant: VoiceAssistant | None = None
        self._trigger = ButtonTrigger()
        self._scheduler: ReminderScheduler | None = None
        self._loop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None
        self._voice_thread: threading.Thread | None = None
        self._pending: dict[int, asyncio.Future[bool]] = {}
        self._ids = itertools.count(1)
        self._started = False
        self._processor = ""
        self._greeted = False
        self._last_reply = ""
        self._voice_announced = False

    # ------------------------------------------------------------------ lifecycle

    @property
    def tool_names(self) -> list[str]:
        executor = self.brain.executor
        return executor.registry.names() if executor else []

    @property
    def voice_active(self) -> bool:
        return self._assistant is not None

    def greeting_text(self, model_ready: bool | None = None, now: datetime | None = None) -> str:
        moment = now or datetime.now()
        text = make_greeting(self.settings.ui.greeting_name, moment, model_ready)
        if self.store is not None:
            try:
                extra = reminder_briefing(self.store.pending_reminders(), moment, self.settings.ui.briefing_max)
            except Exception:
                logger.exception("Could not read reminders for the greeting")
                extra = ""
            # Put the reminders before the closing question so the question stays last.
            if extra:
                text = text.rsplit(" How can I help", 1)[0] + " " + extra + " How can I help" + text.rsplit(" How can I help", 1)[1]
        return text

    def _check_model_ready(self) -> bool | None:
        """Ask Ollama once, from a non-UI thread, so the spoken hello only claims what is true."""
        try:
            status = asyncio.run_coroutine_threadsafe(self.client.health(), self._loop).result(timeout=4)
            return bool(status.reachable and status.model_available)
        except Exception:
            return None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self.settings.ui.greet_on_start and not self._use_voice:
            self.messageAdded.emit("assistant", self.greeting_text())
        self._loop_thread = threading.Thread(target=self._run_loop, name="jarvis-brain", daemon=True)
        self._loop_thread.start()
        self._spawn(self._health_loop())
        self._spawn(self._warm_up())
        if self.profile.location is not None:
            self._spawn(self._weather_loop())
        if self.store is not None:
            self._scheduler = ReminderScheduler(
                self.store, self._announce, interval=self.settings.memory.reminder_check_seconds
            )
            self._scheduler.start()
        if self._use_voice:
            self._voice_thread = threading.Thread(target=self._run_voice, name="jarvis-voice", daemon=True)
            self._voice_thread.start()
        else:
            self.voiceChanged.emit(False, "Voice is off (text only)")

    def restart_voice(self) -> None:
        """Use new microphone/speakers: restart the voice part. The window stays open."""
        if not self._use_voice or not self._started:
            return

        def work() -> None:
            if self._assistant is not None:
                self._assistant.stop()
            if self._voice_thread is not None:
                self._voice_thread.join(timeout=5)
            self._voice_announced = False
            self.voiceChanged.emit(False, "Restarting voice...")
            self._voice_thread = threading.Thread(target=self._run_voice, name="jarvis-voice", daemon=True)
            self._voice_thread.start()

        threading.Thread(target=work, name="jarvis-voice-restart", daemon=True).start()

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._scheduler is not None:
            self._scheduler.stop()
        if self._assistant is not None:
            self._assistant.stop()
        for future in list(self._pending.values()):
            self._loop.call_soon_threadsafe(self._settle, future, False)
        if self._voice_thread is not None:
            self._voice_thread.join(timeout=3)
        if self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.client.aclose(), self._loop).result(timeout=3)
            except Exception:
                logger.exception("Could not close the Ollama client")
            self._loop.call_soon_threadsafe(self._cancel_tasks_and_stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=3)

    def _cancel_tasks_and_stop(self) -> None:
        for task in asyncio.all_tasks(self._loop):
            task.cancel()
        self._loop.call_later(0.05, self._loop.stop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _spawn(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    # ------------------------------------------------------------------ what the window can ask for

    def submit_text(self, text: str) -> None:
        """You typed a command."""
        text = text.strip()
        if text:
            self._spawn(self._text_turn(text))

    def run_quick(self, label: str, tool: str, arguments: dict[str, Any]) -> None:
        """A quick-command button: run one tool directly, without asking the language model."""
        self._spawn(self._tool_turn(label, tool, arguments))

    def answer_confirmation(self, request_id: int, approved: bool) -> None:
        future = self._pending.get(request_id)
        if future is not None:
            self._loop.call_soon_threadsafe(self._settle, future, approved)

    def talk_pressed(self) -> bool:
        """The on-screen microphone was tapped. Returns True if it is now recording."""
        if self._assistant is None:
            self.messageAdded.emit("system", "Voice isn't running. Start the app without --no-voice.")
            return False
        if self._trigger.recording:
            self._trigger.button_up()
            return False
        self._trigger.button_down()
        return True

    def reset_conversation(self) -> None:
        self.brain.reset()

    # ------------------------------------------------------------------ confirmations

    @staticmethod
    def _settle(future: asyncio.Future[bool], value: bool) -> None:
        if not future.done():
            future.set_result(value)

    async def _confirm(self, prompt: str) -> bool:
        """Ask you, in a dialog. The language model has no way to answer this itself."""
        request_id = next(self._ids)
        future: asyncio.Future[bool] = self._loop.create_future()
        self._pending[request_id] = future
        self.confirmRequested.emit(request_id, prompt)
        try:
            return await asyncio.wait_for(future, CONFIRM_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending.pop(request_id, None)

    # ------------------------------------------------------------------ turns

    async def _respond(self, text: str) -> str:
        """Ask the brain, and tell the dashboard how long it took and whether the model was needed."""
        started = time.monotonic()
        reply = await self.brain.respond(text, confirm=self._confirm)
        seconds = time.monotonic() - started
        note = " (instant, no model)" if self.brain.last_route == "fast" else ""
        self._last_reply = f"{seconds:.1f} s{note}"
        self._emit_model_info()
        return reply

    def _emit_model_info(self) -> None:
        parts = [p for p in (self._processor, f"last reply {self._last_reply}" if self._last_reply else "") if p]
        if parts:
            self.modelInfoChanged.emit(" · ".join(parts))

    async def _warm_up(self) -> None:
        """Load the model before the first question so the first answer is quick."""
        await self.brain.warm_up()
        processor = getattr(self.client, "processor", None)
        if processor is not None:
            self._processor = (await processor()) or ""
            self._emit_model_info()

    async def _text_turn(self, text: str) -> None:
        self.messageAdded.emit("user", text)
        self.stateChanged.emit("thinking")
        try:
            reply = await self._respond(text)
        except OllamaError as exc:
            logger.error("Model error: %s", exc)
            self.messageAdded.emit("system", str(exc))
            self._deliver(LLM_DOWN_SPEECH)
            return
        except Exception:
            logger.exception("Brain failed")
            self._deliver("I'm afraid something went wrong with that request.")
            return
        finally:
            self.stateChanged.emit("idle")
            self.memoryChanged.emit()
        self._deliver(reply)

    async def _tool_turn(self, label: str, tool: str, arguments: dict[str, Any]) -> None:
        executor = self.brain.executor
        self.messageAdded.emit("user", label)
        if executor is None:
            self._deliver("My tools are switched off.")
            return
        self.stateChanged.emit("thinking")
        try:
            result: ToolResult = await executor.run(tool, arguments, confirm=self._confirm)
        finally:
            self.stateChanged.emit("idle")
            self.memoryChanged.emit()
        self._deliver(result.message)

    def _deliver(self, text: str) -> None:
        """Show a reply, and say it aloud when the voice is running."""
        if self._assistant is not None and self.settings.ui.speak_typed_replies:
            self._assistant.say(text)  # shows it too, once it is spoken
        else:
            self.messageAdded.emit("assistant", text)

    def _announce(self, text: str) -> None:
        """A reminder came due."""
        if self._assistant is not None:
            self._assistant.say(text)
        else:
            self.messageAdded.emit("assistant", text)
        self.memoryChanged.emit()

    def _on_tool(self, name: str, result: ToolResult) -> None:
        self.toolFinished.emit(name, result.status.value, result.message)

    # ------------------------------------------------------------------ background checks

    async def _health_loop(self) -> None:
        while True:
            try:
                status = await self.client.health()
                ready = bool(status.reachable and status.model_available)
                self.healthChanged.emit(ready, status.detail or ("ready" if ready else "not ready"))
            except Exception as exc:
                self.healthChanged.emit(False, str(exc))
            await asyncio.sleep(self.settings.ui.health_check_seconds)

    async def _weather_loop(self) -> None:
        location = self.profile.location
        assert location is not None
        while True:
            try:
                data = await asyncio.to_thread(fetch_forecast, location)
                self.weatherChanged.emit(current_chip(data))
            except Exception as exc:
                logger.info("Weather chip not updated: %s", exc)
                self.weatherChanged.emit(WEATHER_UNAVAILABLE)
            await asyncio.sleep(self.settings.ui.weather_refresh_minutes * 60)

    # ------------------------------------------------------------------ voice

    def _run_voice(self) -> None:
        """Runs on its own thread: the push-to-talk loop."""
        try:
            hints = [*self.profile.spoken_names(), *hint_names(self.profile), *drive_hint_names(self.profile)]
            if self._voice_builder is not None:
                stack = self._voice_builder(self.settings, hints)
            else:
                from jarvis.voice.factory import build_voice_stack

                stack = build_voice_stack(
                    self.settings, extra_hints=hints, correction_terms=self.profile.spoken_names()
                )
        except VoiceError as exc:
            logger.error("Voice setup failed: %s", exc)
            self.voiceChanged.emit(False, str(exc))
            self.messageAdded.emit("system", f"Voice setup failed: {exc}. You can still type commands.")
            return
        except Exception as exc:
            logger.exception("Voice setup crashed")
            self.voiceChanged.emit(False, str(exc))
            return

        self._trigger = ButtonTrigger(stack.trigger)

        def respond(text: str) -> str:
            return asyncio.run_coroutine_threadsafe(self._respond(text), self._loop).result()

        def on_state(state: State) -> None:
            self.stateChanged.emit(STATE_NAMES.get(state, "idle"))
            if state is State.IDLE and not self._voice_announced:
                self._voice_announced = True
                self.voiceChanged.emit(True, "ready")
            if state in (State.IDLE, State.PROCESSING):
                self.memoryChanged.emit()

        voice = self.settings.voice
        assistant = VoiceAssistant(
            respond=respond,
            recorder=stack.recorder,
            transcriber=stack.transcriber,
            synthesizer=stack.synthesizer,
            player=stack.player,
            trigger=self._trigger,
            on_state=on_state,
            on_message=self.messageAdded.emit,
            min_speech_seconds=voice.min_speech_seconds,
            max_record_seconds=voice.max_record_seconds,
            silence_peak=voice.silence_peak,
        )
        self._assistant = assistant
        if self.settings.ui.greet_on_start and not self._greeted:
            self._greeted = True
            assistant.say(self.greeting_text(self._check_model_ready()))  # spoken once the speech models have loaded
        try:
            assistant.run()
        except VoiceError as exc:
            self.voiceChanged.emit(False, str(exc))
            self.messageAdded.emit("system", f"Voice error: {exc}")
        except Exception as exc:
            logger.exception("Voice loop crashed")
            self.voiceChanged.emit(False, str(exc))
        finally:
            self._assistant = None
