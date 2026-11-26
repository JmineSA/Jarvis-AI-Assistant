import numpy as np

from jarvis.ai.ollama_client import OllamaUnavailableError
from jarvis.orchestrator import (
    GENERIC_ERROR_SPEECH,
    LLM_DOWN_SPEECH,
    State,
    VoiceAssistant,
)
from jarvis.voice.interfaces import SAMPLE_RATE, AudioClip, VoiceError


def speech(seconds=1.0, level=0.3):
    return np.full(int(SAMPLE_RATE * seconds), level, dtype=np.float32)


class FakeRecorder:
    def __init__(self, audio=None, start_error=None):
        self.audio = speech() if audio is None else audio
        self.start_error = start_error
        self.started = 0

    def start(self):
        if self.start_error:
            raise self.start_error
        self.started += 1

    def stop(self):
        return self.audio


class FakeTranscriber:
    def __init__(self, text="hello jarvis", error=None):
        self.text, self.error = text, error
        self.warmed = False
        self.heard = []

    def warm_up(self):
        self.warmed = True

    def transcribe(self, audio):
        self.heard.append(audio)
        if self.error:
            raise self.error
        return self.text


class FakeSynth:
    def __init__(self, sentences=2, error=None):
        self.sentences, self.error = sentences, error
        self.texts = []
        self.warmed = False

    def warm_up(self):
        self.warmed = True

    def synthesize(self, text):
        self.texts.append(text)
        if self.error:
            raise self.error
        for _ in range(self.sentences):
            yield AudioClip(np.zeros(100, dtype=np.float32), 22050)


class FakePlayer:
    def __init__(self):
        self.played = 0

    def play(self, clip, should_stop):
        if should_stop():
            return False
        self.played += 1
        return True


class FakeTrigger:
    """Scripted push-to-talk. `interrupt_after` makes interrupt_requested() flip to True."""

    def __init__(self, presses=1, timeouts=0):
        self.presses = presses
        self.timeouts = timeouts  # how many idle polls (None) to return before pressing
        self.started = self.closed = False
        self.interrupt = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def wait_for_press(self, stop, timeout=None):
        if self.timeouts > 0:
            self.timeouts -= 1
            return None
        if self.presses <= 0 or stop.is_set():
            return False
        self.presses -= 1
        return True

    def wait_for_release(self, stop, timeout=None):
        pass

    def interrupt_requested(self):
        return self.interrupt

    def clear(self):
        self.interrupt = False


def build(respond=None, recorder=None, transcriber=None, synth=None, player=None, trigger=None):
    messages, states = [], []
    parts = dict(
        recorder=recorder or FakeRecorder(),
        transcriber=transcriber or FakeTranscriber(),
        synthesizer=synth or FakeSynth(),
        player=player or FakePlayer(),
        trigger=trigger or FakeTrigger(),
    )
    assistant = VoiceAssistant(
        respond=respond or (lambda text: "Good evening."),
        on_state=states.append,
        on_message=lambda role, text: messages.append((role, text)),
        **parts,
    )
    return assistant, parts, messages, states


def test_full_turn_runs_every_stage_in_order():
    assistant, parts, messages, states = build(respond=lambda text: f"You said {text}.")
    assistant.run()

    assert messages[-2:] == [("user", "hello jarvis"), ("assistant", "You said hello jarvis.")]
    assert parts["synthesizer"].texts == ["You said hello jarvis."]
    assert parts["player"].played == 2
    assert states == [
        State.IDLE, State.LISTENING, State.PROCESSING, State.SPEAKING, State.IDLE, State.IDLE,
    ]


def test_run_warms_up_and_manages_trigger():
    assistant, parts, *_ = build(trigger=FakeTrigger(presses=0))
    assistant.run()
    assert parts["transcriber"].warmed and parts["synthesizer"].warmed
    assert parts["trigger"].started and parts["trigger"].closed


def test_too_short_recording_is_ignored():
    assistant, parts, messages, _ = build(recorder=FakeRecorder(speech(0.1)))
    assistant.run_turn()
    assert parts["transcriber"].heard == []
    assert any("too short" in text for _, text in messages)


def test_silent_recording_gives_microphone_hint():
    assistant, parts, messages, _ = build(recorder=FakeRecorder(speech(1.0, level=0.0)))
    assistant.run_turn()
    assert parts["transcriber"].heard == []
    assert any("no sound" in text for _, text in messages)


def test_microphone_failure_is_reported_not_raised():
    assistant, _, messages, _ = build(recorder=FakeRecorder(start_error=VoiceError("no mic")))
    assistant.run_turn()
    assert ("system", "no mic") in messages


def test_nothing_recognised_does_not_call_the_brain():
    calls = []
    assistant, _, messages, _ = build(
        respond=lambda text: calls.append(text) or "x", transcriber=FakeTranscriber(text="")
    )
    assistant.run_turn()
    assert calls == []
    assert ("system", "I didn't catch that.") in messages


def test_transcription_error_is_reported_not_raised():
    assistant, _, messages, _ = build(transcriber=FakeTranscriber(error=RuntimeError("boom")))
    assistant.run_turn()
    assert any("recognition failed" in text for _, text in messages)


def test_model_down_speaks_apology():
    def respond(text):
        raise OllamaUnavailableError("Cannot reach Ollama.")

    assistant, parts, messages, _ = build(respond=respond)
    assistant.run_turn()
    assert parts["synthesizer"].texts == [LLM_DOWN_SPEECH]
    assert ("system", "Cannot reach Ollama.") in messages


def test_unexpected_brain_error_speaks_generic_message():
    def respond(text):
        raise RuntimeError("bug")

    assistant, parts, *_ = build(respond=respond)
    assistant.run_turn()
    assert parts["synthesizer"].texts == [GENERIC_ERROR_SPEECH]


def test_markdown_is_cleaned_before_speaking():
    assistant, parts, *_ = build(respond=lambda text: "**Done.** Opening it now.")
    assistant.run_turn()
    assert parts["synthesizer"].texts == ["Done. Opening it now."]


def test_pressing_talk_while_speaking_cuts_speech_off():
    trigger = FakeTrigger()

    class InterruptingPlayer(FakePlayer):
        def play(self, clip, should_stop):
            self.played += 1
            trigger.interrupt = True  # user presses the key during the first sentence
            return not should_stop()

    player = InterruptingPlayer()
    assistant, parts, *_ = build(player=player, trigger=trigger, synth=FakeSynth(sentences=3))
    assistant.run_turn()
    assert player.played == 1


def test_speech_failure_falls_back_to_text():
    assistant, _, messages, _ = build(synth=FakeSynth(error=VoiceError("no audio")))
    assistant.run_turn()
    assert ("assistant", "Good evening.") in messages
    assert any("Voice output failed" in text for _, text in messages)


def test_failing_callbacks_do_not_break_the_loop():
    assistant, parts, *_ = build()
    assistant._on_message = lambda role, text: (_ for _ in ()).throw(RuntimeError("ui"))
    assistant._on_state = lambda state: (_ for _ in ()).throw(RuntimeError("ui"))
    assistant.run_turn()
    assert parts["player"].played == 2


def test_stop_ends_the_loop():
    trigger = FakeTrigger(presses=10)
    assistant, *_ = build(trigger=trigger)
    assistant.stop()
    assistant.run()
    assert trigger.presses == 10  # never consumed a press


def test_announcements_are_spoken_while_idle_before_waiting_for_a_press():
    assistant, parts, messages, _ = build(trigger=FakeTrigger(presses=0, timeouts=2))
    assistant.say("Reminder: call the bank.")
    assistant.run()
    assert parts["synthesizer"].texts == ["Reminder: call the bank."]
    assert ("assistant", "Reminder: call the bank.") in messages


def test_announcement_queued_from_another_thread_is_spoken_on_a_later_poll():
    import threading

    trigger = FakeTrigger(presses=0, timeouts=3)
    assistant, parts, *_ = build(trigger=trigger)
    original = trigger.wait_for_press

    def poll(stop, timeout=None):
        if trigger.timeouts == 2:  # arrives while JARVIS is idle
            threading.Thread(target=assistant.say, args=("Reminder: stretch.",)).start()
        return original(stop, timeout)

    trigger.wait_for_press = poll
    assistant.run()
    import time

    time.sleep(0.05)
    assistant._flush_announcements()
    assert parts["synthesizer"].texts == ["Reminder: stretch."]


def test_idle_polling_does_not_start_a_turn():
    assistant, parts, messages, _ = build(trigger=FakeTrigger(presses=1, timeouts=3))
    assistant.run()
    assert parts["recorder"].started == 1  # only the one real press
