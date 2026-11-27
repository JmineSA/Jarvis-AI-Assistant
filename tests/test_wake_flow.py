import numpy as np

from jarvis.orchestrator import VoiceAssistant
from jarvis.voice.audio_utils import trim_leading_silence
from jarvis.voice.interfaces import SAMPLE_RATE
from tests.test_orchestrator import FakePlayer, FakeRecorder, FakeSynth, FakeTranscriber, FakeTrigger, speech


class VoiceTrigger(FakeTrigger):
    """A trigger that was woken by voice and records how often a follow-up listen was requested."""

    def __init__(self, by_voice=True):
        super().__init__(presses=1)
        self.woke_by_voice = by_voice
        self.follow_ups = 0

    def begin_follow_up(self):
        self.follow_ups += 1


class Recorder(FakeRecorder):
    """Returns a different clip for each recording."""

    def __init__(self, clips):
        super().__init__()
        self.clips = list(clips)

    def stop(self):
        return self.clips.pop(0) if self.clips else np.zeros(0, dtype=np.float32)


def assistant(replies, clips=None, by_voice=True, follow_up=True, wake_reply="I'm listening."):
    trigger = VoiceTrigger(by_voice)
    synth = FakeSynth(sentences=1)
    messages = []
    answers = iter(replies)
    voice = VoiceAssistant(
        respond=lambda text: next(answers), recorder=Recorder(clips if clips is not None else [speech()] * 5),
        transcriber=FakeTranscriber("play kanye west power"), synthesizer=synth, player=FakePlayer(), trigger=trigger,
        on_message=lambda role, text: messages.append((role, text)), wake_reply=wake_reply, follow_up=follow_up,
    )
    return voice, trigger, synth, messages


def test_hey_jarvis_says_im_listening_then_answers():
    voice, trigger, synth, _ = assistant(["Playing Power by Kanye West."])
    voice._conversation()
    assert synth.texts == ["I'm listening.", "Playing Power by Kanye West."]
    assert trigger.follow_ups == 0


def test_the_key_does_not_say_im_listening():
    voice, _, synth, _ = assistant(["Done."], by_voice=False)
    voice._conversation()
    assert synth.texts == ["Done."]


def test_silence_after_the_wake_word_just_goes_back_to_sleep():
    voice, _, synth, messages = assistant(["never used"], clips=[np.zeros(SAMPLE_RATE * 2, dtype=np.float32)])
    voice._conversation()
    assert synth.texts == ["I'm listening."]  # nothing more is said
    assert ("system", "Standing by. Say 'Hey JARVIS' when you need me.") in messages


def test_a_question_from_jarvis_is_followed_by_listening_without_the_wake_word():
    voice, trigger, synth, _ = assistant(["Sure. What would you like to play?", "Playing Power."])
    voice._conversation()
    assert trigger.follow_ups == 1
    assert synth.texts == ["I'm listening.", "Sure. What would you like to play?", "Playing Power."]


def test_follow_up_that_hears_nothing_is_silent_and_follow_ups_are_capped():
    voice, trigger, synth, messages = assistant(["What now?", "And now?"], clips=[speech(), np.zeros(SAMPLE_RATE * 2, dtype=np.float32)])
    voice._conversation()
    assert trigger.follow_ups == 1 and synth.texts == ["I'm listening.", "What now?"]
    assert not any("Standing by" in m[1] for m in messages)  # no nagging after a follow-up question

    endless, trigger, synth, _ = assistant(["Really?"] * 10)
    endless._conversation()
    assert trigger.follow_ups == 3


def test_follow_ups_can_be_switched_off_and_never_happen_for_the_key():
    voice, trigger, _, _ = assistant(["Sure?"], follow_up=False)
    voice._conversation()
    assert trigger.follow_ups == 0
    voice, trigger, _, _ = assistant(["Sure?"], by_voice=False)
    voice._conversation()
    assert trigger.follow_ups == 0


def test_leading_quiet_is_trimmed_but_a_little_lead_in_is_kept():
    quiet = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    trimmed = trim_leading_silence(np.concatenate([quiet, speech(1.0)]), SAMPLE_RATE)
    assert 1.2 * SAMPLE_RATE < trimmed.size < 1.5 * SAMPLE_RATE
    assert trim_leading_silence(quiet, SAMPLE_RATE).size == quiet.size  # no speech at all: unchanged


async def test_bare_play_asks_what_and_the_next_words_are_the_title():
    from jarvis.ai.brain import Brain
    from jarvis.ai.router import CommandRouter
    from jarvis.profile import Profile
    from jarvis.tools.executor import ToolExecutor
    from jarvis.tools.registry import Tool, ToolArgs, ToolRegistry, ToolResult

    class Args(ToolArgs):
        name: str = ""

    played = []
    registry = ToolRegistry()
    registry.register(Tool("play_media", "d", {"type": "object", "properties": {}}, Args, lambda a: (played.append(a.name), ToolResult.launched(f"Playing {a.name}."))[1]))

    class Client:
        async def chat(self, messages):
            raise AssertionError("the model should not be needed")

    brain = Brain(Client(), "sys", 4, executor=ToolExecutor(registry), router=CommandRouter(Profile(), {"play_media"}))
    assert await brain.respond("play") == "Sure. What would you like to play?"
    assert "Kanye West" in await brain.respond("Kanye West Power")
    assert played and "kanye west power" in played[-1].lower()
