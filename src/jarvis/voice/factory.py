"""Builds the real voice components from settings."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from jarvis.config import Settings
from jarvis.voice.input import EnterTrigger, HotkeyTrigger, SoundDeviceRecorder
from jarvis.voice.interfaces import Player, Recorder, Synthesizer, Transcriber, Trigger
from jarvis.voice.models import ensure_piper_voice, whisper_dir
from jarvis.voice.output import SoundDevicePlayer
from jarvis.voice.stt import FasterWhisperTranscriber
from jarvis.voice.tts import PiperSynthesizer
from jarvis.voice.wakeword import WakeWordDetector, WakeWordTrigger, make_chime, wake_dir, wake_models_present


@dataclass
class VoiceStack:
    recorder: Recorder
    transcriber: Transcriber
    synthesizer: Synthesizer
    player: Player
    trigger: Trigger


def make_ack_sound(player: Player, enabled: bool) -> Callable[[], None] | None:
    """A function that plays the soft "I heard you" tick, or None when it is switched off."""
    if not enabled:
        return None
    from jarvis.voice.wakeword import make_ack

    clip = make_ack()
    return lambda: player.play(clip, lambda: False)


def build_voice_stack(
    settings: Settings,
    use_enter: bool = False,
    extra_hints: list[str] | None = None,
    correction_terms: list[str] | None = None,
) -> VoiceStack:
    """Create all voice parts. Downloads the Piper voice first if it is missing.

    `extra_hints` are extra names (your apps and projects) the recogniser should expect.
    """
    voice = settings.voice
    hints = list(dict.fromkeys([*voice.stt.hint_words, *(extra_hints or [])]))
    models_dir = settings.resolve_models_dir()
    voice_path = ensure_piper_voice(models_dir, voice.tts.voice)

    recorder = SoundDeviceRecorder(voice.input_device, voice.max_record_seconds)
    player = SoundDevicePlayer(voice.output_device)
    trigger: Trigger = EnterTrigger() if use_enter else HotkeyTrigger(voice.push_to_talk_key)
    if voice.wake_word and not use_enter:
        if wake_models_present(models_dir):
            try:
                chime_clip = make_chime()
                trigger = WakeWordTrigger(
                    trigger,
                    WakeWordDetector(wake_dir(models_dir)),
                    voice.input_device,
                    recorder.recent_peak,
                    threshold=voice.wake_threshold,
                    silence_seconds=voice.wake_silence_seconds,
                    first_word_seconds=voice.wake_listen_seconds,
                    # JARVIS says "I'm listening" itself when a phrase is set; otherwise a short beep
                    chime=(lambda: player.play(chime_clip, lambda: False)) if voice.wake_chime and not voice.wake_reply.strip() else None,
                )
            except Exception:
                logging.getLogger(__name__).exception("Wake word could not start; using push-to-talk only.")
        else:
            print("Wake word is on but its files are missing: run `python -m jarvis wake-setup` (about 4 MB).")

    return VoiceStack(
        recorder=recorder,
        transcriber=FasterWhisperTranscriber(
            model=voice.stt.model,
            language=voice.stt.language,
            compute_type=voice.stt.compute_type,
            cpu_threads=voice.stt.cpu_threads,
            download_root=whisper_dir(models_dir),
            hint_words=hints,
            name_aliases=voice.stt.name_aliases,
            beam_size=voice.stt.beam_size,
            auto_gain=voice.stt.auto_gain,
            correction_terms=[*voice.stt.hint_words, *(correction_terms or [])] if voice.stt.correct_words else [],
        ),
        synthesizer=PiperSynthesizer(voice_path, voice.tts.length_scale),
        player=player,
        trigger=trigger,
    )
