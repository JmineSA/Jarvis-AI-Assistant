"""Builds the real voice components from settings."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.config import Settings
from jarvis.voice.input import EnterTrigger, HotkeyTrigger, SoundDeviceRecorder
from jarvis.voice.interfaces import Player, Recorder, Synthesizer, Transcriber, Trigger
from jarvis.voice.models import ensure_piper_voice, whisper_dir
from jarvis.voice.output import SoundDevicePlayer
from jarvis.voice.stt import FasterWhisperTranscriber
from jarvis.voice.tts import PiperSynthesizer


@dataclass
class VoiceStack:
    recorder: Recorder
    transcriber: Transcriber
    synthesizer: Synthesizer
    player: Player
    trigger: Trigger


def build_voice_stack(
    settings: Settings, use_enter: bool = False, extra_hints: list[str] | None = None
) -> VoiceStack:
    """Create all voice parts. Downloads the Piper voice first if it is missing.

    `extra_hints` are extra names (your apps and projects) the recogniser should expect.
    """
    voice = settings.voice
    hints = list(dict.fromkeys([*voice.stt.hint_words, *(extra_hints or [])]))
    models_dir = settings.resolve_models_dir()
    voice_path = ensure_piper_voice(models_dir, voice.tts.voice)

    return VoiceStack(
        recorder=SoundDeviceRecorder(voice.input_device, voice.max_record_seconds),
        transcriber=FasterWhisperTranscriber(
            model=voice.stt.model,
            language=voice.stt.language,
            compute_type=voice.stt.compute_type,
            cpu_threads=voice.stt.cpu_threads,
            download_root=whisper_dir(models_dir),
            hint_words=hints,
            name_aliases=voice.stt.name_aliases,
        ),
        synthesizer=PiperSynthesizer(voice_path, voice.tts.length_scale),
        player=SoundDevicePlayer(voice.output_device),
        trigger=EnterTrigger() if use_enter else HotkeyTrigger(voice.push_to_talk_key),
    )
