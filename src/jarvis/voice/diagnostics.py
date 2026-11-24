"""`python -m jarvis voice-check`: proves each voice part works, step by step."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from jarvis.config import Settings
from jarvis.voice.audio_utils import peak_level, rms_level
from jarvis.voice.factory import build_voice_stack
from jarvis.voice.interfaces import SAMPLE_RATE, VoiceError

RECORD_SECONDS = 4
BAR_WIDTH = 30


def list_devices(sd: Any) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Input and output devices of the default audio system (the short list Windows shows)."""
    default_api = sd.default.hostapi
    inputs: list[tuple[int, str]] = []
    outputs: list[tuple[int, str]] = []
    for index, device in enumerate(sd.query_devices()):
        if device["hostapi"] != default_api:
            continue
        if device["max_input_channels"] > 0:
            inputs.append((index, device["name"]))
        if device["max_output_channels"] > 0:
            outputs.append((index, device["name"]))
    return inputs, outputs


def _bar(level: float) -> str:
    filled = min(BAR_WIDTH, int(level * BAR_WIDTH * 2))  # x2: speech rarely reaches full scale
    return "#" * filled + "." * (BAR_WIDTH - filled)


def run_voice_check(settings: Settings, out: Callable[..., None] = print) -> int:
    """Return 0 if every step passed, 1 otherwise."""
    failures = 0

    def passed(message: str) -> None:
        out(f"  PASS  {message}")

    def failed(message: str) -> None:
        nonlocal failures
        failures += 1
        out(f"  FAIL  {message}")

    # 1. audio devices ------------------------------------------------------------
    out("\n1. Audio devices")
    try:
        import sounddevice as sd

        default_in = sd.query_devices(kind="input")["name"]
        default_out = sd.query_devices(kind="output")["name"]
        inputs, outputs = list_devices(sd)
    except Exception as exc:
        failed(f"No usable audio system: {exc}")
        return 1
    passed(f"Default microphone: {default_in}")
    passed(f"Default speakers:   {default_out}")
    out("  Microphones:  " + "; ".join(f"[{i}] {n}" for i, n in inputs))
    out("  Speakers:     " + "; ".join(f"[{i}] {n}" for i, n in outputs))
    out("  To use another, set voice.input_device / voice.output_device in config/settings.yaml.")

    try:
        stack = build_voice_stack(settings)
    except VoiceError as exc:
        failed(str(exc))
        return 1

    # 2. voice + speakers -----------------------------------------------------------
    out("\n2. Voice and speakers")
    try:
        started = time.monotonic()
        stack.synthesizer.warm_up()
        out(f"  Voice loaded in {time.monotonic() - started:.1f}s. Speaking now...")
        for clip in stack.synthesizer.synthesize(
            "Systems online. If you can hear this, my voice is working."
        ):
            stack.player.play(clip, lambda: False)
        passed("Voice played without errors. (Did you hear it? If not, check the speaker setting.)")
    except VoiceError as exc:
        failed(str(exc))

    # 3. microphone -------------------------------------------------------------------
    out(f"\n3. Microphone: speak for {RECORD_SECONDS} seconds, e.g. 'testing one two three'")
    time.sleep(1.0)
    audio = None
    try:
        stack.recorder.start()
        out("  Recording now...")
        time.sleep(RECORD_SECONDS)
        audio = stack.recorder.stop()
    except VoiceError as exc:
        failed(str(exc))

    if audio is not None:
        for second in range(0, audio.size, SAMPLE_RATE):
            chunk = audio[second : second + SAMPLE_RATE]
            out(f"  {second // SAMPLE_RATE + 1}s  [{_bar(peak_level(chunk))}]")
        peak, rms = peak_level(audio), rms_level(audio)
        if peak < settings.voice.silence_peak:
            failed(f"No sound detected (peak {peak:.3f}). Check mute switch, device choice, Windows privacy settings.")
            audio = None
        elif peak < 0.05:
            passed(f"Sound detected but very quiet (peak {peak:.2f}). Raise the mic level in Windows.")
        elif peak > 0.98:
            passed(f"Sound detected but clipping (peak {peak:.2f}). Lower the mic level or move back.")
        else:
            passed(f"Microphone works (peak {peak:.2f}, average {rms:.3f}).")

    # 4. speech recognition ---------------------------------------------------------------
    out("\n4. Speech recognition")
    if audio is None:
        out("  Skipped: no usable recording.")
    else:
        try:
            started = time.monotonic()
            stack.transcriber.warm_up()
            out(f"  Model loaded in {time.monotonic() - started:.1f}s.")
            started = time.monotonic()
            text = stack.transcriber.transcribe(audio)
            took = time.monotonic() - started
            seconds = audio.size / SAMPLE_RATE
            out(f"  Heard: \"{text}\"" if text else "  Heard nothing.")
            out(f"  Recognition took {took:.1f}s for {seconds:.1f}s of audio.")
            if text:
                passed("Speech recognition works.")
            else:
                failed("Nothing recognised. Speak closer to the microphone and try again.")
        except VoiceError as exc:
            failed(str(exc))

    out("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
    return 0 if failures == 0 else 1
