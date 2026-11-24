"""Entry point.

    python -m jarvis                  start the API server
    python -m jarvis chat             quick text chat in the terminal (no server needed)
    python -m jarvis voice            talk to JARVIS (hold the push-to-talk key)
    python -m jarvis voice-check      test microphone, speakers and speech recognition
    python -m jarvis download-models  fetch the speech models in advance
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn

from jarvis.ai.ollama_client import OllamaClient, OllamaError
from jarvis.api.app import create_app
from jarvis.config import ConfigError, Settings, load_settings
from jarvis.logging_setup import setup_logging
from jarvis.wiring import build_brain, load_user_profile


async def terminal_confirm(prompt: str) -> bool:
    """Ask for approval on the keyboard. The AI cannot press this key for you."""
    answer = await asyncio.to_thread(input, f"\n  CONFIRM: {prompt} Allow? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


async def _chat_loop(settings: Settings) -> int:
    client = OllamaClient.from_settings(settings.ollama)
    brain = build_brain(settings, client)
    try:
        status = await client.health()
        if not status.reachable:
            print(f"Cannot reach Ollama. {status.detail}", file=sys.stderr)
            return 1
        if not status.model_available:
            print(status.detail, file=sys.stderr)
            return 1

        print("JARVIS online. Type 'reset' to clear context, 'exit' to quit.")
        while True:
            try:
                text = await asyncio.to_thread(input, "You: ")
            except EOFError:
                break
            text = text.strip()
            if text.lower() in {"exit", "quit"}:
                break
            if text.lower() == "reset":
                brain.reset()
                print("JARVIS: Context cleared.")
                continue
            if not text:
                continue
            try:
                print(f"JARVIS: {await brain.respond(text, confirm=terminal_confirm)}")
            except OllamaError as exc:
                print(f"JARVIS: I'm afraid I hit a problem. {exc}", file=sys.stderr)
        return 0
    finally:
        await client.aclose()


def _run_voice(settings: Settings, use_enter: bool) -> int:
    """Voice conversation: hold the push-to-talk key, speak, release, hear the answer."""
    from jarvis.orchestrator import State, VoiceAssistant
    from jarvis.voice.factory import build_voice_stack
    from jarvis.voice.interfaces import VoiceError

    profile = load_user_profile(settings)
    try:
        # Teach the recogniser the names of your own apps, folders and websites.
        stack = build_voice_stack(settings, use_enter=use_enter, extra_hints=profile.spoken_names())
    except VoiceError as exc:
        print(f"Voice setup failed: {exc}", file=sys.stderr)
        return 1

    client = OllamaClient.from_settings(settings.ollama)
    brain = build_brain(settings, client, profile)
    loop = asyncio.new_event_loop()  # the voice loop is synchronous; the brain is async

    def respond(text: str) -> str:
        return loop.run_until_complete(brain.respond(text, confirm=terminal_confirm))

    labels = {"user": "You", "assistant": "JARVIS", "system": " ··"}
    state_notes = {
        State.LISTENING: "  [listening...]",
        State.PROCESSING: "  [thinking...]",
        State.SPEAKING: "  [speaking...]",
    }

    def on_message(role: str, text: str) -> None:
        print(f"{labels.get(role, role)}: {text}", flush=True)

    def on_state(state: State) -> None:
        if state in state_notes:
            print(state_notes[state], flush=True)

    assistant = VoiceAssistant(
        respond=respond,
        recorder=stack.recorder,
        transcriber=stack.transcriber,
        synthesizer=stack.synthesizer,
        player=stack.player,
        trigger=stack.trigger,
        on_state=on_state,
        on_message=on_message,
        min_speech_seconds=settings.voice.min_speech_seconds,
        max_record_seconds=settings.voice.max_record_seconds,
        silence_peak=settings.voice.silence_peak,
    )

    try:
        status = loop.run_until_complete(client.health())
        if not (status.reachable and status.model_available):
            print(f"Warning: language model not ready. {status.detail}", file=sys.stderr)

        how = "Press Enter to start and stop talking" if use_enter else (
            f"Hold [{settings.voice.push_to_talk_key}] to talk"
        )
        print(f"JARVIS voice mode. {how}. Press Ctrl+C to quit.")
        assistant.run()
        return 0
    except VoiceError as exc:
        print(f"Voice error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nJARVIS offline.")
        return 0
    finally:
        assistant.stop()
        loop.run_until_complete(client.aclose())
        loop.close()


def _download_models(settings: Settings) -> int:
    """Download the speech models now, so the first conversation is not slow."""
    from jarvis.voice.interfaces import VoiceError
    from jarvis.voice.models import ensure_piper_voice, ensure_whisper_model

    models_dir = settings.resolve_models_dir()
    try:
        ensure_piper_voice(models_dir, settings.voice.tts.voice)
        ensure_whisper_model(models_dir, settings.voice.stt.model)
    except VoiceError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print("Speech models are ready.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="JARVIS personal assistant")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["serve", "chat", "voice", "voice-check", "download-models"],
        default="serve",
        help="serve: API server | chat: text chat | voice: talk to JARVIS | "
        "voice-check: test microphone, speakers and recognition | "
        "download-models: fetch speech models",
    )
    parser.add_argument(
        "--enter",
        action="store_true",
        help="voice mode only: press Enter to start/stop talking instead of the push-to-talk key",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(
        settings.logging.level,
        settings.resolve_log_dir(),
        settings.logging.max_bytes,
        settings.logging.backup_count,
        # Interactive modes keep the terminal clean; details still go to logs/jarvis.log.
        console_level=None if args.command == "serve" else "WARNING",
    )

    try:
        if args.command == "chat":
            return asyncio.run(_chat_loop(settings))
        if args.command == "voice":
            return _run_voice(settings, use_enter=args.enter)
        if args.command == "voice-check":
            from jarvis.voice.diagnostics import run_voice_check

            return run_voice_check(settings)
        if args.command == "download-models":
            return _download_models(settings)
        # log_config=None lets uvicorn's logs flow through our logging setup.
        uvicorn.run(
            create_app(settings),
            host=settings.api.host,
            port=settings.api.port,
            log_config=None,
        )
    except ConfigError as exc:  # e.g. a typo in config/profile.yaml
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
