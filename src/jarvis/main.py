"""Entry point.

    python -m jarvis                  start the API server
    python -m jarvis chat             quick text chat in the terminal (no server needed)
    python -m jarvis voice            talk to JARVIS (hold the push-to-talk key)
    python -m jarvis voice-check      test microphone, speakers and speech recognition
    python -m jarvis download-models  fetch the speech models in advance
    python -m jarvis ui               open the web interface in your browser
    python -m jarvis status           check every part of JARVIS and the machine
    python -m jarvis memory           search what JARVIS remembers
    python -m jarvis remember "..."   store a fact without asking the model
    python -m jarvis forget <id>      delete one stored memory
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import uvicorn

from jarvis.ai.ollama_client import OllamaClient, OllamaError
from jarvis.api.app import create_app
from jarvis.config import ConfigError, Settings, load_settings
from jarvis.logging_setup import setup_logging
from jarvis.wiring import build_brain, build_memory, load_user_profile


async def terminal_confirm(prompt: str) -> bool:
    """Ask for approval on the keyboard. The AI cannot press this key for you."""
    answer = await asyncio.to_thread(input, f"\n  CONFIRM: {prompt} Allow? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


async def _chat_loop(settings: Settings) -> int:
    client = OllamaClient.from_settings(settings.ollama)
    memory = build_memory(settings)
    brain = build_brain(settings, client, memory=memory)
    try:
        status = await client.health()
        if not status.reachable:
            print(f"Cannot reach Ollama. {status.detail}", file=sys.stderr)
            return 1
        if not status.model_available:
            print(status.detail, file=sys.stderr)
            return 1

        print("JARVIS online. Type 'reset' to clear context, 'exit' to quit.")
        if memory is not None:
            print("Type 'remember <fact>' to store something, 'recall <words>' to search.")
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

            # Terminal shortcuts for the memory, without waking the model.
            if memory is not None and text.lower().startswith("remember "):
                fact = text[len("remember "):].strip()
                if fact:
                    _, created = memory.remember_fact(fact, source="user")
                    print(f"JARVIS: {'Noted.' if created else 'Updated what I had noted.'}")
                    continue
            if memory is not None and text.lower().startswith("recall "):
                query = text[len("recall "):].strip()
                hits = memory.search(query, limit=5)
                if not hits:
                    print("JARVIS: I don't have anything stored about that.")
                else:
                    for hit in hits:
                        print(f"  - {hit.content}")
                continue

            try:
                print(f"JARVIS: {await brain.respond(text, confirm=terminal_confirm)}")
            except OllamaError as exc:
                print(f"JARVIS: I'm afraid I hit a problem. {exc}", file=sys.stderr)
        return 0
    finally:
        await client.aclose()
        if memory is not None:
            memory.close()


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
    memory = build_memory(settings)
    brain = build_brain(settings, client, profile, memory=memory)
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
        if memory is not None:
            memory.close()


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


def _run_ui(settings: Settings) -> int:
    """Start the web interface (Streamlit) and open it in the browser."""
    app = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    if not app.exists():  # pragma: no cover - the file ships with the package
        print(f"The web interface is missing: {app}", file=sys.stderr)
        return 1

    project_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root / 'src'}{';' if existing else ''}{existing}"

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(settings.ui.port),
        "--server.address",
        settings.ui.host,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"Opening the web interface on http://{settings.ui.host}:{settings.ui.port}")
    print("Press Ctrl+C to stop it.")
    try:
        return subprocess.call(command, env=env)
    except FileNotFoundError:
        print("Streamlit is not installed. Run: pip install streamlit", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


async def _run_status(settings: Settings) -> int:
    """Print the health of every part of JARVIS and the state of the machine."""
    from jarvis.monitor.health import collect_health
    from jarvis.monitor.system import SystemMonitor

    client = OllamaClient.from_settings(settings.ollama)
    memory = build_memory(settings)
    brain = build_brain(settings, client, memory=memory)
    try:
        report = await collect_health(
            settings, client, brain=brain, memory=memory, monitor=SystemMonitor()
        )
    finally:
        await client.aclose()
        if memory is not None:
            memory.close()

    print(f"JARVIS status: {report.status.upper()}")
    print()
    print(report.summary())
    print()
    snapshot = report.snapshot
    if snapshot is not None:
        print(f"CPU        {snapshot.cpu_percent:.0f}% over {snapshot.cpu_count} cores")
        print(
            f"Memory     {snapshot.memory_used_gb:.1f} / {snapshot.memory_total_gb:.1f} GB "
            f"({snapshot.memory_percent:.0f}%)"
        )
        for disk in snapshot.disks[:3]:
            print(
                f"Disk       {disk.mount} {disk.percent:.0f}% "
                f"({disk.used_gb:.1f} / {disk.total_gb:.1f} GB)"
            )
        for gpu in snapshot.gpu:
            print(
                f"GPU        {gpu.name} {gpu.utilization_percent:.0f}% "
                f"({gpu.memory_used_mb:.0f} / {gpu.memory_total_mb:.0f} MB)"
            )
        print(f"Uptime     {snapshot.uptime_human}")
    print()
    print(f"Model      {settings.ollama.model}")
    print(f"API        http://{settings.api.host}:{settings.api.port}")
    print(f"Interface  http://{settings.ui.host}:{settings.ui.port}")
    return 0 if report.ok else 1


def _run_memory(settings: Settings, query: str, limit: int) -> int:
    """Search the memory and print what was found."""
    memory = build_memory(settings)
    if memory is None:
        print("Memory is switched off.", file=sys.stderr)
        return 1
    try:
        hits = memory.search(query, limit=limit) if query else memory.recent_messages(limit=limit)
        stats = memory.stats()
        print(f"{stats.messages} messages, {stats.facts} facts, {stats.size_bytes} bytes.")
        print()
        if not hits:
            print("Nothing found.")
            return 0
        for hit in hits:
            print(f"#{hit.id} [{hit.kind}] {hit.date_label}  score {hit.score:.2f}")
            print(f"    {hit.content}")
        return 0
    finally:
        memory.close()


def _run_remember(settings: Settings, fact: str, key: str | None) -> int:
    """Store a fact without involving the model."""
    memory = build_memory(settings)
    if memory is None:
        print("Memory is switched off.", file=sys.stderr)
        return 1
    try:
        memory_id, created = memory.remember_fact(fact, key, source="user")
        verb = "Remembered" if created else "Updated"
        print(f"{verb} #{memory_id}.")
        return 0
    finally:
        memory.close()


def _run_forget(settings: Settings, memory_id: int, wipe_all: bool = False) -> int:
    """Delete one stored memory, or every memory."""
    memory = build_memory(settings)
    if memory is None:
        print("Memory is switched off.", file=sys.stderr)
        return 1
    try:
        if wipe_all:
            deleted = memory.clear_all()
            print(f"Deleted {deleted} memories. JARVIS will start afresh.")
            return 0
        hit = memory.get(memory_id)
        if hit is None:
            print(f"No memory with id {memory_id}.", file=sys.stderr)
            return 1
        if memory.forget(memory_id):
            print(f"Deleted #{memory_id}: {hit.content}")
            return 0
        print("Could not delete that memory.", file=sys.stderr)
        return 1
    finally:
        memory.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="JARVIS personal assistant")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "serve",
            "chat",
            "voice",
            "voice-check",
            "download-models",
            "ui",
            "status",
            "memory",
            "remember",
            "forget",
        ],
        default="serve",
        help="serve: API server | chat: text chat | voice: talk to JARVIS | "
        "voice-check: test microphone, speakers and recognition | "
        "download-models: fetch speech models | ui: web interface | "
        "status: check JARVIS and the machine | memory: search the memory | "
        "remember: store a fact | forget: delete a memory",
    )
    parser.add_argument(
        "--enter",
        action="store_true",
        help="voice mode only: press Enter to start/stop talking instead of the push-to-talk key",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="memory mode: what to search for | remember mode: the fact to store | "
        "forget mode: the id to delete",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="remember mode only: a short name for the fact, e.g. user.dog.name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="memory mode only: how many memories to show (default 20)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="forget mode only: delete every memory, not just one id",
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
        if args.command == "ui":
            return _run_ui(settings)
        if args.command == "status":
            return asyncio.run(_run_status(settings))
        if args.command == "memory":
            return _run_memory(settings, " ".join(args.text).strip(), args.limit)
        if args.command == "remember":
            fact = " ".join(args.text).strip()
            if not fact:
                print("Tell me what to remember.", file=sys.stderr)
                return 2
            return _run_remember(settings, fact, args.key)
        if args.command == "forget":
            if args.all:
                return _run_forget(settings, 0, wipe_all=True)
            if not args.text:
                print("Give me the id of the memory to delete.", file=sys.stderr)
                return 2
            try:
                memory_id = int(args.text[0])
            except ValueError:
                print("The id must be a number.", file=sys.stderr)
                return 2
            return _run_forget(settings, memory_id)
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
