"""Plain-language checks: `python -m jarvis music-check` and `python -m jarvis speed-check`."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from jarvis.ai.ollama_client import OllamaClient
from jarvis.config import Settings
from jarvis.profile import expand_path
from jarvis.tools.media import (
    PlayMediaArgs,
    describe_folder,
    forget_index,
    make_play_media,
)
from jarvis.wiring import build_brain, load_user_profile


class _DryRunLauncher:
    """Records what would be started, and starts nothing."""

    def __init__(self) -> None:
        self.started: list[list[str]] = []

    def start(self, argv: list[str], cwd: str | None = None) -> None:
        self.started.append(argv)

    def open_url(self, url: str) -> None:
        self.started.append([url])

    def open_path(self, path: str | Path) -> None:
        self.started.append([str(path)])


def run_music_check(settings: Settings, words: list[str]) -> int:
    """Shows what JARVIS can see in your music folder, and what a spoken request would play."""
    profile = load_user_profile(settings)
    if not profile.music.folder:
        print("No music folder is set. Add one under music: in config/profile.yaml.")
        return 1
    root = expand_path(profile.music.folder)
    for line in describe_folder(root):
        print(line)

    player = profile.resolve("apps", profile.music.player)
    print(f"Player: {profile.label('apps', player) if player else profile.music.player + ' (not in the profile)'}")

    if words and root.is_dir():
        forget_index()
        launcher = _DryRunLauncher()
        run = make_play_media(profile, launcher, online=lambda: False)  # type: ignore[arg-type]
        result = run(PlayMediaArgs(name=" ".join(words), source="local"))
        print(f'\nIf you said "play {" ".join(words)}" from your music folder:')
        print(f"  {result.message}")
        for argv in launcher.started:
            print(f"  (would start: {' '.join(argv)})")
    elif not words:
        print('\nTip: add words to test a request, e.g.  python -m jarvis music-check love me lil wayne')
    return 0 if root.is_dir() else 1


async def _speed_check(settings: Settings) -> int:
    client = OllamaClient.from_settings(settings.ollama)
    try:
        status = await client.health()
        if not (status.reachable and status.model_available):
            print(f"The model isn't ready: {status.detail}")
            return 1
        profile = load_user_profile(settings)
        brain = build_brain(settings, client, profile, store=None)
        tools = brain.executor.registry.schemas() if brain.executor else []
        print(f"Model: {settings.ollama.model}   tools offered: {len(tools)}   history kept: {settings.brain.max_history_turns} turns")

        print("\n1) Loading the model and caching the prompt...")
        started = time.monotonic()
        await brain.warm_up()
        print(f"   {time.monotonic() - started:.1f} s")
        where = await client.processor()
        print(f"   Running on: {where or 'unknown'}")

        for number, question in ((2, "Tell me something interesting about space."), (3, "Tell me something interesting about space again.")):
            print(f"\n{number}) A question that needs the model: {question!r}")
            started = time.monotonic()
            reply = await brain.respond(question)
            elapsed = time.monotonic() - started
            print(f"   {elapsed:.1f} s -> {reply[:80]}")
            if client.last_stats:
                print(f"   {client.last_stats.summary()}")

        print("\n4) A command that skips the model: 'what time is it'")
        started = time.monotonic()
        reply = await brain.respond("what time is it")
        print(f"   {(time.monotonic() - started) * 1000:.0f} ms ({brain.last_route}) -> {reply}")

        print("\nReading the result:")
        if where and where.startswith("CPU"):
            print("  - The model is running on the CPU, not the GPU. Check your NVIDIA driver and update Ollama.")
        elif where and "/" in where:
            print("  - The model doesn't fully fit in the GPU's memory, so part runs on the CPU. Close other GPU apps (browser tabs, games).")
        else:
            print("  - The model is fully on the GPU, so the GPU hitting 100% while it thinks is normal. It should only last a second or two.")
        stats = client.last_stats
        if stats and stats.prompt_tokens > 1500:
            print(f"  - The model is re-reading {stats.prompt_tokens} words of instructions each time. If this stays high, tell me.")
        return 0
    finally:
        await client.aclose()


def run_speed_check(settings: Settings) -> int:
    return asyncio.run(_speed_check(settings))


def _tail(path: Path, lines: int = 40) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ["(no log file yet)"]
    return text.splitlines()[-lines:]


def build_diagnostics(settings: Settings, engine: object | None = None, log_lines: int = 40) -> str:
    """A report you can paste into the chat. Your home address is never included, only whether it is set."""
    import platform
    import sys

    from jarvis.audio_devices import list_devices

    out: list[str] = ["JARVIS diagnostics", "=" * 18]
    out.append(f"Python {sys.version.split()[0]} on {platform.platform()}")
    try:
        import PySide6

        out.append(f"PySide6 {PySide6.__version__}")
    except ImportError:
        out.append("PySide6 not installed")

    voice = settings.voice
    out += [
        "",
        f"Model: {settings.ollama.model}  keep_alive={settings.ollama.keep_alive}  num_ctx={settings.ollama.num_ctx}  "
        f"num_predict={settings.ollama.num_predict}",
        f"Brain: history={settings.brain.max_history_turns} turns, fast_commands={settings.brain.fast_commands}",
        f"Voice: whisper {voice.stt.model}, piper {voice.tts.voice}, mic device={voice.input_device}, "
        f"output device={voice.output_device}, key={voice.push_to_talk_key}",
        f"Greeting name: {settings.ui.greeting_name}",
    ]
    processor = getattr(engine, "_processor", "") if engine is not None else ""
    last = getattr(engine, "_last_reply", "") if engine is not None else ""
    if processor or last:
        out.append(f"Model running on: {processor or 'unknown'}   last reply: {last or 'n/a'}")

    try:
        profile = load_user_profile(settings)
        out += [
            "",
            f"Profile: {len(profile.apps)} apps, {len(profile.folders)} folders, {len(profile.websites)} websites",
            f"Home address: {'set' if profile.location and profile.location.address else 'not set'}; "
            f"weather location: {profile.location.name if profile.location else 'none'}",
        ]
        if profile.music.folder:
            root = expand_path(profile.music.folder)
            out.append(f"Music folder: {root} ({'found' if root.is_dir() else 'NOT FOUND'}); online source: {profile.music.online}")
        else:
            out.append("Music folder: not set")
    except Exception as exc:  # noqa: BLE001 - a broken profile is exactly what this report should show
        out += ["", f"Profile could not be read: {exc}"]

    inputs, outputs = list_devices()
    out += ["", "Microphones: " + ("; ".join(d.label for d in inputs) or "none found")]
    out.append("Outputs: " + ("; ".join(d.label for d in outputs) or "none found"))

    log = settings.resolve_log_dir() / "jarvis.log"
    out += ["", f"Last {log_lines} log lines ({log}):", *_tail(log, log_lines)]
    return "\n".join(out)
