"""Plain-language checks: `python -m jarvis music-check` and `python -m jarvis speed-check`."""

from __future__ import annotations

import asyncio
import sys
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


CACHED_BELOW_S = 0.45  # reading the prompt faster than this means Ollama reused its stored copy


async def _cache_test(client: OllamaClient, brain, tools: list) -> None:
    """Does Ollama reuse the stored start of the prompt? Judged by TIME: its token count includes reused tokens."""
    print("\n5) Prompt-cache test (short requests with the same instructions; time is what matters)")
    system = brain._system_text()
    results: dict[str, float] = {}

    async def ask(label: str, messages: list) -> None:
        await client.chat_with_tools([{"role": "system", "content": system}, *messages], tools)
        stats = client.last_stats
        seconds = stats.prompt_s if stats else 0.0
        results[label] = seconds
        verdict = "reused" if seconds < CACHED_BELOW_S else "RE-READ"
        print(f"   {label:46} {seconds:4.1f}s  {verdict}")

    user = lambda text: {"role": "user", "content": text}  # noqa: E731
    note = "\n\n(It is Friday, 27 November 2026, 11:30.)"
    await ask("new sentence", [user("Reply with just the word OK.")])
    await ask("the same sentence again", [user("Reply with just the word OK.")])
    await ask("same start, last word differs", [user("Reply with just the word YES.")])
    await ask("same start, clock note added", [user("Reply with just the word YES." + note)])
    await ask("same, clock minute changed", [user("Reply with just the word YES." + note.replace("11:30", "11:31"))])
    await ask("completely different sentence", [user("What is two plus two? Answer in one word.")])
    await ask("different sentence, after a past turn", [user("Hi"), {"role": "assistant", "content": "Hello!"}, user("What is three plus three? One word.")])
    await ask("the same past turn + new sentence", [user("Hi"), {"role": "assistant", "content": "Hello!"}, user("What is four plus four? One word.")])

    print("   What this means:")
    new_sentence = ("completely different sentence", "different sentence, after a past turn", "the same past turn + new sentence")
    late = ("same start, last word differs", "same, clock minute changed")
    if all(results[k] < CACHED_BELOW_S for k in new_sentence):
        print("   - The cache works for every case. Replies should be quick; tell me the timing numbers from steps 2 and 3.")
    elif all(results[k] < CACHED_BELOW_S for k in late):
        print("   - Ollama reuses the stored prompt only when a new request begins with the same words as the last one.")
        print("     A new sentence makes it re-read the whole instruction list every time, so a SHORTER instruction list")
        print("     is the real speed-up here. Send me this output and I will cut it down.")
    else:
        print("   - Ollama is not reusing the stored prompt reliably. Update Ollama to the newest version and restart it; if it")
        print("     stays this way set OLLAMA_NUM_PARALLEL=1 (Windows: search 'environment variables'), restart Ollama and run again.")


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

        await _cache_test(client, brain, tools)

        print("\nReading the result:")
        if where and where.startswith("CPU"):
            print("  - The model is running on the CPU, not the GPU. Check your NVIDIA driver and update Ollama.")
        elif where and "/" in where:
            print("  - The model doesn't fully fit in the GPU's memory, so part runs on the CPU. Close other GPU apps (browser tabs, games).")
        else:
            print("  - The model is fully on the GPU, so the GPU hitting 100% while it thinks is normal. It should only last a second or two.")
        stats = client.last_stats
        if stats and stats.prompt_s > 1.0:
            print(f"  - Reading the instructions ({stats.prompt_tokens} tokens) took {stats.prompt_s:.1f} s of that reply. See step 5 for whether that is reused.")
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
    except Exception as exc:
        out += ["", f"Profile could not be read: {exc}"]

    inputs, outputs = list_devices()
    out += ["", "Microphones: " + ("; ".join(d.label for d in inputs) or "none found")]
    out.append("Outputs: " + ("; ".join(d.label for d in outputs) or "none found"))

    log = settings.resolve_log_dir() / "jarvis.log"
    out += ["", f"Last {log_lines} log lines ({log}):", *_tail(log, log_lines)]
    return "\n".join(out)


DEMO_VOICES = ["en_GB-alan-medium", "en_GB-northern_english_male-medium", "en_US-ryan-high", "en_US-lessac-high"]
DEMO_SENTENCE = "Good morning, boss. All systems are running, and your first meeting is at nine."


def run_voice_demo(settings, voices: list[str] | None = None, say=print) -> int:
    """Speak one sentence in several Piper voices so you can pick one. Missing voices are downloaded."""
    from jarvis.voice.models import ensure_piper_voice
    from jarvis.voice.output import SoundDevicePlayer
    from jarvis.voice.tts import PiperSynthesizer

    models_dir = settings.resolve_models_dir()
    player = SoundDevicePlayer(settings.voice.output_device)
    chosen = voices or DEMO_VOICES
    say("Voice demo. Each voice says the same sentence. Note the number you like best.")
    for number, name in enumerate(chosen, 1):
        say(f"\n{number}. {name}")
        try:
            path = ensure_piper_voice(models_dir, name)
            synth = PiperSynthesizer(path, settings.voice.tts.length_scale)
            for clip in synth.synthesize(DEMO_SENTENCE):
                player.play(clip, lambda: False)
        except Exception as exc:
            say(f"   skipped ({exc})")
    say("\nTo use one, set voice.tts.voice in config/settings.yaml to its name.")
    return 0


def run_wake_setup(settings, config_path, say=print) -> int:
    """Download the wake word files (about 4 MB) and switch the wake word on in settings.yaml."""
    from jarvis.voice.wakeword import ensure_wake_models
    from jarvis.yamlpatch import set_value

    try:
        folder = ensure_wake_models(settings.resolve_models_dir(), say=say)
    except Exception as exc:
        say(f"Could not finish: {exc}")
        return 1
    say(f"Wake word files are in {folder}.")
    try:
        set_value(config_path, "voice", "wake_word", True)
        say('Switched on: voice.wake_word is now true in settings.yaml. Restart JARVIS, then say "Hey JARVIS".')
    except Exception as exc:
        say(f"Could not edit settings.yaml ({exc}). Set voice.wake_word to true by hand.")
    say("To tune it first, run: python -m jarvis wake-check")
    return 0


def run_wake_check(settings, seconds: float = 20.0, say=print) -> int:
    """Listen for `seconds` and print what the wake word model hears, so you can set the threshold."""
    import sounddevice as sd

    from jarvis.voice.input import resolve_device
    from jarvis.voice.wakeword import CHUNK, WakeWordDetector, wake_dir, wake_models_present

    models_dir = settings.resolve_models_dir()
    if not wake_models_present(models_dir):
        say("The wake word files are missing. Run: python -m jarvis wake-setup")
        return 1
    detector = WakeWordDetector(wake_dir(models_dir))
    say(f'Listening for {seconds:.0f} seconds. Say "Hey JARVIS" a few times, calmly, from where you normally sit.')
    best, heard = 0.0, 0
    end = time.monotonic() + seconds
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", device=resolve_device(settings.voice.input_device), blocksize=CHUNK) as stream:
        while time.monotonic() < end:
            block, _ = stream.read(CHUNK)
            score = detector.feed(block.reshape(-1))
            best = max(best, score)
            if score >= settings.voice.wake_threshold and detector.chunks_seen > 12:
                heard += 1
                say(f"  heard it (score {score:.2f})")
                detector.reset()
    say(f"Done. Heard it {heard} time(s). Highest score {best:.2f}; threshold is {settings.voice.wake_threshold}.")
    if heard == 0:
        say("Nothing heard: lower voice.wake_threshold (try 0.3) or raise the microphone gain.")
    return 0


def run_gmail_setup(settings, config_path, say=print, authorize=None) -> int:
    """Sign in to Gmail (read-only) in the browser, then switch gmail.enabled on in settings.yaml."""
    from jarvis.tools.gmail import GmailError
    from jarvis.tools.gmail import authorize as real_authorize
    from jarvis.yamlpatch import set_value

    secret_file, token_file = settings.resolve_gmail_paths()
    try:
        (authorize or real_authorize)(secret_file, token_file, say=say)
    except GmailError as exc:
        say(str(exc))
        if not secret_file.exists():
            say(f"Save the Google client JSON as: {secret_file}")
            say("How to get it is in the README, section 'Read-only Gmail'.")
        return 1
    say(f"Signed in. Your sign-in is saved in {token_file} (only on this PC).")
    try:
        set_value(config_path, "gmail", "enabled", True)
        say('Switched on: gmail.enabled is now true. Restart JARVIS and say "check my email".')
    except Exception as exc:
        say(f"Could not edit settings.yaml ({exc}). Set gmail.enabled to true by hand.")
    return 0


def run_screen_check(settings, say=print) -> int:
    """Try the screen reader once and show what it found, so you can see it works on this PC."""
    from jarvis.tools.screen import ScreenError, WindowsScreen, clean_text

    if sys.platform != "win32":
        say("Screen reading only works on Windows.")
        return 1
    backend = WindowsScreen()
    say("Switch to the window you want to test (for example a terminal with an error). Reading it in 5 seconds...")
    time.sleep(5)
    try:
        window = backend.target()
        say(f"Window found: {window.title!r} ({window.process})" if window else "No window found; will use the whole screen.")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory(prefix="jarvis-screen-") as folder:
            image = Path(folder) / "screen.png"
            backend.capture(window, image)
            say(f"Picture taken ({image.stat().st_size // 1024} KB).")
            text = clean_text(backend.ocr(image), settings.screen.max_chars)
    except ScreenError as exc:
        say(f"Failed: {exc}")
        return 1
    say(f"Read {len(text)} characters. The first part:\n{text[:400]}")
    say("Nothing was saved. Now try saying: what's on my screen?")
    return 0


def run_route_report(settings, say=print) -> int:
    """Show how your recent turns were handled and how long each part took."""
    from jarvis.turnlog import build_report, read_turns

    path = settings.resolve_turns_path()
    say(f"Reading {path}\n")
    say(build_report(read_turns(path)))
    return 0


async def _tool_eval(settings: Settings, model: str | None, say=print) -> int:
    from jarvis.evalset import CASES

    chosen = settings.model_copy(deep=True)
    if model:
        chosen.ollama.model = model
    client = OllamaClient.from_settings(chosen.ollama)
    try:
        status = await client.health()
        if not (status.reachable and status.model_available):
            say(f"The model {chosen.ollama.model!r} isn't ready: {status.detail}")
            if model:
                say(f"Download it first with: ollama pull {model}")
            return 1
        profile = load_user_profile(chosen)
        brain = build_brain(chosen, client, profile, store=None)
        executor = brain.executor
        if executor is None:
            say("Tools are switched off in settings, so there is nothing to test.")
            return 1
        schemas = executor.registry.schemas()
        mode = "only the tools each sentence could need" if chosen.brain.tool_subset else "all tools every time"
        say(f"Model {chosen.ollama.model}: testing {len(CASES)} sentences against {len(schemas)} tools, sending {mode} (this takes a few minutes).")
        await brain.warm_up()
        right = skipped = 0
        times: list[float] = []
        misses: list[str] = []
        for sentence, expected in CASES:
            if expected is not None and not executor.has(expected):
                skipped += 1
                continue
            started = time.monotonic()
            _result, calls = await brain.decide(brain._build_messages(sentence), executor, sentence)
            times.append(time.monotonic() - started)
            got = calls[0].name if calls else None
            if got == expected:
                right += 1
            else:
                misses.append(f"  {sentence!r}: wanted {expected or 'just talk'}, got {got or 'just talk'}")
        tested = len(times)
        say(f"\nCorrect: {right} of {tested} ({100 * right / max(tested, 1):.0f}%).  Average time {sum(times) / max(tested, 1):.1f} s.  Skipped (tool not set up): {skipped}.")
        if misses:
            say("Mistakes:")
            for line in misses:
                say(line)
        say("\nTo compare another model: ollama pull NAME, then run: python -m jarvis tool-eval NAME")
        return 0
    finally:
        await client.aclose()


def run_tool_eval(settings, model: str | None = None) -> int:
    return asyncio.run(_tool_eval(settings, model))
