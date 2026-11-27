"""Screen intelligence, level 1: look at the window you are using and explain what is in it.

How it works, only when you ask ("what's on my screen?", "explain this error"):
  1. find the window you are working in (JARVIS's own window is skipped),
  2. take a picture of just that window,
  3. read the words in it with Windows' built-in OCR (Tesseract if it is installed and Windows OCR fails),
  4. hand the words to the local model, which explains them.

The picture is deleted straight away and the text is never saved or logged. Nothing leaves your PC.
The text on a screen is treated as material to explain, never as instructions to JARVIS.
The model reads words only: it cannot see the shape of a chart or a photo yet.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from jarvis.tools.registry import ToolArgs, ToolResult

logger = logging.getLogger(__name__)

MAX_SIDE = 2400            # Windows OCR refuses pictures bigger than about 2600 pixels
MIN_TEXT = 12              # fewer letters than this means "no readable text"
SCREEN_TIMEOUT = 40.0      # first OCR start-up in PowerShell can take a few seconds
_SKIP_TITLES = {"program manager", "windows input experience", "microsoft text input application", "settings"}


@dataclass(frozen=True)
class WindowInfo:
    title: str
    process: str = ""
    bounds: tuple[int, int, int, int] | None = None  # left, top, right, bottom on the screen


@dataclass(frozen=True)
class RawWindow:
    """A top-level window as Windows lists it (front to back)."""

    title: str
    process: str
    pid: int
    visible: bool
    minimized: bool
    cloaked: bool
    bounds: tuple[int, int, int, int]


def pick_target(windows: list[RawWindow], own_pid: int, foreground: RawWindow | None = None) -> WindowInfo | None:
    """The window the user means: the one in front, or the next one behind if JARVIS itself is in front."""

    def usable(w: RawWindow) -> bool:
        left, top, right, bottom = w.bounds
        return (
            w.visible and not w.minimized and not w.cloaked and w.pid != own_pid
            and bool(w.title.strip()) and w.title.strip().lower() not in _SKIP_TITLES
            and right - left > 80 and bottom - top > 60
        )

    candidates = ([foreground] if foreground is not None else []) + windows
    for w in candidates:
        if usable(w):
            return WindowInfo(w.title.strip(), w.process, w.bounds)
    return None


class ScreenBackend(Protocol):
    def target(self) -> WindowInfo | None: ...

    def capture(self, window: WindowInfo | None, path: Path) -> None:
        """Save a PNG of the window (or the whole screen when `window` is None) at `path`."""

    def ocr(self, image: Path) -> str: ...


# --------------------------------------------------------------------------- reading the words


def clean_text(text: str, max_chars: int) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\r", "").split("\n")]
    kept: list[str] = []
    for line in lines:
        if line or (kept and kept[-1]):
            kept.append(line)
    joined = "\n".join(kept).strip()
    return joined[:max_chars].rsplit("\n", 1)[0] if len(joined) > max_chars else joined


def build_question(question: str, window: WindowInfo | None, text: str) -> str:
    """The message the model sees: the question, which window, and the words from it fenced as data."""
    safe = re.sub(r"</?screen_text>", "", text)
    where = f"{window.title} ({window.process})" if window and window.process else (window.title if window else "the whole screen")
    return f"{question.strip() or 'What is on my screen?'}\n\nWindow: {where}\n\n<screen_text>\n{safe}\n</screen_text>"


class ScreenArgs(ToolArgs):
    question: str = Field("What is on my screen?", max_length=400)
    whole_screen: bool = False  # true = the entire screen instead of just the window in use


def make_look_at_screen(backend: ScreenBackend, max_chars: int = 5000) -> Callable[[ScreenArgs], ToolResult]:
    def run(args: ScreenArgs) -> ToolResult:
        window = None if args.whole_screen else backend.target()
        with tempfile.TemporaryDirectory(prefix="jarvis-screen-") as folder:
            image = Path(folder) / "screen.png"
            try:
                backend.capture(window, image)
                text = backend.ocr(image)
            except ScreenError as exc:
                return ToolResult.failed(str(exc))
        text = clean_text(text, max_chars)
        title = window.title if window else "the screen"
        if len(re.findall(r"\w", text)) < MIN_TEXT:
            return ToolResult.done(
                f"I looked at {title}, but I couldn't read any text in it. If it is a picture or a chart, "
                "I can't see images yet, only words."
            )
        logger.info("Screen read: %d characters from a window titled %d characters long", len(text), len(title))
        return ToolResult.done(f"Read the text in {title}.", ask=build_question(args.question, window, text))

    return run


class ScreenshotArgs(ToolArgs):
    whole_screen: bool = True


def make_take_screenshot(backend: ScreenBackend, folder: Path, clock: Callable[[], datetime] = datetime.now) -> Callable[[ScreenshotArgs], ToolResult]:
    def run(args: ScreenshotArgs) -> ToolResult:
        window = None if args.whole_screen else backend.target()
        path = folder / f"screenshot-{clock():%Y%m%d-%H%M%S}.png"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            backend.capture(window, path)
        except ScreenError as exc:
            return ToolResult.failed(str(exc))
        return ToolResult.done(f"Screenshot saved as {path.name} in {folder.name}.")

    return run


# --------------------------------------------------------------------------- Windows


class ScreenError(Exception):
    """Capturing or reading the screen failed; the message is safe to say aloud."""


# The OCR script is fixed text. The picture path arrives through an environment variable, never in the script.
_OCR_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, $type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
    $null = $task.Wait(-1)
    $task.Result
}
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($env:JARVIS_OCR_IMAGE)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) { [Console]::Error.WriteLine('NO_OCR_LANGUAGE'); exit 3 }
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
foreach ($line in $result.Lines) { $line.Text }
"""


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


class WindowsScreen:
    """Real capture and OCR. Only works on Windows 10/11; everything is fetched lazily."""

    def __init__(self, run: Callable[..., Any] = subprocess.run, which: Callable[[str], str | None] = shutil.which) -> None:
        self._run = run
        self._which = which

    # -- which window
    def _windows(self) -> tuple[list[RawWindow], RawWindow | None]:
        import ctypes
        from ctypes import wintypes

        import psutil

        user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
        found: list[RawWindow] = []

        def describe(hwnd: int) -> RawWindow:
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process = ""
            with contextlib.suppress(Exception):
                process = psutil.Process(pid.value).name()
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            cloaked = ctypes.c_int(0)
            dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
            return RawWindow(
                buffer.value, process, pid.value, bool(user32.IsWindowVisible(hwnd)), bool(user32.IsIconic(hwnd)),
                bool(cloaked.value), (rect.left, rect.top, rect.right, rect.bottom),
            )

        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def each(hwnd: int, _lparam: int) -> bool:
            found.append(describe(hwnd))
            return True

        user32.EnumWindows(callback_type(each), 0)
        front = user32.GetForegroundWindow()
        return found, (describe(front) if front else None)

    def target(self) -> WindowInfo | None:
        if sys.platform != "win32":
            raise ScreenError("Screen reading only works on Windows.")
        try:
            windows, front = self._windows()
        except Exception as exc:
            logger.exception("Could not list windows")
            raise ScreenError("I couldn't tell which window you are using.") from exc
        return pick_target(windows, os.getpid(), front)

    # -- picture
    def capture(self, window: WindowInfo | None, path: Path) -> None:
        if sys.platform != "win32":
            raise ScreenError("Screen reading only works on Windows.")
        try:
            import ctypes

            from PIL import ImageGrab

            with contextlib.suppress(Exception):
                ctypes.windll.user32.SetProcessDPIAware()  # so window sizes and picture pixels agree
            image = ImageGrab.grab(bbox=window.bounds if window and window.bounds else None, all_screens=True)
            image.convert("RGB").save(path, "PNG")
        except ImportError as exc:
            raise ScreenError("Screen reading needs the Pillow package: pip install pillow") from exc
        except Exception as exc:
            logger.exception("Screen capture failed")
            raise ScreenError("I couldn't take a picture of the screen.") from exc

    # -- words
    def ocr(self, image: Path) -> str:
        self._shrink(image)
        errors: list[str] = []
        try:
            return self._windows_ocr(image)
        except ScreenError as exc:
            errors.append(str(exc))
        tesseract = self._which("tesseract")
        if tesseract:
            try:
                done = self._run([tesseract, str(image), "stdout"], capture_output=True, text=True, timeout=30, creationflags=_no_window(), check=False)
                if done.returncode == 0:
                    return str(done.stdout)
            except (OSError, subprocess.SubprocessError):
                logger.exception("Tesseract failed")
        raise ScreenError(errors[0] if errors else "I couldn't read the screen.")

    @staticmethod
    def _shrink(image: Path) -> None:
        try:
            from PIL import Image

            with Image.open(image) as picture:
                if max(picture.size) > MAX_SIDE:
                    scale = MAX_SIDE / max(picture.size)
                    picture.resize((int(picture.width * scale), int(picture.height * scale)), Image.LANCZOS).save(image, "PNG")
        except Exception:
            logger.exception("Could not resize the screenshot")

    def _windows_ocr(self, image: Path) -> str:
        env = {**os.environ, "JARVIS_OCR_IMAGE": str(image)}
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _OCR_SCRIPT]
        try:
            done = self._run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, env=env, creationflags=_no_window(), check=False)
        except subprocess.TimeoutExpired as exc:
            raise ScreenError("Reading the screen took too long.") from exc
        except OSError as exc:
            raise ScreenError("I couldn't start Windows text recognition.") from exc
        if done.returncode == 3 or "NO_OCR_LANGUAGE" in (done.stderr or ""):
            raise ScreenError("Windows has no text-recognition language installed. Add one in Settings, Time and language, Language.")
        if done.returncode != 0:
            logger.warning("Windows OCR failed: %s", (done.stderr or "").strip()[:300])
            raise ScreenError("Windows text recognition failed on that screen.")
        return str(done.stdout)
