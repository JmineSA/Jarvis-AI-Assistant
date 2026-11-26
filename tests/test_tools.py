from datetime import datetime
from types import SimpleNamespace

import psutil
import pytest

from jarvis.profile import Profile
from jarvis.tools.apps import (
    OpenAppArgs,
    OpenFolderArgs,
    OpenWebsiteArgs,
    find_executable,
    make_open_app,
    make_open_folder,
    make_open_website,
)
from jarvis.tools.builtin import build_registry
from jarvis.tools.files import MAX_SPOKEN, ListFilesArgs, make_list_files
from jarvis.tools.registry import ToolStatus
from jarvis.tools.system import NoArgs, make_current_time, make_system_status


class FakeLauncher:
    def __init__(self, fail=None):
        self.fail = fail
        self.started, self.paths, self.urls = [], [], []

    def start(self, argv, cwd=None):
        if self.fail:
            raise self.fail
        self.started.append((argv, cwd))

    def open_path(self, path):
        if self.fail:
            raise self.fail
        self.paths.append(path)

    def open_url(self, url):
        if self.fail:
            raise self.fail
        self.urls.append(url)


INSTALLED = {"code": "/bin/code", "jupyter": "/bin/jupyter", "chrome": "/bin/chrome"}
which = INSTALLED.get


def never_exists(path):
    return False


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "lightson"
    root.mkdir()
    (root / "data").mkdir()
    (root / "src").mkdir()
    (root / "notes.md").write_text("x")
    (root / "train.py").write_text("x")
    (root / ".git").mkdir()
    (root / ".env").write_text("secret")
    return root


@pytest.fixture
def profile(tmp_path, project) -> Profile:
    return Profile.model_validate(
        {
            "owner_name": "Lesiba",
            "apps": {
                "vscode": {"label": "VS Code", "names": ["code"], "candidates": ["code"], "folder_mode": "arg"},
                "jupyter": {"label": "Jupyter", "candidates": ["jupyter"], "args": ["lab"], "folder_mode": "cwd"},
                "chrome": {"label": "Chrome", "candidates": ["chrome"]},
                "powerbi": {"label": "Power BI", "candidates": ["PBIDesktop"]},
                "ghost": {"label": "Ghost", "candidates": ["/no/such/ghost.exe"]},
            },
            "folders": {
                "lightson": {"label": "LightsOn", "names": ["solar project"], "path": str(project)},
                "missing": {"label": "Missing", "path": str(tmp_path / "nope")},
            },
            "websites": {
                "databricks": {"label": "Databricks", "url": "", "app": "chrome"},
                "github": {"label": "GitHub", "url": "https://github.com/x", "app": "chrome"},
                "plain": {"label": "Plain", "url": "https://example.com"},
                "bad": {"label": "Bad", "url": "file:///etc/passwd", "app": "chrome"},
            },
        }
    )


# --------------------------------------------------------------------------- find_executable


def test_find_executable_prefers_existing_explicit_path():
    assert find_executable(["/opt/app.exe", "code"], which, lambda p: p == "/opt/app.exe") == "/opt/app.exe"


def test_find_executable_falls_back_to_path_lookup():
    assert find_executable(["/missing/app.exe", "code"], which, never_exists) == "/bin/code"


def test_find_executable_none_found():
    assert find_executable(["/missing/app.exe", "nothing"], which, never_exists) is None


# --------------------------------------------------------------------------- open_app


def run_app(profile, launcher, **args):
    return make_open_app(profile, launcher, which, never_exists)(OpenAppArgs(**args))


def test_open_app_launches_program_by_spoken_name(profile):
    launcher = FakeLauncher()
    result = run_app(profile, launcher, name="Code")
    assert result.status is ToolStatus.LAUNCHED
    assert result.message == "Opening VS Code."
    assert launcher.started == [(["/bin/code"], None)]


def test_open_app_with_folder_argument(profile, project):
    launcher = FakeLauncher()
    result = run_app(profile, launcher, name="vscode", folder="solar project")
    assert result.message == "Opening VS Code in LightsOn."
    assert launcher.started == [(["/bin/code", str(project)], None)]


def test_open_app_starts_inside_folder_when_app_wants_cwd(profile, project):
    launcher = FakeLauncher()
    result = run_app(profile, launcher, name="jupyter", folder="lightson")
    assert result.status is ToolStatus.LAUNCHED
    assert launcher.started == [(["/bin/jupyter", "lab"], project)]


def test_open_app_folder_not_supported_by_app(profile):
    result = run_app(profile, FakeLauncher(), name="chrome", folder="lightson")
    assert result.status is ToolStatus.FAILED
    assert "can't be opened inside a folder" in result.message


def test_open_app_unknown_app_lists_options(profile):
    launcher = FakeLauncher()
    result = run_app(profile, launcher, name="firefox")
    assert result.status is ToolStatus.FAILED
    assert "VS Code" in result.message and "Power BI" in result.message
    assert launcher.started == []


def test_open_app_not_installed_points_to_profile(profile):
    result = run_app(profile, FakeLauncher(), name="powerbi")
    assert result.status is ToolStatus.FAILED
    assert "profile.yaml" in result.message


def test_open_app_unknown_or_missing_folder(profile):
    assert run_app(profile, FakeLauncher(), name="vscode", folder="nowhere").status is ToolStatus.FAILED
    missing = run_app(profile, FakeLauncher(), name="vscode", folder="missing")
    assert missing.status is ToolStatus.FAILED and "doesn't exist" in missing.message


def test_open_app_launcher_error_is_reported(profile):
    result = run_app(profile, FakeLauncher(fail=PermissionError(13, "Access is denied")), name="vscode")
    assert result.status is ToolStatus.FAILED
    assert "Access is denied" in result.message


# --------------------------------------------------------------------------- open_folder


def test_open_folder_happy_path(profile, project):
    launcher = FakeLauncher()
    result = make_open_folder(profile, launcher)(OpenFolderArgs(name="the solar project folder"))
    assert result.message == "Opening the LightsOn folder."
    assert launcher.paths == [project]


def test_open_folder_missing_or_unknown(profile):
    launcher = FakeLauncher()
    run = make_open_folder(profile, launcher)
    assert "doesn't exist" in run(OpenFolderArgs(name="missing")).message
    assert run(OpenFolderArgs(name="windows")).status is ToolStatus.FAILED
    assert launcher.paths == []


def test_open_folder_launcher_error(profile):
    run = make_open_folder(profile, FakeLauncher(fail=OSError("boom")))
    assert run(OpenFolderArgs(name="lightson")).status is ToolStatus.FAILED


# --------------------------------------------------------------------------- open_website


def run_site(profile, launcher, name):
    return make_open_website(profile, launcher, which, never_exists)(OpenWebsiteArgs(name=name))


def test_open_website_uses_the_chosen_app(profile):
    launcher = FakeLauncher()
    result = run_site(profile, launcher, "github")
    assert result.message == "Opening GitHub."
    assert launcher.started == [(["/bin/chrome", "https://github.com/x"], None)]
    assert launcher.urls == []


def test_open_website_falls_back_to_default_browser(profile):
    launcher = FakeLauncher()
    assert run_site(profile, launcher, "plain").status is ToolStatus.LAUNCHED
    assert launcher.urls == ["https://example.com"]


def test_open_website_without_saved_address(profile):
    launcher = FakeLauncher()
    result = run_site(profile, launcher, "databricks")
    assert result.status is ToolStatus.FAILED
    assert "No address is saved" in result.message
    assert launcher.started == [] and launcher.urls == []


def test_open_website_rejects_non_web_addresses(profile):
    launcher = FakeLauncher()
    result = run_site(profile, launcher, "bad")
    assert result.status is ToolStatus.FAILED
    assert launcher.started == [] and launcher.urls == []


# --------------------------------------------------------------------------- list_files


def test_list_files_folders_first_and_hides_dotfiles(profile):
    result = make_list_files(profile)(ListFilesArgs(folder="lightson"))
    assert result.status is ToolStatus.DONE
    assert result.message == "LightsOn has 4 items: data, src, notes.md, train.py."
    assert ".env" not in result.message and ".git" not in result.message


def test_list_files_subfolder(profile, project):
    (project / "data" / "raw.csv").write_text("x")
    result = make_list_files(profile)(ListFilesArgs(folder="lightson", subfolder="data"))
    assert result.message == "LightsOn, in data has 1 item: raw.csv."


def test_list_files_empty_folder(profile, project):
    result = make_list_files(profile)(ListFilesArgs(folder="lightson", subfolder="src"))
    assert result.message.endswith("is empty.")


@pytest.mark.parametrize("sneaky", ["..", "../..", "data/../..", "/etc"])
def test_list_files_cannot_escape_the_folder(profile, sneaky):
    result = make_list_files(profile)(ListFilesArgs(folder="lightson", subfolder=sneaky))
    assert result.status is ToolStatus.FAILED
    assert "outside" in result.message


def test_list_files_missing_subfolder_and_file(profile):
    run = make_list_files(profile)
    assert run(ListFilesArgs(folder="lightson", subfolder="nope")).status is ToolStatus.FAILED
    assert run(ListFilesArgs(folder="lightson", subfolder="notes.md")).status is ToolStatus.FAILED


def test_list_files_unknown_folder_only_offers_listed_folders(profile):
    result = make_list_files(profile)(ListFilesArgs(folder="C drive"))
    assert result.status is ToolStatus.FAILED
    assert "LightsOn" in result.message


def test_list_files_long_listing_is_summarised(profile, project):
    for i in range(20):
        (project / f"file{i:02}.txt").write_text("x")
    result = make_list_files(profile)(ListFilesArgs(folder="lightson"))
    assert "has 24 items" in result.message
    assert result.message.endswith(f", and {24 - MAX_SPOKEN} more.")


# --------------------------------------------------------------------------- system


def test_system_status_message(monkeypatch):
    gb = 1024**3
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 12.4)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(percent=41.0, used=13.1 * gb, total=32 * gb)
    )
    result = make_system_status()(NoArgs())
    assert result.message == "CPU is at 12 percent. Memory is at 41 percent, 13.1 of 32 gigabytes."


def test_current_time_message():
    clock = lambda: datetime(2026, 11, 4, 9, 5)
    assert make_current_time(clock)(NoArgs()).message == "It's 09:05 on Wednesday, 4 November 2026."


# --------------------------------------------------------------------------- registry


def test_registry_contains_expected_tools_and_schemas(profile):
    registry = build_registry(profile, FakeLauncher())
    assert registry.names() == [
        "open_app", "open_folder", "list_files", "open_website", "web_search", "maps", "compose_email",
        "play_media", "media_control", "system_status", "current_time",
    ]
    schemas = {s["function"]["name"]: s["function"] for s in registry.schemas()}
    assert schemas["open_app"]["parameters"]["properties"]["name"]["enum"] == [
        "vscode", "jupyter", "chrome", "powerbi", "ghost",
    ]
    assert schemas["open_app"]["parameters"]["required"] == ["name"]
    assert schemas["current_time"]["parameters"]["properties"] == {}


def test_empty_profile_only_offers_system_tools():
    registry = build_registry(Profile(), FakeLauncher())
    # No apps, folders, music or location: only the tools that need nothing from the profile.
    assert registry.names() == ["web_search", "maps", "compose_email", "play_media", "media_control", "system_status", "current_time"]
