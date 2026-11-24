"""Tests for the system monitor and the health report (Phase 5)."""

import pytest

from jarvis.config import Settings
from jarvis.monitor.health import collect_health
from jarvis.monitor.system import SystemMonitor


def test_snapshot_has_every_measurement():
    snapshot = SystemMonitor(gpu=False).snapshot()
    assert snapshot.cpu_percent >= 0
    assert snapshot.memory_total_gb > 0
    assert snapshot.memory_percent >= 0
    assert snapshot.disks, "at least one disk should be reported"
    assert snapshot.uptime_seconds > 0
    assert snapshot.process_count > 0


def test_snapshot_can_be_serialised():
    data = SystemMonitor(gpu=False).snapshot().as_dict()
    assert data["memory_total_gb"] > 0
    assert isinstance(data["disks"], list)
    assert "uptime_human" in data


def test_the_summary_reads_like_a_sentence():
    summary = SystemMonitor(gpu=False).summary()
    assert "CPU" in summary
    assert "RAM" in summary


def test_uptime_is_written_in_words():
    from jarvis.monitor.system import SystemSnapshot

    snapshot = SystemSnapshot(
        timestamp=0.0,
        cpu_percent=0.0,
        cpu_percent_per_core=[1.0],
        memory_total_gb=1.0,
        memory_used_gb=1.0,
        memory_available_gb=0.0,
        memory_percent=50.0,
        swap_percent=0.0,
        uptime_seconds=90061,
    )
    assert snapshot.uptime_human == "1d 1h 1m"


async def test_health_reports_the_model(settings, fake_client_cls):
    from jarvis.ai.ollama_client import HealthStatus

    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    report = await collect_health(settings, client)
    names = [check.name for check in report.checks]
    assert "Language model" in names
    model_check = next(check for check in report.checks if check.name == "Language model")
    assert model_check.status == "ok"


async def test_health_reports_a_missing_model(settings, fake_client_cls):
    from jarvis.ai.ollama_client import HealthStatus

    client = fake_client_cls(
        health_status=HealthStatus(True, "m", False, "Model 'm' is not installed.")
    )
    report = await collect_health(settings, client)
    model_check = next(check for check in report.checks if check.name == "Language model")
    assert model_check.status == "warn"


async def test_health_reports_an_unreachable_model(settings, fake_client_cls):
    from jarvis.ai.ollama_client import HealthStatus

    client = fake_client_cls(health_status=HealthStatus(False, "m", False, "unreachable"))
    report = await collect_health(settings, client)
    assert report.ok is False
    assert report.status == "error"


async def test_health_includes_the_machine_reading(settings, fake_client_cls):
    from jarvis.ai.ollama_client import HealthStatus

    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    report = await collect_health(settings, client)
    assert report.snapshot is not None
    assert report.snapshot.memory_total_gb > 0


async def test_health_serialises_to_a_dict(settings, fake_client_cls):
    from jarvis.ai.ollama_client import HealthStatus

    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    report = await collect_health(settings, client)
    data = report.as_dict()
    assert data["status"] in {"ok", "warn", "error", "unknown"}
    assert isinstance(data["checks"], list)
    assert data["snapshot"] is not None


async def test_the_summary_lists_every_check(settings, fake_client_cls):
    from jarvis.ai.ollama_client import HealthStatus

    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    report = await collect_health(settings, client)
    summary = report.summary()
    for check in report.checks:
        assert check.name in summary


async def test_memory_is_checked_when_it_is_attached(settings, fake_client_cls, tmp_path):
    from jarvis.ai.ollama_client import HealthStatus
    from jarvis.memory.db import MemoryDatabase
    from jarvis.memory.store import MemoryStore

    client = fake_client_cls(health_status=HealthStatus(True, "m", True))
    database = MemoryDatabase(tmp_path / "memory.db")
    store = MemoryStore(database)
    try:
        report = await collect_health(settings, client, memory=store)
        memory_check = next(check for check in report.checks if check.name == "Memory")
        assert memory_check.status == "ok"
        assert report.extras["memory"]["messages"] == 0
    finally:
        database.close()


def test_memory_settings_have_sane_defaults():
    memory = Settings().memory
    assert memory.enabled is True
    assert memory.max_results >= 1
    assert memory.max_context_chars > 0
    assert memory.retention_days == 0


def test_the_memory_path_is_project_relative(tmp_path):
    settings = Settings()
    assert settings.resolve_memory_path().is_absolute()
    assert settings.resolve_memory_path().name.endswith(".db")


def test_the_memory_database_can_be_overridden(tmp_path):
    from jarvis.config import load_settings

    path = tmp_path / "other.db"
    settings = load_settings(env={"JARVIS_MEMORY_DB": str(path)})
    assert settings.resolve_memory_path() == path
