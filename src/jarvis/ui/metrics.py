"""Live CPU, RAM, disk and graphics-card readings for the system monitor."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

GPU_QUERY = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]


@dataclass(frozen=True)
class GpuReading:
    percent: float
    memory_used_mb: float
    memory_total_mb: float


@dataclass(frozen=True)
class Metrics:
    cpu: float
    ram: float
    ram_used_gb: float
    ram_total_gb: float
    disk: float
    gpu: GpuReading | None


def parse_nvidia_smi(text: str) -> GpuReading | None:
    """Parse "31, 1024, 4096" (the first graphics card). None if it isn't that shape."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        percent, used, total = (float(p) for p in parts)
    except ValueError:
        return None
    return GpuReading(percent, used, total)


def read_gpu(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> GpuReading | None:
    """Ask the NVIDIA driver (a fixed read-only command). None if there is no NVIDIA card."""
    extra: dict = {}
    if os.name == "nt":
        extra["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        done = run(GPU_QUERY, capture_output=True, text=True, timeout=2, check=False, **extra)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return parse_nvidia_smi(done.stdout)


def read_metrics(
    gpu: Callable[[], GpuReading | None] = read_gpu,
    disk_path: Path | None = None,
) -> Metrics:
    memory = psutil.virtual_memory()
    anchor = str((disk_path or Path.cwd()).anchor or "/")
    try:
        disk = psutil.disk_usage(anchor).percent
    except OSError:
        disk = 0.0
    gib = 1024**3
    return Metrics(
        cpu=psutil.cpu_percent(interval=None),
        ram=memory.percent,
        ram_used_gb=memory.used / gib,
        ram_total_gb=memory.total / gib,
        disk=disk,
        gpu=gpu(),
    )
