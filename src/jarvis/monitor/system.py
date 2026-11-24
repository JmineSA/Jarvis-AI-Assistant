"""Live system monitoring for the web interface and the status command.

Everything here is read-only and comes from ``psutil`` (already a dependency)
plus ``nvidia-smi`` when an NVIDIA GPU is present. If a measurement is not
available on this machine it is simply left out, never guessed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

_BYTES_IN_GB = 1024**3
_BYTES_IN_MB = 1024**2


@dataclass(frozen=True)
class DiskUsage:
    """One disk or mount point."""

    mount: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mount": self.mount,
            "total_gb": round(self.total_gb, 1),
            "used_gb": round(self.used_gb, 1),
            "free_gb": round(self.free_gb, 1),
            "percent": round(self.percent, 1),
        }


@dataclass(frozen=True)
class GpuUsage:
    """One NVIDIA GPU."""

    name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float | None = None

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return round(self.memory_used_mb / self.memory_total_mb * 100, 1)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "utilization_percent": round(self.utilization_percent, 1),
            "memory_used_mb": round(self.memory_used_mb, 1),
            "memory_total_mb": round(self.memory_total_mb, 1),
            "memory_percent": self.memory_percent,
            "temperature_c": self.temperature_c,
        }


@dataclass(frozen=True)
class SystemSnapshot:
    """A picture of the machine at one moment."""

    timestamp: float
    cpu_percent: float
    cpu_percent_per_core: list[float]
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    swap_percent: float
    disks: list[DiskUsage] = field(default_factory=list)
    gpu: list[GpuUsage] = field(default_factory=list)
    boot_time: float = 0.0
    uptime_seconds: float = 0.0
    process_count: int = 0
    net_sent_mb: float = 0.0
    net_recv_mb: float = 0.0
    net_sent_mb_per_sec: float = 0.0
    net_recv_mb_per_sec: float = 0.0
    cpu_temp_c: float | None = None

    @property
    def cpu_count(self) -> int:
        return len(self.cpu_percent_per_core) or (psutil.cpu_count() or 1)

    @property
    def uptime_human(self) -> str:
        seconds = int(self.uptime_seconds)
        days, rest = divmod(seconds, 86400)
        hours, rest = divmod(rest, 3600)
        minutes = rest // 60
        if days:
            return f"{days}d {hours}h {minutes}m"
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 1),
            "cpu_percent_per_core": [round(v, 1) for v in self.cpu_percent_per_core],
            "cpu_count": self.cpu_count,
            "memory_total_gb": round(self.memory_total_gb, 2),
            "memory_used_gb": round(self.memory_used_gb, 2),
            "memory_available_gb": round(self.memory_available_gb, 2),
            "memory_percent": round(self.memory_percent, 1),
            "swap_percent": round(self.swap_percent, 1),
            "disks": [disk.as_dict() for disk in self.disks],
            "gpu": [gpu.as_dict() for gpu in self.gpu],
            "boot_time": self.boot_time,
            "uptime_seconds": self.uptime_seconds,
            "uptime_human": self.uptime_human,
            "process_count": self.process_count,
            "net_sent_mb": round(self.net_sent_mb, 1),
            "net_recv_mb": round(self.net_recv_mb, 1),
            "net_sent_mb_per_sec": round(self.net_sent_mb_per_sec, 3),
            "net_recv_mb_per_sec": round(self.net_recv_mb_per_sec, 3),
            "cpu_temp_c": self.cpu_temp_c,
        }


class SystemMonitor:
    """Collects system snapshots.

    The network counters are remembered between calls so the interface can
    show a speed rather than a running total.
    """

    def __init__(self, disk_paths: list[str] | None = None, gpu: bool = True) -> None:
        self._disk_paths = disk_paths
        self._use_gpu = gpu
        self._last_net: tuple[float, float] | None = None
        self._last_net_time: float | None = None
        self._gpu_cache: list[GpuUsage] = []
        self._gpu_checked_at: float = 0.0

    # ----------------------------------------------------------------- disks

    def _disk_targets(self) -> list[str]:
        if self._disk_paths:
            return self._disk_paths
        targets: list[str] = []
        for part in psutil.disk_partitions(all=False):
            # Skip network drives and optical media; they are slow or empty.
            if part.fstype.lower() in {"", "cd9660", "dvd", "nfs", "cifs", "smbfs"}:
                continue
            targets.append(part.mountpoint)
        if not targets:
            home = str(psutil.Path.home())
            targets = [home]
        return targets

    def _disks(self) -> list[DiskUsage]:
        usage: list[DiskUsage] = []
        seen: set[str] = set()
        for path in self._disk_targets():
            if path in seen:
                continue
            seen.add(path)
            try:
                part = psutil.disk_usage(path)
            except (OSError, PermissionError):
                continue
            usage.append(
                DiskUsage(
                    mount=path,
                    total_gb=part.total / _BYTES_IN_GB,
                    used_gb=part.used / _BYTES_IN_GB,
                    free_gb=part.free / _BYTES_IN_GB,
                    percent=part.percent,
                )
            )
        usage.sort(key=lambda disk: disk.total_gb, reverse=True)
        return usage

    # ------------------------------------------------------------------- gpu

    def _gpus(self) -> list[GpuUsage]:
        """Read the NVIDIA GPUs. The result is cached for a few seconds."""
        now = time.monotonic()
        if self._gpu_checked_at and now - self._gpu_checked_at < 5:
            return self._gpu_cache
        self._gpu_checked_at = now
        self._gpu_cache = self._read_gpus()
        return self._gpu_cache

    @staticmethod
    def _read_gpus() -> list[GpuUsage]:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return []
        query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
        try:
            result = subprocess.run(
                [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            logger.debug("nvidia-smi could not be run.")
            return []
        if result.returncode != 0:
            return []

        gpus: list[GpuUsage] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue

            def number(text: str) -> float:
                try:
                    return float(text)
                except ValueError:
                    return 0.0

            temperature: float | None = None
            if len(parts) >= 5 and parts[4] not in ("", "[N/A]"):
                temperature = number(parts[4])
            gpus.append(
                GpuUsage(
                    name=parts[0] or "NVIDIA GPU",
                    utilization_percent=number(parts[1]),
                    memory_used_mb=number(parts[2]),
                    memory_total_mb=number(parts[3]),
                    temperature_c=temperature,
                )
            )
        return gpus

    # ------------------------------------------------------------- temperature

    @staticmethod
    def _cpu_temperature() -> float | None:
        try:
            sensors = psutil.sensors_temperatures()
        except (AttributeError, NotImplementedError, OSError):
            return None
        if not sensors:
            return None
        for name, entries in sensors.items():
            if not entries:
                continue
            if any(key in name.lower() for key in ("core", "cpu", "package", "tctl")):
                return float(entries[0].current)
        for entries in sensors.values():
            if entries:
                return float(entries[0].current)
        return None

    # --------------------------------------------------------------- snapshot

    def snapshot(self) -> SystemSnapshot:
        """Take a reading of the machine."""
        cpu_percent = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        net = psutil.net_io_counters()
        now = time.time()

        sent_mb = net.bytes_sent / _BYTES_IN_MB
        recv_mb = net.bytes_recv / _BYTES_IN_MB
        sent_rate = 0.0
        recv_rate = 0.0
        if self._last_net is not None and self._last_net_time is not None:
            elapsed = max(now - self._last_net_time, 0.001)
            sent_rate = max(0.0, (sent_mb - self._last_net[0]) / elapsed)
            recv_rate = max(0.0, (recv_mb - self._last_net[1]) / elapsed)
        self._last_net = (sent_mb, recv_mb)
        self._last_net_time = now

        boot_time = psutil.boot_time()
        try:
            process_count = len(psutil.pids())
        except Exception:  # pragma: no cover - psutil.pids rarely fails
            process_count = 0

        return SystemSnapshot(
            timestamp=now,
            cpu_percent=cpu_percent,
            cpu_percent_per_core=list(per_core),
            memory_total_gb=memory.total / _BYTES_IN_GB,
            memory_used_gb=memory.used / _BYTES_IN_GB,
            memory_available_gb=memory.available / _BYTES_IN_GB,
            memory_percent=memory.percent,
            swap_percent=swap.percent,
            disks=self._disks(),
            gpu=self._gpus() if self._use_gpu else [],
            boot_time=boot_time,
            uptime_seconds=max(0.0, now - boot_time),
            process_count=process_count,
            net_sent_mb=sent_mb,
            net_recv_mb=recv_mb,
            net_sent_mb_per_sec=sent_rate,
            net_recv_mb_per_sec=recv_rate,
            cpu_temp_c=self._cpu_temperature(),
        )

    def summary(self) -> str:
        """A short, readable line for the terminal."""
        snap = self.snapshot()
        parts = [
            f"CPU {snap.cpu_percent:.0f}%",
            f"RAM {snap.memory_used_gb:.1f}/{snap.memory_total_gb:.1f} GB ({snap.memory_percent:.0f}%)",
            f"up {snap.uptime_human}",
        ]
        if snap.gpu:
            parts.append(f"GPU {snap.gpu[0].utilization_percent:.0f}%")
        return " | ".join(parts)
