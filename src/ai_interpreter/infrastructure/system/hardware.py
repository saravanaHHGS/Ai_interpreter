"""Hardware inventory.

Detection runs once at startup. Its output drives automatic profile selection
and is printed by ``--check``, so any bug report states exactly what the
application ran on.

NVIDIA detection deliberately shells out to ``nvidia-smi`` rather than
importing ``torch.cuda``. At this stage torch may not be installed at all, and
even when it is, importing it costs seconds and hundreds of megabytes - far
too expensive for a question asked before the configuration is even loaded.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import psutil

from ai_interpreter.domain.entities import GpuInfo, HardwareInfo

__all__ = ["HardwareProbe"]

logger = logging.getLogger(__name__)

_NVIDIA_SMI_TIMEOUT_SECONDS: Final[float] = 5.0
_BYTES_PER_GB: Final[float] = 1024.0**3


class HardwareProbe:
    """Collects a snapshot of the machine's capabilities."""

    def detect(self, reference_path: Path | None = None) -> HardwareInfo:
        """Inspect the machine.

        Args:
            reference_path: Path whose drive is measured for free space, or
                ``None`` to use the current working directory.

        Returns:
            A hardware snapshot. Individual probes degrade to safe defaults
            rather than raising: an unusable free-space figure must never stop
            the application from starting.
        """
        memory = psutil.virtual_memory()
        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or physical

        return HardwareInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            cpu_name=self._cpu_name(),
            physical_cores=physical,
            logical_cores=logical,
            total_ram_gb=round(memory.total / _BYTES_PER_GB, 2),
            available_ram_gb=round(memory.available / _BYTES_PER_GB, 2),
            free_disk_gb=self._free_disk_gb(reference_path),
            python_version=platform.python_version(),
            gpus=self._detect_gpus(),
        )

    # -- individual probes -------------------------------------------------
    @staticmethod
    def _cpu_name() -> str:
        """Return a human-readable processor name.

        ``platform.processor()`` on Windows returns a family/model string such
        as ``"Intel64 Family 6 Model 142"``, which tells a user nothing. The
        registry holds the marketing name the user recognises.

        Returns:
            The processor name, falling back to ``platform.processor()``.
        """
        if sys.platform == "win32":
            try:
                import winreg

                key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                    value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            except OSError as exc:
                logger.debug("Could not read processor name from registry: %s", exc)

        return platform.processor() or platform.machine() or "Unknown CPU"

    @staticmethod
    def _free_disk_gb(reference_path: Path | None) -> float:
        """Return free space on the drive holding a path.

        Args:
            reference_path: Path to measure, or ``None`` for the working
                directory.

        Returns:
            Free space in gigabytes, or ``0.0`` if it cannot be determined.
        """
        target = reference_path or Path.cwd()
        try:
            usage = shutil.disk_usage(target)
        except OSError as exc:
            logger.debug("Could not determine free disk space for %s: %s", target, exc)
            return 0.0
        return round(usage.free / _BYTES_PER_GB, 2)

    @staticmethod
    def _detect_gpus() -> tuple[GpuInfo, ...]:
        """Detect NVIDIA GPUs via ``nvidia-smi``.

        Returns:
            Detected NVIDIA GPUs, empty when the tool is absent (which is the
            normal case on a machine with only integrated graphics).
        """
        executable = shutil.which("nvidia-smi")
        if executable is None:
            logger.debug("nvidia-smi not found; assuming no NVIDIA GPU")
            return ()

        command = [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("nvidia-smi failed to run: %s", exc)
            return ()

        if completed.returncode != 0:
            logger.warning(
                "nvidia-smi exited with code %d: %s",
                completed.returncode,
                completed.stderr.strip(),
            )
            return ()

        gpus: list[GpuInfo] = []
        for line in completed.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) < 3:
                continue
            name, memory_raw, driver = fields[0], fields[1], fields[2]
            try:
                memory_mb = int(float(memory_raw))
            except ValueError:
                logger.debug("Unparseable memory value from nvidia-smi: %r", memory_raw)
                continue
            gpus.append(
                GpuInfo(
                    name=name,
                    total_memory_mb=memory_mb,
                    vendor="nvidia",
                    driver_version=driver,
                )
            )

        return tuple(gpus)
