"""Unit tests for hardware detection.

Detection reads the real machine, so assertions check invariants that hold on
any machine rather than specific values. Probes that shell out are exercised
with stubs so both the present and absent cases are covered on every machine.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_interpreter.infrastructure.system import hardware as hardware_module
from ai_interpreter.infrastructure.system.hardware import HardwareProbe

pytestmark = pytest.mark.unit


class _FakeCompletedProcess:
    """Stand-in for :class:`subprocess.CompletedProcess`.

    Args:
        returncode: Exit status to report.
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestDetection:
    """Invariants that hold on any machine."""

    def test_reports_plausible_values(self, tmp_path: Path) -> None:
        info = HardwareProbe().detect(reference_path=tmp_path)

        assert info.physical_cores >= 1
        assert info.logical_cores >= info.physical_cores
        assert info.total_ram_gb > 0
        assert 0 <= info.available_ram_gb <= info.total_ram_gb
        assert info.free_disk_gb >= 0
        assert info.cpu_name
        assert info.python_version.startswith("3.")

    def test_free_disk_survives_a_bad_path(self) -> None:
        # A nonexistent drive must not stop the application from starting.
        assert HardwareProbe()._free_disk_gb(Path("Q:/does/not/exist")) == 0.0


class TestGpuDetection:
    """nvidia-smi parsing, including its absence."""

    def test_returns_empty_when_nvidia_smi_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware_module.shutil, "which", lambda _: None)
        assert HardwareProbe()._detect_gpus() == ()

    def test_parses_a_single_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware_module.shutil, "which", lambda _: "nvidia-smi")
        monkeypatch.setattr(
            hardware_module.subprocess,
            "run",
            lambda *_args, **_kwargs: _FakeCompletedProcess(
                0, "NVIDIA GeForce RTX 3060, 12288, 551.61\n"
            ),
        )

        gpus = HardwareProbe()._detect_gpus()

        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA GeForce RTX 3060"
        assert gpus[0].total_memory_mb == 12288
        assert gpus[0].total_memory_gb == pytest.approx(12.0)
        assert gpus[0].vendor == "nvidia"

    def test_parses_multiple_gpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware_module.shutil, "which", lambda _: "nvidia-smi")
        monkeypatch.setattr(
            hardware_module.subprocess,
            "run",
            lambda *_args, **_kwargs: _FakeCompletedProcess(
                0, "RTX 4090, 24576, 551.61\nRTX 3060, 12288, 551.61\n"
            ),
        )

        assert len(HardwareProbe()._detect_gpus()) == 2

    def test_ignores_unparseable_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware_module.shutil, "which", lambda _: "nvidia-smi")
        monkeypatch.setattr(
            hardware_module.subprocess,
            "run",
            lambda *_args, **_kwargs: _FakeCompletedProcess(
                0, "garbage\nRTX 3060, not-a-number, 551.61\nRTX 3060, 12288, 551.61\n"
            ),
        )

        assert len(HardwareProbe()._detect_gpus()) == 1

    def test_handles_non_zero_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hardware_module.shutil, "which", lambda _: "nvidia-smi")
        monkeypatch.setattr(
            hardware_module.subprocess,
            "run",
            lambda *_args, **_kwargs: _FakeCompletedProcess(9, "", "driver mismatch"),
        )

        assert HardwareProbe()._detect_gpus() == ()

    def test_handles_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_args: object, **_kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

        monkeypatch.setattr(hardware_module.shutil, "which", lambda _: "nvidia-smi")
        monkeypatch.setattr(hardware_module.subprocess, "run", _raise)

        assert HardwareProbe()._detect_gpus() == ()
