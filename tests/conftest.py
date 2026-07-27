"""Shared pytest fixtures.

Tests never touch the real project directory or the real user profile. Each
one gets a temporary project root containing a copy of the committed
configuration, so a test can corrupt configuration freely without affecting
the developer's working copy.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_interpreter.domain.entities import GpuInfo, HardwareInfo
from ai_interpreter.infrastructure.paths import ApplicationPaths

# The real project root: tests/conftest.py -> tests -> <root>
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_user_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the per-user directories away from the real user profile.

    Applied automatically to every test. Without it, a test that saves a user
    override would write into the developer's real ``%APPDATA%``, silently
    changing their configuration and making later runs order-dependent.

    Args:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest environment patcher.
    """
    monkeypatch.setenv("AI_INTERPRETER_USER_HOME", str(tmp_path / "user_home"))


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A temporary project root with the real configuration copied in.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Path to the temporary project root.
    """
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    return tmp_path


@pytest.fixture
def paths(project_root: Path) -> Iterator[ApplicationPaths]:
    """Application paths rooted in the temporary project directory.

    Args:
        project_root: Temporary project root.

    Yields:
        Resolved paths with all writable directories created.
    """
    resolved = ApplicationPaths.resolve(project_root)
    resolved.ensure_directories()
    yield resolved


@pytest.fixture
def cpu_only_hardware() -> HardwareInfo:
    """A two-core laptop with no discrete GPU.

    Mirrors the development machine this project targets, so the low tier is
    exercised by default rather than by exception.

    Returns:
        A hardware snapshot describing a low-power CPU-only machine.
    """
    return HardwareInfo(
        os_name="Windows",
        os_version="10.0.26200",
        cpu_name="Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz",
        physical_cores=2,
        logical_cores=4,
        total_ram_gb=15.9,
        available_ram_gb=8.0,
        free_disk_gb=99.0,
        python_version="3.12.10",
        gpus=(),
    )


@pytest.fixture
def multicore_cpu_hardware() -> HardwareInfo:
    """An eight-core desktop with no NVIDIA GPU.

    Returns:
        A hardware snapshot that should select the high CPU tier.
    """
    return HardwareInfo(
        os_name="Windows",
        os_version="10.0.26200",
        cpu_name="AMD Ryzen 7 5800X",
        physical_cores=8,
        logical_cores=16,
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        free_disk_gb=500.0,
        python_version="3.12.10",
        gpus=(),
    )


@pytest.fixture
def cuda_hardware() -> HardwareInfo:
    """A workstation with an RTX 3060.

    Returns:
        A hardware snapshot that should select the CUDA tier.
    """
    return HardwareInfo(
        os_name="Windows",
        os_version="10.0.26200",
        cpu_name="Intel(R) Core(TM) i7-12700K",
        physical_cores=12,
        logical_cores=20,
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        free_disk_gb=800.0,
        python_version="3.12.10",
        gpus=(
            GpuInfo(
                name="NVIDIA GeForce RTX 3060",
                total_memory_mb=12288,
                vendor="nvidia",
                driver_version="551.61",
            ),
        ),
    )


@pytest.fixture
def small_gpu_hardware() -> HardwareInfo:
    """A laptop with an NVIDIA GPU too small to hold the model set.

    Returns:
        A hardware snapshot that should fall back to a CPU tier.
    """
    return HardwareInfo(
        os_name="Windows",
        os_version="10.0.26200",
        cpu_name="Intel(R) Core(TM) i7-9750H",
        physical_cores=6,
        logical_cores=12,
        total_ram_gb=16.0,
        available_ram_gb=9.0,
        free_disk_gb=200.0,
        python_version="3.12.10",
        gpus=(
            GpuInfo(
                name="NVIDIA GeForce GTX 1650",
                total_memory_mb=4096,
                vendor="nvidia",
                driver_version="551.61",
            ),
        ),
    )
