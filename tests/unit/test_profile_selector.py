"""Unit tests for automatic hardware profile selection."""

from __future__ import annotations

import pytest

from ai_interpreter.application.services.profile_selector import ProfileSelector
from ai_interpreter.domain.entities import HardwareInfo
from ai_interpreter.domain.errors import ConfigurationError
from ai_interpreter.infrastructure.config.settings import Profile

pytestmark = pytest.mark.unit

ALL_PROFILES = ("cpu_low", "cpu_high", "cuda")


class TestAutomaticSelection:
    """Hardware to profile mapping."""

    def test_low_power_cpu_selects_low_tier(self, cpu_only_hardware: HardwareInfo) -> None:
        selection = ProfileSelector(ALL_PROFILES).select(Profile.AUTO, cpu_only_hardware)

        assert selection.profile is Profile.CPU_LOW
        assert selection.was_automatic is True
        assert "2 physical cores" in selection.reason

    def test_multicore_cpu_selects_high_tier(self, multicore_cpu_hardware: HardwareInfo) -> None:
        selection = ProfileSelector(ALL_PROFILES).select(Profile.AUTO, multicore_cpu_hardware)

        assert selection.profile is Profile.CPU_HIGH
        assert "no NVIDIA GPU" in selection.reason

    def test_large_gpu_selects_cuda(self, cuda_hardware: HardwareInfo) -> None:
        selection = ProfileSelector(ALL_PROFILES).select(Profile.AUTO, cuda_hardware)

        assert selection.profile is Profile.CUDA
        assert "RTX 3060" in selection.reason

    def test_small_gpu_falls_back_to_cpu(self, small_gpu_hardware: HardwareInfo) -> None:
        selection = ProfileSelector(ALL_PROFILES).select(Profile.AUTO, small_gpu_hardware)

        # 4 GB VRAM cannot hold the model set; 6 physical cores earn the high tier.
        assert selection.profile is Profile.CPU_HIGH
        assert "4.0 GB VRAM" in selection.reason


class TestExplicitSelection:
    """Configuration and command line overrides."""

    def test_explicit_profile_bypasses_detection(self, cpu_only_hardware: HardwareInfo) -> None:
        selection = ProfileSelector(ALL_PROFILES).select(Profile.CUDA, cpu_only_hardware)

        assert selection.profile is Profile.CUDA
        assert selection.was_automatic is False
        assert "explicitly requested" in selection.reason


class TestAvailabilityChecking:
    """Profiles with no file behind them are rejected loudly."""

    def test_missing_profile_file_raises(self, cuda_hardware: HardwareInfo) -> None:
        selector = ProfileSelector(("cpu_low",))
        with pytest.raises(ConfigurationError, match="has no file in config/profiles"):
            selector.select(Profile.AUTO, cuda_hardware)

    def test_error_lists_available_profiles(self, cpu_only_hardware: HardwareInfo) -> None:
        selector = ProfileSelector(("cuda",))
        with pytest.raises(ConfigurationError, match="Available profiles: cuda"):
            selector.select(Profile.AUTO, cpu_only_hardware)
