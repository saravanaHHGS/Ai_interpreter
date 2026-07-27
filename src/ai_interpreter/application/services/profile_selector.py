"""Automatic hardware profile selection.

Chooses which model tier to load by inspecting the machine. This is the piece
that lets one codebase run on a two-core laptop and an RTX workstation without
any conditional logic anywhere else: the profile picks a set of YAML values,
and the composition root builds adapters from those values.

Pure logic with no I/O - hardware is passed in, so every branch is testable
with a fabricated :class:`HardwareInfo` and no special machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from ai_interpreter.domain.entities import HardwareInfo
from ai_interpreter.domain.errors import ConfigurationError
from ai_interpreter.infrastructure.config.settings import Profile

__all__ = ["ProfileSelection", "ProfileSelector"]

logger = logging.getLogger(__name__)

# Below this much video memory, a GPU cannot hold a speech model, a
# translation model and a synthesizer at once, so CPU execution is both
# simpler and more predictable.
_MIN_CUDA_MEMORY_GB: Final[float] = 6.0

# Below this many physical cores, pipeline stages must run one at a time
# rather than overlapping.
_MIN_CORES_FOR_HIGH_TIER: Final[int] = 6


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    """The outcome of profile selection.

    Args:
        profile: Concrete profile that will be loaded.
        reason: Human-readable justification, shown by ``--check`` so the
            choice is never mysterious.
        was_automatic: Whether the profile was detected rather than requested.
    """

    profile: Profile
    reason: str
    was_automatic: bool


class ProfileSelector:
    """Maps detected hardware onto a configuration profile.

    Args:
        available_profiles: Profile names present in ``config/profiles``.
            Selection refuses to return a profile with no file behind it,
            turning a missing file into a clear startup error instead of a
            confusing "file not found" later.
    """

    def __init__(self, available_profiles: tuple[str, ...]) -> None:
        self._available = frozenset(available_profiles)

    def select(self, requested: Profile, hardware: HardwareInfo) -> ProfileSelection:
        """Resolve a requested profile against the current machine.

        Args:
            requested: Profile from configuration, possibly
                :attr:`Profile.AUTO`.
            hardware: Detected hardware snapshot.

        Returns:
            The concrete profile to load, with the reason it was chosen.

        Raises:
            ConfigurationError: If the resulting profile has no YAML file.
        """
        if requested is not Profile.AUTO:
            self._require_available(requested)
            return ProfileSelection(
                profile=requested,
                reason=f"explicitly requested in configuration (app.profile={requested.value})",
                was_automatic=False,
            )

        profile, reason = self._detect(hardware)
        self._require_available(profile)
        logger.info("Auto-selected profile %s: %s", profile.value, reason)
        return ProfileSelection(profile=profile, reason=reason, was_automatic=True)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _detect(hardware: HardwareInfo) -> tuple[Profile, str]:
        """Pick a profile from hardware characteristics.

        Args:
            hardware: Detected hardware snapshot.

        Returns:
            The profile and the reason for choosing it.
        """
        if hardware.has_cuda_gpu and hardware.cuda_memory_gb >= _MIN_CUDA_MEMORY_GB:
            gpu_name = next(gpu.name for gpu in hardware.gpus if gpu.vendor == "nvidia")
            reason = f"NVIDIA GPU detected ({gpu_name}, {hardware.cuda_memory_gb:.1f} GB VRAM)"
            return Profile.CUDA, reason

        if hardware.has_cuda_gpu:
            reason = (
                f"NVIDIA GPU present but only {hardware.cuda_memory_gb:.1f} GB VRAM "
                f"(minimum {_MIN_CUDA_MEMORY_GB:.0f} GB), so CPU execution was chosen"
            )
            if hardware.physical_cores >= _MIN_CORES_FOR_HIGH_TIER:
                return Profile.CPU_HIGH, reason
            return Profile.CPU_LOW, reason

        if hardware.physical_cores >= _MIN_CORES_FOR_HIGH_TIER:
            reason = (
                f"no NVIDIA GPU, but {hardware.physical_cores} physical cores "
                "allow overlapping pipeline stages"
            )
            return Profile.CPU_HIGH, reason

        reason = (
            f"no NVIDIA GPU and {hardware.physical_cores} physical cores, "
            "so small quantised models run one at a time"
        )
        return Profile.CPU_LOW, reason

    def _require_available(self, profile: Profile) -> None:
        """Verify that a profile has a YAML file behind it.

        Args:
            profile: Profile to check.

        Raises:
            ConfigurationError: If the profile file is missing.
        """
        if profile.value in self._available:
            return
        available = ", ".join(sorted(self._available)) or "none"
        msg = (
            f"Profile {profile.value!r} has no file in config/profiles. "
            f"Available profiles: {available}"
        )
        raise ConfigurationError(msg)
