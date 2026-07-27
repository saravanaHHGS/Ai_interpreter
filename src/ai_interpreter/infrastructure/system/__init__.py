"""Operating-system probes: hardware inventory and audio endpoint discovery."""

from __future__ import annotations

from ai_interpreter.infrastructure.system.audio_endpoints import (
    AudioEndpoint,
    list_windows_audio_endpoints,
)
from ai_interpreter.infrastructure.system.hardware import HardwareProbe

__all__ = ["AudioEndpoint", "HardwareProbe", "list_windows_audio_endpoints"]
