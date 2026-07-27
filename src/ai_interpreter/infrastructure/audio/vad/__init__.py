"""Voice activity detection adapters."""

from __future__ import annotations

from ai_interpreter.infrastructure.audio.vad.energy import EnergyVad
from ai_interpreter.infrastructure.audio.vad.silero import SileroVad

__all__ = ["EnergyVad", "SileroVad"]
