"""Audio capture, processing and device management."""

from __future__ import annotations

from ai_interpreter.infrastructure.audio.buffers import AudioBlockBuffer, FrameAssembler
from ai_interpreter.infrastructure.audio.devices import SounddeviceDeviceEnumerator
from ai_interpreter.infrastructure.audio.dsp import (
    AudioPreprocessor,
    HighPassFilter,
    StreamingResampler,
    apply_gain,
)
from ai_interpreter.infrastructure.audio.recording import WavRecorder

__all__ = [
    "AudioBlockBuffer",
    "AudioPreprocessor",
    "FrameAssembler",
    "HighPassFilter",
    "SounddeviceDeviceEnumerator",
    "StreamingResampler",
    "WavRecorder",
    "apply_gain",
]
