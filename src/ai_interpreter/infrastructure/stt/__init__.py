"""Speech-to-text adapters."""

from __future__ import annotations

from ai_interpreter.infrastructure.stt.faster_whisper import (
    FasterWhisperRecognizer,
    WhisperDecodeOptions,
)
from ai_interpreter.infrastructure.stt.onnx_metadata import ensure_onnx_metadata
from ai_interpreter.infrastructure.stt.sherpa_nemo import (
    SherpaNemoCtcRecognizer,
    SherpaNemoStreamingRecognizer,
)

__all__ = [
    "FasterWhisperRecognizer",
    "SherpaNemoCtcRecognizer",
    "SherpaNemoStreamingRecognizer",
    "WhisperDecodeOptions",
    "ensure_onnx_metadata",
]
