"""Speech-to-text adapters."""

from __future__ import annotations

from ai_interpreter.infrastructure.stt.faster_whisper import (
    FasterWhisperRecognizer,
    WhisperDecodeOptions,
)

__all__ = ["FasterWhisperRecognizer", "WhisperDecodeOptions"]
