"""Text-to-speech adapters."""

from __future__ import annotations

from ai_interpreter.infrastructure.tts.sherpa_vits import (
    SherpaVitsSynthesizer,
    split_sentences,
)

__all__ = ["SherpaVitsSynthesizer", "split_sentences"]
