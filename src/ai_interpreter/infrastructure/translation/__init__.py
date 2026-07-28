"""Machine translation adapters."""

from __future__ import annotations

from ai_interpreter.infrastructure.translation.cache import LruTranslationCache
from ai_interpreter.infrastructure.translation.indictrans2 import IndicTrans2Translator
from ai_interpreter.infrastructure.translation.transliteration import (
    from_devanagari,
    to_devanagari,
)

__all__ = [
    "IndicTrans2Translator",
    "LruTranslationCache",
    "from_devanagari",
    "to_devanagari",
]
