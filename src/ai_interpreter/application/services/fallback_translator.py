"""A translator that tries one engine and falls back to another.

Built for the online mode: the primary is the cloud LLM (better
understanding, needs a network), the fallback is the local IndicTrans2
(always available). The contract that matters in a live meeting: **a
network problem may degrade quality, but it must never silence the
interpreter.** Every primary failure is logged, counted, and answered by
the fallback within the same call.
"""

from __future__ import annotations

import logging

from ai_interpreter.domain.entities import Translation
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.domain.ports import Translator
from ai_interpreter.domain.value_objects import LanguagePair

__all__ = ["FallbackTranslator"]

logger = logging.getLogger(__name__)


class FallbackTranslator:
    """Primary-with-fallback composition, satisfying the ``Translator`` port.

    Args:
        primary: Preferred engine (in practice the online LLM).
        fallback: Engine used when the primary fails (the local model).
    """

    def __init__(self, primary: Translator, fallback: Translator) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fallbacks = 0

    @property
    def model_id(self) -> str:
        """Both engines, primary first."""
        return f"{self._primary.model_id} (fallback: {self._fallback.model_id})"

    @property
    def fallbacks(self) -> int:
        """Times the fallback answered because the primary failed."""
        return self._fallbacks

    def supports(self, pair: LanguagePair) -> bool:
        """Whether at least the fallback can serve a direction.

        Args:
            pair: Direction to check.

        Returns:
            ``True`` when either engine supports it - the fallback alone is
            enough to keep the pipeline alive.
        """
        return self._primary.supports(pair) or self._fallback.supports(pair)

    def warmup(self) -> None:
        """Warm both engines; the fallback must be hot BEFORE it is needed."""
        self._fallback.warmup()
        self._primary.warmup()

    def translate(self, text: str, pair: LanguagePair) -> Translation:
        """Translate through the primary, falling back on any failure.

        Args:
            text: Source text.
            pair: Direction to translate.

        Returns:
            The primary's translation, or the fallback's when the primary
            fails or does not serve the direction.
        """
        if self._primary.supports(pair):
            try:
                return self._primary.translate(text, pair)
            except InterpreterError as exc:
                self._fallbacks += 1
                logger.warning(
                    "Online translation failed (%s); using the local engine: %s",
                    self._primary.model_id,
                    exc,
                )
        return self._fallback.translate(text, pair)

    def close(self) -> None:
        """Release both engines."""
        self._primary.close()
        self._fallback.close()
