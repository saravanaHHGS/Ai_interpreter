"""Translation caching as a decorator over the ``Translator`` port.

Application-layer on purpose: it composes two ports (``Translator`` and
``TranslationCacheRepository``) and knows nothing about how either is
implemented. The pipeline holds a ``Translator`` and cannot tell whether a
cache sits behind it - which is exactly how a cross-cutting concern should be
attached under this architecture.
"""

from __future__ import annotations

import logging

from ai_interpreter.domain.entities import Translation, UtteranceId
from ai_interpreter.domain.ports import TranslationCacheRepository, Translator
from ai_interpreter.domain.value_objects import LanguagePair

__all__ = ["CachedTranslator"]

logger = logging.getLogger(__name__)


class CachedTranslator:
    """Serves repeated translations from cache, satisfying ``Translator``.

    Args:
        inner: The translator doing real work on a miss.
        cache: Where results are remembered.
    """

    def __init__(self, inner: Translator, cache: TranslationCacheRepository) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def model_id(self) -> str:
        """Identifier of the underlying model."""
        return self._inner.model_id

    @property
    def cache(self) -> TranslationCacheRepository:
        """The cache in use, exposed for the Performance page."""
        return self._cache

    def supports(self, pair: LanguagePair) -> bool:
        """Whether the underlying translator handles a direction.

        Args:
            pair: Direction to check.

        Returns:
            ``True`` when supported.
        """
        return self._inner.supports(pair)

    def translate(self, text: str, pair: LanguagePair) -> Translation:
        """Translate, consulting the cache first.

        Args:
            text: Source text.
            pair: Direction.

        Returns:
            A cached result marked ``from_cache`` with near-zero latency, or
            the inner translator's result, which is then remembered.
        """
        cached = self._cache.get(text, pair)
        if cached is not None:
            return Translation(
                utterance_id=UtteranceId("mt"),
                source_text=text,
                translated_text=cached,
                pair=pair,
                model_id=self._inner.model_id,
                from_cache=True,
                latency_ms=0.0,
            )

        result = self._inner.translate(text, pair)
        if not result.is_empty:
            self._cache.put(text, pair, result.translated_text)
        return result

    def warmup(self) -> None:
        """Warm the underlying translator."""
        self._inner.warmup()

    def close(self) -> None:
        """Release the underlying translator."""
        self._inner.close()
