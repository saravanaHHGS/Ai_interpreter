"""In-memory LRU translation cache.

Conversational speech is heavily repetitive - greetings, confirmations, "can
you hear me" - and a cache hit costs microseconds against 0.2-1.3 s for a real
translation. The cache is bounded LRU: unbounded growth over an hours-long
session would contradict the low-memory constraint, and least-recently-used
is the right eviction for conversation, where recent phrases recur.

Keys are (normalised text, direction). Normalisation is NFC plus whitespace
collapse and case-folding for the lookup only - "Hello" and "hello" translate
identically, so they should hit the same entry - while the stored translation
is whatever the translator produced for the first occurrence.

Thread safety: a plain lock around every operation. The pipeline calls this
from one worker at a time today, but the Phase 9 design has no obligation to
keep it that way, and a lock this uncontended costs nothing measurable.

Persistence (``translation.cache.persist``) is accepted by the schema but not
implemented here; privacy defaults keep conversation text off disk, and an
in-memory cache dies with the session, which is exactly what that default
promises.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict
from typing import Final

from ai_interpreter.domain.value_objects import LanguagePair

__all__ = ["LruTranslationCache"]

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Build the lookup form of a source text.

    Args:
        text: Raw source text.

    Returns:
        NFC-normalised, whitespace-collapsed, case-folded text.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text).strip()).casefold()


class LruTranslationCache:
    """Bounded LRU cache, satisfying the ``TranslationCacheRepository`` port.

    Args:
        max_entries: Entries retained before the least recently used is
            evicted. Zero disables storage entirely (every ``get`` misses).
    """

    def __init__(self, max_entries: int = 5000) -> None:
        if max_entries < 0:
            msg = f"max_entries cannot be negative, got {max_entries}"
            raise ValueError(msg)
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # -- port interface ----------------------------------------------------
    def get(self, text: str, pair: LanguagePair) -> str | None:
        """Look up a cached translation.

        Args:
            text: Source text.
            pair: Direction.

        Returns:
            The cached translation, or ``None`` on a miss.
        """
        key = (_normalise(text), pair.key)
        with self._lock:
            translation = self._entries.get(key)
            if translation is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return translation

    def put(self, text: str, pair: LanguagePair, translation: str) -> None:
        """Store a translation.

        Args:
            text: Source text.
            pair: Direction.
            translation: Result to cache. Empty results are not stored -
                caching a failure would replay it forever.
        """
        if self._max_entries == 0 or not translation.strip():
            return
        key = (_normalise(text), pair.key)
        with self._lock:
            self._entries[key] = translation
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Remove every cached entry and reset statistics."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache since startup."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total else 0.0

    @property
    def size(self) -> int:
        """Entries currently held."""
        with self._lock:
            return len(self._entries)
