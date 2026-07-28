"""Transcript glossary: recovering code-switched and technical terms.

The problem, seen in a live session: the Tamil recogniser's vocabulary is
Tamil-only, so English words spoken mid-sentence come out as phonetic Tamil -
"world research" became ``வேர்ல்ட் ரெஸ்சர்ச்`` - and product names like "VALD"
or "catapult" fare worse. A true code-switching model does not exist in a
form the target machine can run at speed, so this is the practical fix: a
user-editable mapping from the phonetic forms the recogniser actually
produces back to the intended term, applied between recognition and
translation.

It works because IndicTrans2 passes Latin tokens through code-mixed input
untouched (verified live: ``நாளைக்கு VALD session இருக்கு`` translates to
"There will be a VALD session tomorrow"), so a recovered term survives into
the English output verbatim.

The workflow is deliberately simple: run ``--interpret``, read the ``[ta]``
line when a term comes out wrong, copy the exact wrong form into the glossary
under the intended term, done. Personal glossaries belong in the user config
layer (``%APPDATA%``), where they survive updates.

Matching is deliberately conservative: whole-token (whitespace-delimited)
matches only, longest variant first. A variant never matches inside a longer
word, because silently rewriting fragments of correct Tamil would be worse
than missing a term.
"""

from __future__ import annotations

import logging
import re
import unicodedata

__all__ = ["GlossaryRewriter"]

logger = logging.getLogger(__name__)


class GlossaryRewriter:
    """Replaces known mis-transcribed variants with their intended term.

    Args:
        terms: Mapping of the intended term to the transcript forms that
            should become it, e.g. ``{"world": ["வேர்ல்ட்", "வலட"]}``.
            Variants may be multi-word phrases.
    """

    def __init__(self, terms: dict[str, list[str]]) -> None:
        rules: list[tuple[re.Pattern[str], str]] = []
        for canonical, variants in terms.items():
            for variant in variants:
                cleaned = unicodedata.normalize("NFC", variant.strip())
                if not cleaned:
                    continue
                # Whole-token boundaries: not preceded or followed by
                # non-space. Unicode \b is unreliable across scripts, and a
                # partial-word rewrite of correct Tamil would be worse than a
                # missed term.
                pattern = re.compile(
                    r"(?<!\S)" + re.escape(cleaned) + r"(?!\S)",
                    re.IGNORECASE,
                )
                rules.append((pattern, canonical))
        # Longest variant first, so a phrase wins over a word it contains.
        rules.sort(key=lambda rule: len(rule[0].pattern), reverse=True)
        self._rules = rules
        self._hits = 0

    @property
    def size(self) -> int:
        """Number of variant rules loaded."""
        return len(self._rules)

    @property
    def hits(self) -> int:
        """Total replacements made since creation."""
        return self._hits

    def rewrite(self, text: str) -> str:
        """Apply every rule to a transcript.

        Args:
            text: Recognised text.

        Returns:
            The text with known variants replaced; unchanged when nothing
            matches.
        """
        if not self._rules or not text:
            return text

        result = unicodedata.normalize("NFC", text)
        for pattern, canonical in self._rules:
            result, count = pattern.subn(canonical, result)
            self._hits += count
        return result
