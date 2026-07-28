"""Detecting English rendered as phonetic Tamil.

The Tamil recogniser's vocabulary is Tamil-only, so English spoken
mid-sentence comes out transliterated: a live session produced ``ட்ரை`` for
"try", ``கிரெட்`` for "create", ``ப்ரான்ச்ஸ்`` for "branches" and
``டிஃபெரெண்ட்`` for "different". The words *look* like Tamil but violate
Tamil phonotactics, and that violation is detectable:

* **Native Tamil words never begin with a consonant cluster.** ``ட்ரை``
  opens with ``ட் + ர`` - impossible in native orthography, routine in
  transliterated English (tr-, br-, cr-, st-...).
* **The āytham + p sequence ``ஃப`` writes the foreign /f/** (``டிஃபெரெண்ட்``);
  ``ஃஜ`` writes /z/. Native words use ஃ differently and rarely.
* **Native words end in vowels or the sonorants ண் ந் ம் ன் ய் ர் ல் ள் ழ்**.
  An ending stop like ``ட்`` (``கிரெட்``, ``நேட்``) or the borrowed ``ஸ்``
  (English plural: ``ப்ரான்ச்ஸ்``) marks a transliteration.
* **Grantha letters** (ஜ ஷ ஸ ஹ), imported precisely for foreign sounds, are
  a weaker hint on their own - plenty of assimilated loanwords carry them -
  but combine with the above.

None of this identifies *which* English word was said; it identifies *that*
a token is English-in-Tamil-clothing. That powers two features: routing an
English-heavy utterance to the English recogniser instead of translating
transliteration soup, and suggesting glossary entries at session end.

Deliberate bias: precision over recall. ``பேமிலியர்`` ("familiar") is
phonotactically plausible Tamil and goes undetected - an accepted miss,
because flagging real Tamil words as English would corrupt correct
transcripts, while missing an English word merely leaves today's behaviour.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["english_phonetic_score", "flag_english_tokens"]

# Grantha letters: imported for foreign sounds, weak evidence alone.
_GRANTHA: Final[frozenset[str]] = frozenset("ஜஷஸஹ")

_VIRAMA: Final[str] = "்"  # ்
_AYTHAM: Final[str] = "ஃ"  # ஃ

# Word endings that native Tamil phonotactics allows: sonorant + virama.
_NATIVE_FINAL_CONSONANTS: Final[frozenset[str]] = frozenset("ணநமனயரலளழ")

# Every Tamil consonant, enumerated rather than expressed as codepoint
# ranges - the block interleaves consonants with unused slots, and a wrong
# range would silently change what counts as a cluster.
_TAMIL_CONSONANT: Final[str] = "கஙசஜஞடணதநனபமயரறலளழவஶஷஸஹ"

# Consonant + virama + consonant at the very start of a token: an initial
# cluster, which native Tamil never has.
_INITIAL_CLUSTER: Final[re.Pattern[str]] = re.compile(
    f"^[{_TAMIL_CONSONANT}]{_VIRAMA}[{_TAMIL_CONSONANT}]"
)

_HAS_TAMIL: Final[re.Pattern[str]] = re.compile(r"[஀-௿]")


def _is_english_phonetic(token: str) -> bool:
    """Judge one whitespace-delimited token.

    Args:
        token: Token from a Tamil transcript, punctuation tolerated.

    Returns:
        ``True`` when the token violates native Tamil phonotactics in a way
        characteristic of transliterated English.
    """
    stripped = token.strip(".,!?;:\"'()[]{}")
    if not stripped or not _HAS_TAMIL.search(stripped):
        return False

    # Strong markers - each alone is decisive.
    if _INITIAL_CLUSTER.match(stripped):
        return True
    if _AYTHAM + "ப" in stripped or _AYTHAM + "ஜ" in stripped:
        return True
    if stripped.endswith(_VIRAMA) and len(stripped) >= 2:
        final_consonant = stripped[-2]
        if final_consonant not in _NATIVE_FINAL_CONSONANTS:
            # Native words end in vowels or sonorants; a final stop
            # (கிரெட், நேட்) or Grantha ஸ் (English plural: ப்ரான்ச்ஸ்)
            # does not occur in native orthography.
            return True

    return False


def flag_english_tokens(text: str) -> list[str]:
    """List the tokens of a transcript that look like transliterated English.

    Args:
        text: Tamil transcript.

    Returns:
        Flagged tokens in order of appearance, duplicates preserved.
    """
    return [token for token in text.split() if _is_english_phonetic(token)]


def english_phonetic_score(text: str) -> float:
    """Fraction of a transcript's substantive tokens that look like English.

    The denominator counts Tamil tokens of at least two characters:
    single-character fragments (the recogniser's stumbles over unfamiliar
    sounds, like the ``இ ப வ`` debris around a mangled English phrase) would
    otherwise dilute the score of exactly the utterances this exists to
    catch.

    Args:
        text: Tamil transcript.

    Returns:
        A value in ``[0.0, 1.0]``; ``0.0`` for empty input. Used to decide
        whether an utterance should be re-recognised as English rather than
        translated as Tamil.
    """
    substantive = [
        token
        for token in text.split()
        if _HAS_TAMIL.search(token) and len(token.strip(".,!?;:\"'()[]{}")) >= 2
    ]
    if not substantive:
        return 0.0
    flagged = sum(1 for token in substantive if _is_english_phonetic(token))
    return flagged / len(substantive)
