"""Indic script transliteration by Unicode block offset.

Why this exists - and why the first translation attempt produced garbage
------------------------------------------------------------------------

IndicTrans2's tokenizer was trained on text with **all Indic scripts unified
into Devanagari**. Feed it raw Tamil and the sentencepiece model falls back to
one character per piece; the encoder then sees a sequence it never saw in
training, and the decoder produces confident nonsense ("En Beyer
Saravandakumar"). Transliterate the same sentence to Devanagari first and the
pieces become proper subwords, and the same model translates correctly.

The transliteration itself is nearly free: the major Indic script blocks in
Unicode are **deliberately aligned** - each script occupies a 128-codepoint
block laid out in the same order (ISCII heritage), so Tamil ``க`` (0x0B95) and
Devanagari ``क`` (0x0915) differ by a constant offset. Shifting codepoints is
exactly what the official IndicProcessor does under the hood (via
indic-nlp-library, which would drag in pandas and morfessor as dependencies
for what is, for our languages, a ten-line loop).

Characters outside the source block - punctuation, digits, Latin - pass
through unchanged, matching the training-time behaviour. Urdu is excluded:
it is written in Perso-Arabic script, which has no aligned block.
"""

from __future__ import annotations

from typing import Final

from ai_interpreter.domain.value_objects import LanguageCode

__all__ = [
    "INDICTRANS2_TAGS",
    "from_devanagari",
    "supports_script_mapping",
    "to_devanagari",
]

_DEVANAGARI_BASE: Final[int] = 0x0900
_BLOCK_SIZE: Final[int] = 0x80

# Base codepoint of each language's script block. Devanagari-script languages
# map with offset zero (identity). Urdu is deliberately absent - Perso-Arabic
# has no aligned block.
_SCRIPT_BASES: Final[dict[str, int]] = {
    "hi": 0x0900,  # Devanagari (identity)
    "mr": 0x0900,  # Devanagari (identity)
    "bn": 0x0980,
    "pa": 0x0A00,  # Gurmukhi
    "gu": 0x0A80,
    "ta": 0x0B80,
    "te": 0x0C00,
    "kn": 0x0C80,
    "ml": 0x0D00,
}

# IndicTrans2 language tags: the model's own names for each language, used as
# the first two source tokens of every translation request.
INDICTRANS2_TAGS: Final[dict[str, str]] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
    "gu": "guj_Gujr",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "ur": "urd_Arab",
}


def supports_script_mapping(language: LanguageCode) -> bool:
    """Whether a language's script can be offset-mapped to Devanagari.

    Args:
        language: Language to check.

    Returns:
        ``True`` when transliteration is available for it.
    """
    return language.code in _SCRIPT_BASES


def to_devanagari(text: str, language: LanguageCode) -> str:
    """Shift a language's script block onto Devanagari.

    Args:
        text: Text in the language's native script.
        language: Its language. Devanagari-script languages return the text
            unchanged.

    Returns:
        The text with in-block characters shifted; everything else untouched.

    Raises:
        KeyError: If the language has no aligned script block.
    """
    base = _SCRIPT_BASES[language.code]
    if base == _DEVANAGARI_BASE:
        return text
    offset = _DEVANAGARI_BASE - base
    return "".join(
        chr(ord(ch) + offset) if base <= ord(ch) < base + _BLOCK_SIZE else ch for ch in text
    )


def from_devanagari(text: str, language: LanguageCode) -> str:
    """Shift Devanagari output back into a language's native script.

    Args:
        text: Model output in Devanagari.
        language: Target language whose script to restore.

    Returns:
        The text with Devanagari characters shifted into the target block;
        everything else untouched.

    Raises:
        KeyError: If the language has no aligned script block.
    """
    base = _SCRIPT_BASES[language.code]
    if base == _DEVANAGARI_BASE:
        return text
    offset = base - _DEVANAGARI_BASE
    return "".join(
        chr(ord(ch) + offset)
        if _DEVANAGARI_BASE <= ord(ch) < _DEVANAGARI_BASE + _BLOCK_SIZE
        else ch
        for ch in text
    )
