"""Unit tests for the English-in-Tamil-script detector.

Every fixture here is a real transcript from the live session that motivated
the feature - the recogniser's actual renderings of code-switched speech,
and real pure-Tamil sentences that must never be flagged.
"""

from __future__ import annotations

import pytest

from ai_interpreter.application.services.code_switch import (
    english_phonetic_score,
    flag_english_tokens,
)

pytestmark = pytest.mark.unit


class TestFlagging:
    """Token-level detection."""

    @pytest.mark.parametrize(
        "token",
        [
            "ட்ரை",  # "try" - initial cluster
            "ட்ரி",  # "try" again
            "ப்ரான்ச்ஸ்",  # "branches" - cluster + English plural
            "டிஃபெரெண்ட்",  # "different" - contains f
            "கிரெட்",  # "create" - final stop
            "நேட்",  # "mate" - final stop
            "பஸ்",  # "bus" - final Grantha
        ],
    )
    def test_flags_real_transliterations(self, token: str) -> None:
        assert flag_english_tokens(token) == [token]

    @pytest.mark.parametrize(
        "token",
        [
            "நாளைக்கு",
            "என்ன",
            "பண்ணனும்",
            "முடிச்சோம்",
            "சொன்ன",
            "இருக்கிறீர்கள்",
            "வணக்கம்",
            "மென்பொருள்",  # ends ள் - a native final
            "சரவணகுமார்",  # ends ர் - a native final
        ],
    )
    def test_never_flags_native_tamil(self, token: str) -> None:
        assert flag_english_tokens(token) == []

    def test_latin_tokens_are_not_flagged(self) -> None:
        # Already-English text needs no rescue.
        assert flag_english_tokens("world research") == []

    def test_punctuation_does_not_hide_a_flag(self) -> None:
        assert flag_english_tokens("கிரெட்.") == ["கிரெட்."]


class TestScore:
    """Utterance-level scoring, on real sentences from the session."""

    def test_english_heavy_sentences_score_high(self) -> None:
        # "e-bus can create more branches", transliterated.
        assert english_phonetic_score("இ பஸ் கன் கிரெட் மோர் பிரான்ச") >= 0.3
        assert english_phonetic_score("இ சுுபா ப வ கன் கிரெட் மோர் பிரான்ச்ஸ்") >= 0.3

    def test_mixed_sentence_scores_moderately(self) -> None:
        score = english_phonetic_score("நம்ம டிஃபெரெண்ட் ப்ரான்ச்ஸ் அ கிரெட் பண்ண முடியும்")
        assert 0.3 <= score <= 0.7

    def test_single_english_word_scores_low(self) -> None:
        # One term inside Tamil belongs to the glossary path, not a reroute.
        assert english_phonetic_score("இத நம்ம ட்ரி பண்ணலாமா") < 0.3

    @pytest.mark.parametrize(
        "text",
        [
            "நாளைக்கு என்ன பண்ணனும்",
            "நீ சொன்ன தான் முடிச்சோம்",
            "என் பெயர் சரவணகுமார் இன்று நான் தமிழ் குரல் அடையாளம் சரியாக செயல்படுகிறதா என்று சோதித்து வருகின்றேன்",
            "இந்த மென்பொருள் என் குரலை சரியாக எழுத்தாக மாற்றுகிறதா",
        ],
    )
    def test_pure_tamil_scores_zero(self, text: str) -> None:
        # False positives would corrupt correct Tamil; zero is mandatory.
        assert english_phonetic_score(text) == 0.0

    def test_single_char_debris_does_not_dilute(self) -> None:
        # The recogniser's stumbles over English produce one-character
        # fragments; they must not lower the score of what they surround.
        with_debris = english_phonetic_score("இ ப வ கிரெட் பிரான்ச்ஸ்")
        assert with_debris >= 0.5

    def test_empty_text_scores_zero(self) -> None:
        assert english_phonetic_score("") == 0.0
        assert english_phonetic_score("   ") == 0.0
