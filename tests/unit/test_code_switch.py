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
    has_native_anchor,
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


class TestNativeAnchor:
    """Mixed-vs-fully-English discrimination, on real session sentences.

    Score alone cannot separate them: the mixed "வேர்ல்ட் அஸிஸ்மெண்ட்
    முடிஞ்சிடுச்சு" scored HIGHER (0.67) than the fully-English "வி நீட் டூ
    சால்வ் தத்" (0.60) in the live recording. The anchor - a long unflagged
    native word - is what separates keep-the-Tamil from replace-it-all.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "வேர்ல்ட் அஸிஸ்மெண்ட் முடிஞ்சிடுச்சு",  # anchor: முடிஞ்சிடுச்சு
            "மேட்சிங் மட்டும் பெண்டிங் ல இருக்கு",  # anchors: மட்டும், இருக்கு
            "நெக்ஸ்ட் நவேகேஷன் இஸ்ஸு ரெண்டு மூணு இதுல இருக்கு",
        ],
    )
    def test_mixed_sentences_have_an_anchor(self, text: str) -> None:
        assert has_native_anchor(text)

    @pytest.mark.parametrize(
        "text",
        [
            "வி நீட் டூ சால்வ் தத்",  # "we need to solve that"
            "இ பஸ் கன் கிரெட் மோர்",  # "e-bus can create more"
        ],
    )
    def test_fully_english_sentences_have_none(self, text: str) -> None:
        # Only short debris (வி, கன், மோர்) remains unflagged; none of it is
        # long enough to prove the sentence contains real Tamil.
        assert not has_native_anchor(text)

    def test_latin_words_are_not_anchors(self) -> None:
        assert not has_native_anchor("Pending status ready")

    def test_empty_text_has_no_anchor(self) -> None:
        assert not has_native_anchor("")
