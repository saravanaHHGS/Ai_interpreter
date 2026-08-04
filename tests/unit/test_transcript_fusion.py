"""Unit tests for word-level transcript fusion.

The fixtures replay utterances 2 and 3 of the live Tanglish recording that
motivated the feature: the Tamil conformer's actual word timings and the
English recogniser's actual words, reduced to round numbers.
"""

from __future__ import annotations

import pytest

from ai_interpreter.application.services.transcript_fusion import fuse_transcripts
from ai_interpreter.domain.entities import Transcript, TranscriptSegment, UtteranceId
from ai_interpreter.domain.value_objects import Confidence, LanguageCode

pytestmark = pytest.mark.unit

TAMIL = LanguageCode("ta")
ENGLISH = LanguageCode("en")


def _transcript(
    words: list[tuple[str, float, float]],
    language: LanguageCode = TAMIL,
    text: str | None = None,
) -> Transcript:
    """Build a word-level transcript from (text, start_ms, end_ms) triples.

    Args:
        words: One triple per word.
        language: Transcript language.
        text: Full text override; defaults to the joined words.

    Returns:
        The transcript.
    """
    segments = tuple(
        TranscriptSegment(text=word, start_ms=start, end_ms=end, confidence=Confidence(0.9))
        for word, start, end in words
    )
    return Transcript(
        utterance_id=UtteranceId("u1"),
        text=text if text is not None else " ".join(word for word, _, _ in words),
        language=language,
        confidence=Confidence(0.9),
        is_final=True,
        segments=segments,
    )


class TestFusion:
    """The splice itself."""

    def test_mixed_sentence_keeps_tamil_and_gains_english(self) -> None:
        # Utterance 3 of the live recording: "matching மட்டும் pending ல
        # இருக்கு". The conformer transliterates the English; Whisper garbles
        # the Tamil ("Matching Mutum Bending"). Fusion takes each side's
        # good half.
        tamil = _transcript(
            [
                ("மேட்சிங்", 400, 900),
                ("மட்டும்", 900, 1300),
                ("பெண்டிங்", 1300, 1700),
                ("ல", 1700, 1800),
                ("இருக்கு", 1800, 2200),
            ]
        )
        english = _transcript(
            [
                ("Matching", 400, 950),
                ("Mutum", 950, 1250),
                ("Bending.", 1300, 1750),
            ],
            language=ENGLISH,
        )

        result = fuse_transcripts(tamil, english)

        assert result is not None
        assert result.text == "Matching மட்டும் Bending ல இருக்கு"
        assert result.replaced == ("மேட்சிங்", "பெண்டிங்")
        assert result.inserted == ("Matching", "Bending.")

    def test_adjacent_flagged_words_form_one_window(self) -> None:
        # Utterance 2: "வேர்ல்ட் அஸிஸ்மெண்ட் முடிஞ்சிடுச்சு" - two flagged
        # words, one English phrase. Both English words land in the merged
        # window; the Tamil verb survives.
        tamil = _transcript(
            [
                ("வேர்ல்ட்", 400, 900),
                ("அஸிஸ்மெண்ட்", 900, 1500),
                ("முடிஞ்சிடுச்சு", 1500, 2100),
            ]
        )
        english = _transcript(
            [("World", 400, 900), ("assessment.", 900, 1500)],
            language=ENGLISH,
        )

        result = fuse_transcripts(tamil, english)

        assert result is not None
        assert result.text == "World assessment முடிஞ்சிடுச்சு"

    def test_trailing_punctuation_is_stripped_from_the_splice(self) -> None:
        tamil = _transcript([("பெண்டிங்", 0, 500), ("இருக்கு", 500, 1000)])
        english = _transcript([("Pending.", 0, 500)], language=ENGLISH)

        result = fuse_transcripts(tamil, english)

        assert result is not None
        assert result.text == "Pending இருக்கு"

    def test_region_with_no_english_words_keeps_its_tamil(self) -> None:
        # "No evidence" must never become "delete the user's speech".
        tamil = _transcript([("பெண்டிங்", 3000, 3500), ("இருக்கு", 3500, 4000)])
        english = _transcript([("Hello", 0, 400)], language=ENGLISH)

        assert fuse_transcripts(tamil, english) is None

    def test_real_timings_splice_correctly_and_reject_hallucinations(self) -> None:
        # The EXACT timings both models produced on live utterance 3.
        # Whisper's "Mutum" - its garbled rendering of the TAMIL word
        # மட்டும் - must stay out of both flagged regions, while "Matching"
        # (whose onset Whisper smeared back to 0 ms) must still splice in.
        tamil = _transcript(
            [
                ("மேட்சிங்", 400, 880),
                ("மட்டும்", 880, 1120),
                ("பெண்டிங்", 1120, 1520),
                ("ல", 1520, 1680),
                ("இருக்கு", 1680, 2730),
            ]
        )
        english = _transcript(
            [
                ("Matching", 0, 780),
                ("Mutum", 780, 1100),
                ("Bending.", 1100, 1420),
            ],
            language=ENGLISH,
        )

        result = fuse_transcripts(tamil, english)

        assert result is not None
        assert result.text == "Matching மட்டும் Bending ல இருக்கு"

    def test_early_onset_words_still_match_by_their_end(self) -> None:
        # Live utterance 2: Whisper heard "World" at 0-760 ms against the
        # conformer's வேர்ல்ட் at 400-840 ms. Whisper smears word onsets
        # backward through leading silence; the word's END is what places it.
        tamil = _transcript(
            [
                ("வேர்ல்ட்", 400, 840),
                ("அஸிஸ்மெண்ட்", 840, 1520),
                ("முடிஞ்சிடுச்சு", 1520, 3020),
            ]
        )
        english = _transcript(
            [("World", 0, 760), ("assessment.", 760, 1200)],
            language=ENGLISH,
        )

        result = fuse_transcripts(tamil, english)

        assert result is not None
        assert result.text == "World assessment முடிஞ்சிடுச்சு"


class TestClockAlignment:
    """Fusing against the streaming partner, whose clock runs ~30% slow.

    The fixtures are the EXACT timings both models produced on live
    utterance 3: the conformer on real time, the NeMo streaming partner on
    its own compressed frame clock (word onsets 0.32-1.00 s for speech that
    really spans 0.40-1.68 s).
    """

    TAMIL = [
        ("மேட்சிங்", 400, 880),
        ("மட்டும்", 880, 1120),
        ("பெண்டிங்", 1120, 1520),
        ("ல", 1520, 1680),
        ("இருக்கு", 1680, 2700),
    ]
    PARTNER = [
        ("matching", 320, 560),
        ("mottum", 560, 800),
        ("bending", 800, 1000),
        ("work", 1000, 1240),
    ]

    def test_aligned_partner_words_splice_correctly(self) -> None:
        tamil = _transcript(self.TAMIL)
        partner = _transcript(self.PARTNER, language=ENGLISH)

        result = fuse_transcripts(tamil, partner, align_clock=True)

        assert result is not None
        # Each side's good half: the English words land in their flagged
        # regions; the partner's garbled renderings of TAMIL words
        # ("mottum" for மட்டும், "work" for இருக்கு) stay out.
        assert result.text == "matching மட்டும் bending ல இருக்கு"

    def test_without_alignment_the_raw_clock_misfuses(self) -> None:
        # The reason align_clock exists: raw partner times are compressed,
        # so region matching cannot land every word where it belongs.
        tamil = _transcript(self.TAMIL)
        partner = _transcript(self.PARTNER, language=ENGLISH)

        result = fuse_transcripts(tamil, partner, align_clock=False)

        assert result is None or result.text != "matching மட்டும் bending ல இருக்கு"

    def test_short_spans_are_left_unscaled(self) -> None:
        # One word gives a line-fit nothing to hold on to.
        tamil = _transcript([("பெண்டிங்", 0, 500), ("இருக்கு", 500, 1000)])
        partner = _transcript([("pending", 0, 400)], language=ENGLISH)

        result = fuse_transcripts(tamil, partner, align_clock=True)

        assert result is not None
        assert result.text == "pending இருக்கு"


class TestDeclining:
    """Fusion refuses rather than guessing."""

    def test_nothing_flagged_declines(self) -> None:
        tamil = _transcript([("நாளைக்கு", 0, 500), ("என்ன", 500, 900)])
        english = _transcript([("Hello", 0, 500)], language=ENGLISH)

        assert fuse_transcripts(tamil, english) is None

    def test_base_without_word_segments_declines(self) -> None:
        # One covering segment (a recogniser without word timestamps) is the
        # wrong granularity to splice at.
        tamil = _transcript(
            [("பெண்டிங் இருக்கு", 0, 1000)],
            text="பெண்டிங் இருக்கு",
        )
        english = _transcript([("Pending", 0, 500)], language=ENGLISH)

        assert fuse_transcripts(tamil, english) is None

    def test_other_without_word_segments_declines(self) -> None:
        tamil = _transcript([("பெண்டிங்", 0, 500)])
        english = _transcript(
            [("Pending now", 0, 800)],
            language=ENGLISH,
            text="Pending now",
        )

        assert fuse_transcripts(tamil, english) is None

    def test_empty_secondary_declines(self) -> None:
        tamil = _transcript([("பெண்டிங்", 0, 500)])
        english = _transcript([], language=ENGLISH, text="")

        assert fuse_transcripts(tamil, english) is None
