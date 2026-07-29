"""Real-model regression tests against the ground-truthed recording.

Every expectation here was confirmed by the speaker or measured in the
sessions that shaped the features - these tests exist so a model swap, a
quantisation change or a decode-option tweak that silently degrades any of
them fails loudly.

Assertions are deliberately tolerant of benign decode variance (substring
checks, not exact transcripts) but strict about the behaviours features
depend on: word-level segments, hotword recognition, Latin passthrough,
fusion keeping the Tamil.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_interpreter.application.services.code_switch import (
    english_phonetic_score,
    flag_english_tokens,
    has_native_anchor,
)
from ai_interpreter.application.services.transcript_fusion import fuse_transcripts
from ai_interpreter.domain.entities import Utterance, UtteranceId
from ai_interpreter.domain.value_objects import LanguageCode, LanguagePair, SampleRate

pytestmark = [pytest.mark.integration, pytest.mark.requires_model]

TAMIL = LanguageCode("ta")
ENGLISH = LanguageCode("en")
RATE = SampleRate(16000)


def _utterance(pcm: np.ndarray, language: LanguageCode, name: str = "u") -> Utterance:
    return Utterance(
        id=UtteranceId(name),
        pcm=pcm,
        sample_rate=RATE,
        started_at_ms=0.0,
        ended_at_ms=pcm.size / 16.0,
        language=language,
    )


class TestTamilRecognition:
    """IndicConformer against the speaker's ground truth."""

    def test_pure_tamil_is_recognised(self, real_container, slice_utterance) -> None:  # type: ignore[no-untyped-def]
        # Utterance 4: "நம நாளைக்கு அத முடிச்சிரலாம்" (confirmed pure Tamil).
        recognizer = real_container.create_recognizer(TAMIL)
        recognizer.warmup()

        transcript = recognizer.transcribe(_utterance(slice_utterance(4), TAMIL))

        assert "நாளைக்கு" in transcript.text
        assert english_phonetic_score(transcript.text) == 0.0  # never flagged

    def test_word_segments_cover_the_text(self, real_container, slice_utterance) -> None:  # type: ignore[no-untyped-def]
        # Fusion depends on per-word segments matching the text word count.
        recognizer = real_container.create_recognizer(TAMIL)
        recognizer.warmup()

        transcript = recognizer.transcribe(_utterance(slice_utterance(3), TAMIL))

        assert [segment.text for segment in transcript.segments] == transcript.text.split()
        starts = [segment.start_ms for segment in transcript.segments]
        assert starts == sorted(starts)

    def test_mixed_speech_is_flagged_with_an_anchor(self, real_container, slice_utterance) -> None:  # type: ignore[no-untyped-def]
        # Utterance 3: "matching மட்டும் pending ல இருக்கு" - the detector
        # must flag the transliterated English AND keep the native anchor
        # that routes it to fusion rather than the wholesale reroute.
        recognizer = real_container.create_recognizer(TAMIL)
        recognizer.warmup()

        transcript = recognizer.transcribe(_utterance(slice_utterance(3), TAMIL))

        assert flag_english_tokens(transcript.text)
        assert has_native_anchor(transcript.text)


class TestEnglishRecognition:
    """Whisper with hotword biasing."""

    def test_english_sentence_is_recognised(self, real_container, slice_utterance) -> None:  # type: ignore[no-untyped-def]
        # Utterance 6: "We need to solve that" (confirmed).
        recognizer = real_container.create_recognizer(ENGLISH, word_timestamps=True)
        recognizer.warmup()

        transcript = recognizer.transcribe(_utterance(slice_utterance(6), ENGLISH))

        assert "we need to solve" in transcript.text.lower()

    def test_word_timestamps_are_emitted(self, real_container, slice_utterance) -> None:  # type: ignore[no-untyped-def]
        recognizer = real_container.create_recognizer(ENGLISH, word_timestamps=True)
        recognizer.warmup()

        transcript = recognizer.transcribe(_utterance(slice_utterance(6), ENGLISH))

        assert len(transcript.segments) == len(transcript.text.split())


class TestFusionOnRealAudio:
    """The two models' views spliced by time, on the motivating utterance."""

    def test_mixed_utterance_fuses_keeping_the_tamil(  # type: ignore[no-untyped-def]
        self, real_container, slice_utterance
    ) -> None:
        pcm = slice_utterance(3)
        tamil = real_container.create_recognizer(TAMIL)
        english = real_container.create_recognizer(ENGLISH, word_timestamps=True)
        tamil.warmup()
        english.warmup()

        base = tamil.transcribe(_utterance(pcm, TAMIL))
        other = english.transcribe(_utterance(pcm, ENGLISH))
        fused = fuse_transcripts(base, other)

        assert fused is not None
        # The Tamil survives...
        assert "மட்டும்" in fused.text
        assert "இருக்கு" in fused.text
        # ...at least one English word was spliced in...
        assert any(word.isascii() for word in fused.text.split())
        # ...and Whisper's garbled rendering of மட்டும் stays out.
        assert "Mutum" not in fused.text


class TestTranslation:
    """IndicTrans2 behaviours features depend on."""

    def test_latin_terms_pass_through_untouched(self, real_container) -> None:  # type: ignore[no-untyped-def]
        # The property the whole glossary/fusion design rests on.
        translator = real_container.create_translator(LanguagePair.of("ta", "en"))
        translator.warmup()

        translation = translator.translate(
            "நாளைக்கு VALD session இருக்கு", LanguagePair.of("ta", "en")
        )

        assert "VALD" in translation.translated_text

    def test_tamil_translates_to_english(self, real_container) -> None:  # type: ignore[no-untyped-def]
        translator = real_container.create_translator(LanguagePair.of("ta", "en"))
        translator.warmup()

        translation = translator.translate("நாளைக்கு என்ன பண்ணனும்", LanguagePair.of("ta", "en"))

        assert translation.translated_text
        assert translation.translated_text.isascii()


class TestSynthesis:
    """The English voice produces real audio."""

    def test_english_voice_speaks(self, real_container) -> None:  # type: ignore[no-untyped-def]
        synthesizer = real_container.create_synthesizer(ENGLISH)
        synthesizer.warmup()

        chunk = synthesizer.synthesize("The VALD assessment is complete.", ENGLISH)

        assert chunk.pcm.size > 8000  # comfortably more than half a second
        assert float(np.max(np.abs(chunk.pcm))) > 0.01  # audibly non-silent
