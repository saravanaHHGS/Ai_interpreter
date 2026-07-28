"""Unit tests for the VITS synthesizer adapter.

The sherpa-onnx engine is replaced with a fake, as throughout: what is under
test is sentence-chunked streaming, chunk metadata, language enforcement and
error handling. Real voice quality is judged by ear via ``--speak`` and the
saved WAV files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_interpreter.domain.errors import ModelLoadError, SynthesisError
from ai_interpreter.domain.value_objects import LanguageCode
from ai_interpreter.infrastructure.tts.sherpa_vits import (
    SherpaVitsSynthesizer,
    split_sentences,
)

pytestmark = pytest.mark.unit

TAMIL = LanguageCode("ta")
ENGLISH = LanguageCode("en")


class FakeGenerated:
    """Stand-in for sherpa's GeneratedAudio."""

    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate


class FakeTtsEngine:
    """Produces 100 samples per input character, recording every call."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self.calls: list[dict[str, Any]] = []

    def generate(self, text: str, sid: int = 0, speed: float = 1.0) -> FakeGenerated:
        self.calls.append({"text": text, "sid": sid, "speed": speed})
        return FakeGenerated(np.zeros(max(1, 100 * len(text)), dtype=np.float32), self.sample_rate)


def _synth(**kwargs: Any) -> tuple[SherpaVitsSynthesizer, FakeTtsEngine]:
    """Build a synthesizer with a fake engine injected.

    Args:
        **kwargs: Constructor overrides.

    Returns:
        The synthesizer and its fake engine.
    """
    options: dict[str, Any] = {
        "model_path": Path("unused.onnx"),
        "tokens_path": Path("unused.txt"),
        "model_id": "piper-en-lessac",
        "language": ENGLISH,
    }
    options.update(kwargs)
    synthesizer = SherpaVitsSynthesizer(**options)
    fake = FakeTtsEngine()
    synthesizer._engine = fake
    return synthesizer, fake


class TestSplitSentences:
    """Sentence chunking for streamed synthesis."""

    def test_splits_on_terminators(self) -> None:
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_splits_on_danda(self) -> None:
        assert split_sentences("पहला। दूसरा।") == ["पहला।", "दूसरा।"]

    def test_unpunctuated_text_is_one_sentence(self) -> None:
        assert split_sentences("no punctuation here") == ["no punctuation here"]

    def test_empty_text_yields_nothing(self) -> None:
        assert split_sentences("   ") == []

    def test_does_not_split_inside_a_sentence(self) -> None:
        # A terminator not followed by whitespace (decimals, abbreviations
        # glued to text) must not split.
        assert split_sentences("version 2.5 is out. yes.") == ["version 2.5 is out.", "yes."]


class TestSynthesize:
    """Whole-text synthesis."""

    def test_produces_one_final_chunk(self) -> None:
        synthesizer, _ = _synth()
        chunk = synthesizer.synthesize("Hello there.", ENGLISH)

        assert chunk.is_last
        assert chunk.chunk_index == 0
        assert chunk.pcm.size > 0
        assert chunk.sample_rate.hz == 22050
        assert chunk.voice_id == "piper-en-lessac"

    def test_empty_text_yields_an_empty_final_chunk(self) -> None:
        synthesizer, engine = _synth()
        chunk = synthesizer.synthesize("   ", ENGLISH)

        assert chunk.pcm.size == 0
        assert chunk.is_last
        assert engine.calls == []

    def test_rejects_the_wrong_language(self) -> None:
        synthesizer, _ = _synth()
        with pytest.raises(SynthesisError, match="speaks English, not Tamil"):
            synthesizer.synthesize("வணக்கம்", TAMIL)

    def test_speed_multiplies_the_configured_default(self) -> None:
        synthesizer, engine = _synth(speed=1.2)
        synthesizer.synthesize("Hello", ENGLISH, speed=1.5)

        assert engine.calls[0]["speed"] == pytest.approx(1.8)

    def test_tracks_statistics(self) -> None:
        synthesizer, _ = _synth()
        synthesizer.synthesize("One.", ENGLISH)
        synthesizer.synthesize("Two.", ENGLISH)

        assert synthesizer.chunks_generated == 2
        assert synthesizer.mean_chunk_ms >= 0.0

    def test_engine_failure_is_wrapped(self) -> None:
        class ExplodingEngine(FakeTtsEngine):
            def generate(self, text: str, sid: int = 0, speed: float = 1.0) -> FakeGenerated:
                raise RuntimeError("vits exploded")

        synthesizer, _ = _synth()
        synthesizer._engine = ExplodingEngine()
        with pytest.raises(SynthesisError, match="Synthesis failed"):
            synthesizer.synthesize("Hello", ENGLISH)

    def test_silent_output_is_an_error(self) -> None:
        class SilentEngine(FakeTtsEngine):
            def generate(self, text: str, sid: int = 0, speed: float = 1.0) -> FakeGenerated:
                return FakeGenerated(np.empty(0, dtype=np.float32), 22050)

        synthesizer, _ = _synth()
        synthesizer._engine = SilentEngine()
        with pytest.raises(SynthesisError, match="produced no audio"):
            synthesizer.synthesize("Hello", ENGLISH)


class TestSynthesizeStream:
    """Sentence-by-sentence streaming."""

    def test_yields_one_chunk_per_sentence(self) -> None:
        synthesizer, engine = _synth()
        chunks = list(synthesizer.synthesize_stream("One. Two. Three.", ENGLISH))

        assert len(chunks) == 3
        assert [call["text"] for call in engine.calls] == ["One.", "Two.", "Three."]

    def test_chunk_indices_and_last_flag(self) -> None:
        synthesizer, _ = _synth()
        chunks = list(synthesizer.synthesize_stream("One. Two. Three.", ENGLISH))

        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
        assert [chunk.is_last for chunk in chunks] == [False, False, True]

    def test_single_sentence_is_immediately_last(self) -> None:
        synthesizer, _ = _synth()
        chunks = list(synthesizer.synthesize_stream("Just one sentence.", ENGLISH))

        assert len(chunks) == 1
        assert chunks[0].is_last

    def test_splitting_can_be_disabled(self) -> None:
        synthesizer, engine = _synth(sentence_split=False)
        chunks = list(synthesizer.synthesize_stream("One. Two.", ENGLISH))

        assert len(chunks) == 1
        assert engine.calls[0]["text"] == "One. Two."

    def test_empty_text_yields_one_empty_chunk(self) -> None:
        synthesizer, _ = _synth()
        chunks = list(synthesizer.synthesize_stream("  ", ENGLISH))

        assert len(chunks) == 1
        assert chunks[0].pcm.size == 0
        assert chunks[0].is_last

    def test_rejects_the_wrong_language(self) -> None:
        synthesizer, _ = _synth()
        with pytest.raises(SynthesisError, match="speaks English"):
            list(synthesizer.synthesize_stream("வணக்கம்", TAMIL))


class TestVoiceReporting:
    """Capability and voice listing."""

    def test_supports_only_its_own_language(self) -> None:
        synthesizer, _ = _synth()
        assert synthesizer.supports(ENGLISH)
        assert not synthesizer.supports(TAMIL)

    def test_voices_lists_the_single_voice(self) -> None:
        synthesizer, _ = _synth(voice_name="Lessac")
        voices = synthesizer.voices()

        assert len(voices) == 1
        assert voices[0].name == "Lessac"
        assert voices[0].language == ENGLISH
        assert voices[0].sample_rate.hz == 22050

    def test_voices_filters_by_language(self) -> None:
        synthesizer, _ = _synth()
        assert synthesizer.voices(TAMIL) == ()


class TestLoading:
    """Missing files fail clearly."""

    def test_missing_model_reports_the_path(self, tmp_path: Path) -> None:
        synthesizer = SherpaVitsSynthesizer(
            model_path=tmp_path / "absent.onnx",
            tokens_path=tmp_path / "absent.txt",
            model_id="piper-en-lessac",
            language=ENGLISH,
        )
        with pytest.raises(ModelLoadError, match=r"absent\.onnx"):
            synthesizer.warmup()

    def test_close_is_idempotent(self) -> None:
        synthesizer, _ = _synth()
        synthesizer.close()
        synthesizer.close()
