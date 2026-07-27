"""Unit tests for the Whisper recogniser.

The CTranslate2 model is replaced with a fake. Downloading 145 MB of weights
and spending 1.7 seconds per decode would make the suite unusable, and none
of the behaviour under test here belongs to the model: it is the conversion
from Whisper's output into domain transcripts, the confidence calculation,
and the error handling.

Real model behaviour is measured by ``--benchmark`` and exercised by the
integration test, which is marked ``requires_model`` so it can be skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_interpreter.domain.entities import Utterance, UtteranceId
from ai_interpreter.domain.errors import ModelLoadError, TranscriptionError
from ai_interpreter.domain.value_objects import LanguageCode, SampleRate
from ai_interpreter.infrastructure.stt.faster_whisper import (
    FasterWhisperRecognizer,
    WhisperDecodeOptions,
    _confidence_from_log_prob,
    frames_from_audio,
    is_repetitive,
)

pytestmark = pytest.mark.unit

RATE = SampleRate(16000)


@dataclass
class FakeSegment:
    """Stand-in for a faster-whisper segment."""

    text: str
    start: float
    end: float
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.01
    compression_ratio: float = 1.2


class FakeInfo:
    """Stand-in for faster-whisper's TranscriptionInfo."""

    def __init__(self, language: str | None = "en") -> None:
        self.language = language
        self.language_probability = 0.99


class FakeWhisperModel:
    """Records calls and returns scripted segments.

    Args:
        segments: Segments to return from every transcribe call.
        info: Detection info to return.
        error: Exception to raise instead of returning, if any.
    """

    def __init__(
        self,
        segments: list[FakeSegment] | None = None,
        info: FakeInfo | None = None,
        error: Exception | None = None,
    ) -> None:
        self._segments = segments if segments is not None else []
        self._info = info or FakeInfo()
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, FakeInfo]:
        if self._error is not None:
            raise self._error
        self.calls.append({"samples": int(np.asarray(audio).size), **kwargs})
        return iter(list(self._segments)), self._info


def _recognizer(model: FakeWhisperModel, **kwargs: Any) -> FasterWhisperRecognizer:
    """Build a recogniser with its model already injected.

    Args:
        model: Fake model to use.
        **kwargs: Constructor overrides.

    Returns:
        The recogniser, with loading bypassed.
    """
    defaults: dict[str, Any] = {
        "model_dir": Path("unused"),
        "model_id": "whisper-base",
        "language": LanguageCode("en"),
    }
    defaults.update(kwargs)
    recognizer = FasterWhisperRecognizer(**defaults)
    recognizer._model = model
    return recognizer


def _utterance(seconds: float = 2.0, rate: SampleRate = RATE) -> Utterance:
    """Build a silent utterance of a given length.

    Args:
        seconds: Duration.
        rate: Sample rate.

    Returns:
        The utterance.
    """
    return Utterance(
        id=UtteranceId("u1"),
        pcm=np.zeros(int(rate.hz * seconds), dtype=np.float32),
        sample_rate=rate,
        started_at_ms=0.0,
        ended_at_ms=seconds * 1000.0,
    )


class TestConfidence:
    """Turning a log probability into a confidence score."""

    def test_zero_log_prob_is_full_confidence(self) -> None:
        assert _confidence_from_log_prob(0.0) == pytest.approx(1.0)

    def test_typical_value_is_high(self) -> None:
        # -0.2 average log probability is a healthy Whisper decode.
        assert _confidence_from_log_prob(-0.2) == pytest.approx(0.819, abs=1e-3)

    def test_poor_value_is_low(self) -> None:
        assert _confidence_from_log_prob(-2.0) == pytest.approx(0.135, abs=1e-3)

    def test_is_monotonic(self) -> None:
        values = [_confidence_from_log_prob(x) for x in (-3.0, -2.0, -1.0, -0.5, 0.0)]
        assert values == sorted(values)

    def test_handles_negative_infinity(self) -> None:
        assert _confidence_from_log_prob(-math.inf) == 0.0

    def test_clamps_positive_values(self) -> None:
        assert _confidence_from_log_prob(5.0) == 1.0


class TestTranscribe:
    """Producing a final transcript."""

    def test_joins_segments_into_text(self) -> None:
        model = FakeWhisperModel(
            [FakeSegment(" Hello there.", 0.0, 1.0), FakeSegment(" How are you?", 1.0, 2.0)]
        )
        transcript = _recognizer(model).transcribe(_utterance())

        assert transcript.text == "Hello there. How are you?"
        assert transcript.is_final is True

    def test_preserves_segment_timestamps(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hello", 0.5, 1.25)])
        transcript = _recognizer(model).transcribe(_utterance())

        assert transcript.segments[0].start_ms == pytest.approx(500.0)
        assert transcript.segments[0].end_ms == pytest.approx(1250.0)

    def test_computes_confidence_from_log_probability(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hello", 0.0, 1.0, avg_logprob=-0.2)])
        transcript = _recognizer(model).transcribe(_utterance())

        assert transcript.confidence.value == pytest.approx(0.819, abs=1e-3)

    def test_averages_confidence_across_segments(self) -> None:
        model = FakeWhisperModel(
            [
                FakeSegment("A", 0.0, 1.0, avg_logprob=-0.1),
                FakeSegment("B", 1.0, 2.0, avg_logprob=-0.3),
            ]
        )
        transcript = _recognizer(model).transcribe(_utterance())

        assert transcript.confidence.value == pytest.approx(math.exp(-0.2), abs=1e-3)

    def test_empty_result_is_valid(self) -> None:
        # Silence is a legitimate outcome, not a failure.
        transcript = _recognizer(FakeWhisperModel([])).transcribe(_utterance())

        assert transcript.is_empty
        assert transcript.confidence.value == 0.0

    def test_skips_whitespace_only_segments(self) -> None:
        model = FakeWhisperModel([FakeSegment("   ", 0.0, 1.0), FakeSegment("Hi", 1.0, 2.0)])
        transcript = _recognizer(model).transcribe(_utterance())

        assert transcript.text == "Hi"
        assert len(transcript.segments) == 1

    def test_records_latency_and_model_id(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        transcript = _recognizer(model, model_id="whisper-tiny").transcribe(_utterance())

        assert transcript.model_id == "whisper-tiny"
        assert transcript.latency_ms >= 0.0

    def test_tracks_running_statistics(self) -> None:
        recognizer = _recognizer(FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)]))
        for _ in range(3):
            recognizer.transcribe(_utterance())

        assert recognizer.utterances_decoded == 3
        assert recognizer.mean_decode_ms >= 0.0

    def test_rejects_the_wrong_sample_rate(self) -> None:
        # Whisper resamples silently, which would hide a pipeline bug.
        recognizer = _recognizer(FakeWhisperModel([]))
        with pytest.raises(TranscriptionError, match="requires 16000 Hz"):
            recognizer.transcribe(_utterance(rate=SampleRate(48000)))

    def test_wraps_model_failures(self) -> None:
        model = FakeWhisperModel(error=RuntimeError("CTranslate2 exploded"))
        with pytest.raises(TranscriptionError, match="Whisper decoding failed"):
            _recognizer(model).transcribe(_utterance())


class TestDecodeOptions:
    """Options passed through to faster-whisper."""

    def test_uses_greedy_decoding_by_default(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        _recognizer(model).transcribe(_utterance())

        assert model.calls[0]["beam_size"] == 1

    def test_disables_conditioning_on_previous_text(self) -> None:
        # Conditioning lets one hallucination poison every later utterance.
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        _recognizer(model).transcribe(_utterance())

        assert model.calls[0]["condition_on_previous_text"] is False

    def test_disables_whispers_own_vad(self) -> None:
        # Silero already segmented the audio; a second detector would trim
        # the trailing silence the first one deliberately kept.
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        _recognizer(model).transcribe(_utterance())

        assert model.calls[0]["vad_filter"] is False

    def test_passes_the_configured_language(self) -> None:
        model = FakeWhisperModel([FakeSegment("வணக்கம்", 0.0, 1.0)])
        _recognizer(model, language=LanguageCode("ta")).transcribe(_utterance())

        assert model.calls[0]["language"] == "ta"

    def test_auto_detects_when_no_language_is_set(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        _recognizer(model, language=None).transcribe(_utterance())

        assert model.calls[0]["language"] is None

    def test_utterance_language_overrides_the_default(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        recognizer = _recognizer(model, language=LanguageCode("en"))
        utterance = Utterance(
            id=UtteranceId("u1"),
            pcm=np.zeros(16000, dtype=np.float32),
            sample_rate=RATE,
            started_at_ms=0.0,
            ended_at_ms=1000.0,
            language=LanguageCode("hi"),
        )
        recognizer.transcribe(utterance)

        assert model.calls[0]["language"] == "hi"

    def test_honours_a_custom_beam_size(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        _recognizer(model, options=WhisperDecodeOptions(beam_size=5)).transcribe(_utterance())

        assert model.calls[0]["beam_size"] == 5


class TestRepetitionDetection:
    """Catching Whisper's decoder loops.

    The strings here are verbatim from a live Tamil session on the target
    machine, including the confidence scores they were produced with.
    """

    # Observed at confidence 0.91 and 0.93, while correct transcripts from the
    # same session scored 0.46 to 0.67. Confidence is inverted on loops.
    LOOP_A = " ".join(["பண்டும்"] * 32)
    LOOP_B = " ".join(["கட்டிக்"] * 32)

    def test_detects_a_single_word_loop(self) -> None:
        assert is_repetitive(self.LOOP_A, min_word_diversity=0.4)
        assert is_repetitive(self.LOOP_B, min_word_diversity=0.4)

    def test_accepts_ordinary_tamil_speech(self) -> None:
        # Verbatim from the same session, a plausible transcript.
        text = "இது நாங்கள் திருப்பனி பார்த்தும் ஆனா இங்களைக் கடைக்கில்லை."
        assert not is_repetitive(text, min_word_diversity=0.4)

    def test_accepts_ordinary_english_speech(self) -> None:
        assert not is_repetitive("Hi, hello, how are you doing today?", 0.4)

    def test_ignores_short_utterances(self) -> None:
        # "yes yes" and "no no no" are things people actually say.
        assert not is_repetitive("no no no", min_word_diversity=0.4)
        assert not is_repetitive("yes yes", min_word_diversity=0.4)

    def test_detects_a_repeated_phrase(self) -> None:
        assert is_repetitive("thank you thank you thank you thank you", 0.4)

    def test_discards_a_repetitive_transcript(self) -> None:
        # A loop is more confident by construction, so it must be rejected on
        # structure rather than on confidence.
        model = FakeWhisperModel([FakeSegment(self.LOOP_A, 0.0, 8.0, avg_logprob=-0.09)])
        transcript = _recognizer(model).transcribe(_utterance())

        assert transcript.is_empty
        assert transcript.confidence.value > 0.9

    def test_discards_a_segment_that_compresses_too_well(self) -> None:
        model = FakeWhisperModel([FakeSegment("looping text", 0.0, 1.0, compression_ratio=3.5)])
        assert _recognizer(model).transcribe(_utterance()).is_empty

    def test_keeps_a_segment_with_a_normal_compression_ratio(self) -> None:
        model = FakeWhisperModel(
            [FakeSegment("normal speech here", 0.0, 1.0, compression_ratio=1.2)]
        )
        assert _recognizer(model).transcribe(_utterance()).text == "normal speech here"

    def test_keeps_good_segments_when_one_is_repetitive(self) -> None:
        model = FakeWhisperModel(
            [
                FakeSegment("Good morning everyone", 0.0, 1.0, compression_ratio=1.1),
                FakeSegment("loop loop loop", 1.0, 2.0, compression_ratio=4.0),
            ]
        )
        assert _recognizer(model).transcribe(_utterance()).text == "Good morning everyone"

    def test_repetition_penalty_is_passed_to_the_decoder(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        _recognizer(model).transcribe(_utterance())

        assert model.calls[0]["repetition_penalty"] > 1.0

    def test_compression_threshold_is_passed_to_the_decoder(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        _recognizer(model).transcribe(_utterance())

        assert model.calls[0]["compression_ratio_threshold"] == pytest.approx(2.4)


class TestConfidenceFloor:
    """Discarding low-confidence transcripts."""

    def test_discards_below_the_floor(self) -> None:
        # Speaking a confident wrong sentence is worse than saying nothing.
        model = FakeWhisperModel([FakeSegment("garbled", 0.0, 1.0, avg_logprob=-3.0)])
        recognizer = _recognizer(model, options=WhisperDecodeOptions(min_confidence=0.5))
        transcript = recognizer.transcribe(_utterance())

        assert transcript.is_empty
        assert transcript.segments == ()

    def test_keeps_results_above_the_floor(self) -> None:
        model = FakeWhisperModel([FakeSegment("clear speech", 0.0, 1.0, avg_logprob=-0.1)])
        recognizer = _recognizer(model, options=WhisperDecodeOptions(min_confidence=0.5))

        assert recognizer.transcribe(_utterance()).text == "clear speech"

    def test_floor_is_disabled_by_default(self) -> None:
        model = FakeWhisperModel([FakeSegment("garbled", 0.0, 1.0, avg_logprob=-5.0)])
        assert _recognizer(model).transcribe(_utterance()).text == "garbled"


class TestLanguageResolution:
    """Deciding which language to record on a transcript."""

    def test_uses_the_detected_language(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)], info=FakeInfo("hi"))
        transcript = _recognizer(model, language=None).transcribe(_utterance())

        assert transcript.language == LanguageCode("hi")

    def test_ignores_a_language_the_application_does_not_model(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)], info=FakeInfo("sv"))
        transcript = _recognizer(model, language=LanguageCode("en")).transcribe(_utterance())

        assert transcript.language == LanguageCode("en")

    def test_falls_back_to_english_when_nothing_is_known(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)], info=FakeInfo(None))
        transcript = _recognizer(model, language=None).transcribe(_utterance())

        assert transcript.language == LanguageCode("en")


class TestSupportedLanguages:
    """Language capability reporting."""

    @pytest.mark.parametrize("code", ["ta", "en", "hi", "te", "ml"])
    def test_supports_the_required_languages(self, code: str) -> None:
        assert _recognizer(FakeWhisperModel()).supports(LanguageCode(code))


class TestStreaming:
    """Interim transcripts during a long utterance."""

    def test_yields_a_final_transcript(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hello", 0.0, 1.0)])
        frames = frames_from_audio(np.zeros(16000, dtype=np.float32), RATE)
        results = list(_recognizer(model).transcribe_stream(frames))

        assert results[-1].is_final is True

    def test_emits_interim_results_on_long_input(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hello", 0.0, 1.0)])
        # 10 seconds, with interim decodes every 3 seconds.
        frames = frames_from_audio(np.zeros(16000 * 10, dtype=np.float32), RATE)
        results = list(_recognizer(model).transcribe_stream(frames))

        interim = [transcript for transcript in results if not transcript.is_final]
        assert len(interim) >= 2

    def test_short_input_produces_only_a_final_result(self) -> None:
        # Interim decodes cost a full encoder pass, so they are not worth
        # running on an utterance shorter than the interval.
        model = FakeWhisperModel([FakeSegment("Hi", 0.0, 1.0)])
        frames = frames_from_audio(np.zeros(16000, dtype=np.float32), RATE)
        results = list(_recognizer(model).transcribe_stream(frames))

        assert len(results) == 1
        assert results[0].is_final

    def test_only_the_final_result_counts_towards_statistics(self) -> None:
        model = FakeWhisperModel([FakeSegment("Hello", 0.0, 1.0)])
        recognizer = _recognizer(model)
        frames = frames_from_audio(np.zeros(16000 * 10, dtype=np.float32), RATE)
        list(recognizer.transcribe_stream(frames))

        assert recognizer.utterances_decoded == 1


class TestFramesFromAudio:
    """Splitting a buffer into frames."""

    def test_produces_whole_frames(self) -> None:
        frames = list(frames_from_audio(np.zeros(16000, dtype=np.float32), RATE, frame_ms=32))
        assert all(frame.pcm.size == 512 for frame in frames)

    def test_pads_the_final_partial_frame(self) -> None:
        frames = list(frames_from_audio(np.zeros(600, dtype=np.float32), RATE, frame_ms=32))
        assert len(frames) == 2
        assert frames[-1].pcm.size == 512

    def test_timestamps_advance(self) -> None:
        frames = list(frames_from_audio(np.zeros(2048, dtype=np.float32), RATE, frame_ms=32))
        assert [frame.timestamp_ms for frame in frames] == [0, 32, 64, 96]


class TestWarmup:
    """Priming the model before the first real utterance."""

    def test_warmup_uses_the_configured_decode_options(self) -> None:
        # Regression: warmup called the model directly, bypassing these
        # options. faster-whisper's default temperature fallback then ran up
        # to six decoding passes on silence, and warmup took 56 seconds.
        model = FakeWhisperModel([])
        _recognizer(model).warmup()

        assert model.calls[0]["temperature"] == 0.0
        assert model.calls[0]["condition_on_previous_text"] is False
        assert model.calls[0]["beam_size"] == 1

    def test_warmup_does_not_count_as_a_decoded_utterance(self) -> None:
        recognizer = _recognizer(FakeWhisperModel([]))
        recognizer.warmup()

        assert recognizer.utterances_decoded == 0

    def test_warmup_reports_model_failures_as_load_errors(self) -> None:
        model = FakeWhisperModel(error=RuntimeError("bad weights"))
        with pytest.raises(ModelLoadError, match="warmup failed"):
            _recognizer(model).warmup()


class TestModelLoading:
    """Failure to load is reported clearly."""

    def test_missing_directory_is_reported(self, tmp_path: Path) -> None:
        recognizer = FasterWhisperRecognizer(
            model_dir=tmp_path / "absent",
            model_id="whisper-base",
        )
        with pytest.raises(ModelLoadError, match="model directory not found"):
            recognizer.warmup()

    def test_close_is_idempotent(self) -> None:
        recognizer = _recognizer(FakeWhisperModel())
        recognizer.close()
        recognizer.close()
