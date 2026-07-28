"""Unit tests for the sherpa-onnx NeMo recognisers.

The native engines are replaced with fakes, exactly as in the Whisper tests:
what is under test is the adapter logic - especially the chunked commitment
algorithm, which is the piece that turns a linear-cost model into low
end-of-utterance latency and is also the piece most likely to harbour subtle
bugs. Real-model behaviour is verified through ``--transcribe`` against the
recorded Tamil reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_interpreter.domain.entities import AudioFrame, Utterance, UtteranceId
from ai_interpreter.domain.errors import ModelLoadError, TranscriptionError
from ai_interpreter.domain.value_objects import LanguageCode, SampleRate
from ai_interpreter.infrastructure.stt.sherpa_nemo import (
    SherpaNemoCtcRecognizer,
    SherpaNemoStreamingRecognizer,
    _join_tokens,
)

pytestmark = pytest.mark.unit

RATE = SampleRate(16000)
TAMIL = LanguageCode("ta")
ENGLISH = LanguageCode("en")


# ---------------------------------------------------------------------------
# Fakes for the offline (CTC) engine
# ---------------------------------------------------------------------------
class FakeOfflineResult:
    """Scripted result for one decode call."""

    def __init__(self, tokens: list[str], timestamps: list[float]) -> None:
        self.tokens = tokens
        self.timestamps = timestamps
        self.text = _join_tokens(tokens)


class FakeOfflineStream:
    """Records the waveform it was given."""

    def __init__(self, owner: FakeOfflineEngine) -> None:
        self._owner = owner
        self.samples: np.ndarray = np.empty(0, dtype=np.float32)
        self.result: FakeOfflineResult | None = None

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        assert sample_rate == 16000
        self.samples = np.asarray(samples, dtype=np.float32)


class FakeOfflineEngine:
    """Maps the decoded buffer to scripted token output.

    The script receives the raw sample buffer, which matters for the chunking
    tests: after a commit, the adapter *trims* the buffer, so the fake must
    know which stretch of absolute time it is looking at. Those tests encode
    the absolute time into the samples themselves (each frame is filled with
    its own start time in seconds), and the script reads it back - mimicking a
    real engine, whose timestamps are always relative to the audio it was
    handed.
    """

    def __init__(self, script: Any) -> None:
        self._script = script
        self.decode_durations: list[float] = []

    def create_stream(self) -> FakeOfflineStream:
        return FakeOfflineStream(self)

    def decode_stream(self, stream: FakeOfflineStream) -> None:
        self.decode_durations.append(round(stream.samples.size / 16000.0, 3))
        tokens, stamps = self._script(stream.samples)
        stream.result = FakeOfflineResult(tokens, stamps)


def _ctc(script: Any, **kwargs: Any) -> SherpaNemoCtcRecognizer:
    """Build an offline recogniser with a scripted fake engine.

    Args:
        script: Duration-to-output function for the fake engine.
        **kwargs: Constructor overrides.

    Returns:
        The recogniser with loading bypassed.
    """
    options: dict[str, Any] = {
        "model_path": Path("unused.onnx"),
        "tokens_path": Path("unused.txt"),
        "model_id": "indicconformer-ta",
        "language": TAMIL,
        "chunk_seconds": 2.0,
        "commit_margin_seconds": 0.5,
        "max_buffer_seconds": 8.0,
    }
    options.update(kwargs)
    recognizer = SherpaNemoCtcRecognizer(**options)
    recognizer._engine = FakeOfflineEngine(script)
    return recognizer


def _frames(seconds: float, frame_ms: int = 100) -> list[AudioFrame]:
    """Build frames whose samples encode their own absolute start time.

    Frame k is filled with the constant ``k * frame_ms / 1000`` so a scripted
    engine can recover which stretch of the utterance a trimmed buffer covers.

    Args:
        seconds: Total duration.
        frame_ms: Frame length.

    Returns:
        The frames, in order.
    """
    samples = RATE.samples_for_ms(frame_ms)
    count = int(seconds * 1000 / frame_ms)
    return [
        AudioFrame(
            pcm=np.full(samples, index * frame_ms / 1000.0, dtype=np.float32),
            sample_rate=RATE,
            timestamp_ms=index * frame_ms,
        )
        for index in range(count)
    ]


def _utterance(seconds: float = 2.0, language: LanguageCode | None = TAMIL) -> Utterance:
    return Utterance(
        id=UtteranceId("u1"),
        pcm=np.zeros(int(RATE.hz * seconds), dtype=np.float32),
        sample_rate=RATE,
        started_at_ms=0.0,
        ended_at_ms=seconds * 1000.0,
        language=language,
    )


class TestJoinTokens:
    """BPE token joining."""

    def test_joins_word_marked_tokens(self) -> None:
        assert _join_tokens(["▁என்", "▁பெ", "யர்"]) == "என் பெயர்"

    def test_empty_is_empty(self) -> None:
        assert _join_tokens([]) == ""


class TestOfflineTranscribe:
    """Whole-utterance decoding."""

    def test_produces_the_scripted_text(self) -> None:
        recognizer = _ctc(lambda samples: (["▁வண", "க்கம்"], [0.1, 0.5]))
        transcript = recognizer.transcribe(_utterance())

        assert transcript.text == "வணக்கம்"
        assert transcript.is_final
        assert transcript.language == TAMIL
        assert transcript.model_id == "indicconformer-ta"

    def test_empty_audio_yields_empty_text_with_zero_confidence(self) -> None:
        recognizer = _ctc(lambda samples: ([], []))
        transcript = recognizer.transcribe(_utterance())

        assert transcript.is_empty
        assert transcript.confidence.value == 0.0

    def test_nonempty_text_reports_availability_confidence(self) -> None:
        recognizer = _ctc(lambda samples: (["▁சரி"], [0.2]))
        assert recognizer.transcribe(_utterance()).confidence.value == 1.0

    def test_rejects_the_wrong_language(self) -> None:
        recognizer = _ctc(lambda samples: ([], []))
        with pytest.raises(TranscriptionError, match="only supports Tamil"):
            recognizer.transcribe(_utterance(language=ENGLISH))

    def test_rejects_the_wrong_sample_rate(self) -> None:
        recognizer = _ctc(lambda samples: ([], []))
        bad = Utterance(
            id=UtteranceId("u1"),
            pcm=np.zeros(48000, dtype=np.float32),
            sample_rate=SampleRate(48000),
            started_at_ms=0.0,
            ended_at_ms=1000.0,
            language=TAMIL,
        )
        with pytest.raises(TranscriptionError, match="require 16000 Hz"):
            recognizer.transcribe(bad)

    def test_supports_only_its_own_language(self) -> None:
        recognizer = _ctc(lambda samples: ([], []))
        assert recognizer.supports(TAMIL)
        assert not recognizer.supports(ENGLISH)

    def test_tracks_statistics(self) -> None:
        recognizer = _ctc(lambda samples: (["▁சரி"], [0.1]))
        recognizer.transcribe(_utterance())
        recognizer.transcribe(_utterance())

        assert recognizer.utterances_decoded == 2
        assert recognizer.mean_decode_ms >= 0.0


class TestChunkedStreaming:
    """The committed-chunk algorithm - the heart of low-latency Tamil.

    The fake script places one word per second of absolute utterance time:
    word k has tokens ``▁word{k}`` at k.0 and ``x`` at k.4. The absolute
    position of a (possibly trimmed) buffer is recovered from the encoded
    frame values, and timestamps are emitted relative to the buffer start -
    exactly the contract of the real engine. With chunk_seconds=2.0 and
    commit_margin=0.5, the first decode happens at 2.5 s of buffered audio
    and may commit words starting strictly before 2.0 s.
    """

    @staticmethod
    def _script(samples: np.ndarray) -> tuple[list[str], list[float]]:
        start = float(samples[0]) if samples.size else 0.0
        end = start + samples.size / 16000.0
        tokens: list[str] = []
        stamps: list[float] = []
        first_word = int(np.ceil(start - 1e-9))
        for word_index in range(first_word, int(np.floor(end + 1e-9)) + 1):
            if word_index >= end:
                break
            tokens.append(f"▁word{word_index}")
            stamps.append(word_index - start)
            if word_index + 0.4 < end:
                tokens.append("x")
                stamps.append(word_index - start + 0.4)
        return tokens, stamps

    def test_final_transcript_contains_every_word(self) -> None:
        recognizer = _ctc(self._script)
        results = list(recognizer.transcribe_stream(iter(_frames(6.0))))

        final = results[-1]
        assert final.is_final
        assert final.text == "word0x word1x word2x word3x word4x word5x"

    def test_partials_are_yielded_before_the_final(self) -> None:
        recognizer = _ctc(self._script)
        results = list(recognizer.transcribe_stream(iter(_frames(6.0))))

        partials = [transcript for transcript in results if not transcript.is_final]
        assert partials, "no interim transcripts were produced"
        assert all(not transcript.is_final for transcript in results[:-1])

    def test_committed_audio_is_discarded_from_the_buffer(self) -> None:
        # The whole point: each decode covers roughly one chunk, not the
        # entire utterance. Without commitment the last decode would cover
        # all 6 seconds.
        recognizer = _ctc(self._script)
        list(recognizer.transcribe_stream(iter(_frames(6.0))))

        engine = recognizer._engine
        assert isinstance(engine, FakeOfflineEngine)
        assert max(engine.decode_durations) < 4.0

    def test_words_are_never_split_across_a_commit(self) -> None:
        # Commit boundaries must fall on the ▁ word marker; a mid-word cut
        # would corrupt the joined text with a stray space.
        recognizer = _ctc(self._script)
        final = list(recognizer.transcribe_stream(iter(_frames(6.0))))[-1]

        assert "x " in final.text or final.text.endswith("x")
        for fragment in final.text.split():
            assert fragment.startswith("word"), f"split word detected: {fragment!r}"

    def test_short_utterance_produces_only_a_final(self) -> None:
        # Below one chunk there is nothing to commit incrementally.
        recognizer = _ctc(self._script)
        results = list(recognizer.transcribe_stream(iter(_frames(0.9))))

        assert len(results) == 1
        assert results[0].is_final
        assert results[0].text == "word0x"

    def test_boundaryless_speech_is_force_committed_at_the_ceiling(self) -> None:
        # A single unbroken "word" longer than max_buffer_seconds must not
        # grow the buffer without bound.
        def one_long_word(samples: np.ndarray) -> tuple[list[str], list[float]]:
            duration = samples.size / 16000.0
            return (["▁aaa"] + ["a"] * int(duration), [0.0] + [0.1] * int(duration))

        recognizer = _ctc(one_long_word, max_buffer_seconds=4.0)
        list(recognizer.transcribe_stream(iter(_frames(10.0))))

        engine = recognizer._engine
        assert isinstance(engine, FakeOfflineEngine)
        assert max(engine.decode_durations) <= 5.5

    def test_rejects_the_wrong_language(self) -> None:
        recognizer = _ctc(self._script)
        with pytest.raises(TranscriptionError, match="only supports Tamil"):
            list(recognizer.transcribe_stream(iter(_frames(1.0)), language=ENGLISH))

    def test_stream_counts_one_utterance(self) -> None:
        recognizer = _ctc(self._script)
        list(recognizer.transcribe_stream(iter(_frames(6.0))))

        assert recognizer.utterances_decoded == 1


class TestCommitIndex:
    """The boundary-selection rule in isolation."""

    def test_commits_up_to_the_last_boundary_before_the_limit(self) -> None:
        tokens = ["▁a", "x", "▁b", "y", "▁c"]
        stamps = [0.0, 0.4, 1.0, 1.4, 2.0]
        # limit 1.5: word c (at 2.0) is beyond it; commit everything before b?
        # No - b starts at 1.0 < 1.5, so tokens[:2] (word a) commit and b is
        # the first kept word.
        assert SherpaNemoCtcRecognizer._commit_index(tokens, stamps, 1.5) == 2

    def test_no_boundary_before_the_limit_returns_none(self) -> None:
        tokens = ["▁a", "x", "y"]
        stamps = [0.0, 0.4, 0.8]
        assert SherpaNemoCtcRecognizer._commit_index(tokens, stamps, 2.0) is None

    def test_never_commits_zero_tokens(self) -> None:
        # Index 0 would commit nothing while consuming nothing - livelock.
        tokens = ["▁a", "▁b"]
        stamps = [0.0, 5.0]
        assert SherpaNemoCtcRecognizer._commit_index(tokens, stamps, 0.5) is None

    def test_empty_input_returns_none(self) -> None:
        assert SherpaNemoCtcRecognizer._commit_index([], [], 1.0) is None


# ---------------------------------------------------------------------------
# Fakes for the online (streaming) engine
# ---------------------------------------------------------------------------
class FakeOnlineStream:
    def __init__(self) -> None:
        self.samples: list[np.ndarray] = []
        self.finished = False

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        assert sample_rate == 16000
        self.samples.append(np.asarray(samples, dtype=np.float32))

    def input_finished(self) -> None:
        self.finished = True


class FakeOnlineEngine:
    """Reveals one scripted word per second of audio received."""

    def __init__(self, words: list[str]) -> None:
        self._words = words

    def create_stream(self) -> FakeOnlineStream:
        return FakeOnlineStream()

    def is_ready(self, stream: FakeOnlineStream) -> bool:
        return False  # decoding is modelled as instantaneous

    def decode_stream(self, stream: FakeOnlineStream) -> None:  # pragma: no cover
        raise AssertionError("is_ready is always False")

    def get_result(self, stream: FakeOnlineStream) -> str:
        seconds = sum(chunk.size for chunk in stream.samples) / 16000.0
        revealed = self._words[: int(seconds)]
        return " ".join(revealed)


def _online(words: list[str], **kwargs: Any) -> SherpaNemoStreamingRecognizer:
    options: dict[str, Any] = {
        "model_path": Path("unused.onnx"),
        "tokens_path": Path("unused.txt"),
        "model_id": "nemo-streaming-en",
        "language": ENGLISH,
    }
    options.update(kwargs)
    recognizer = SherpaNemoStreamingRecognizer(**options)
    recognizer._engine = FakeOnlineEngine(words)
    return recognizer


class TestOnlineStreaming:
    """The thin wrapper over the natively streaming model."""

    def test_yields_growing_partials_then_a_final(self) -> None:
        recognizer = _online(["hi", "hello", "there"])
        results = list(recognizer.transcribe_stream(iter(_frames(3.0))))

        texts = [transcript.text for transcript in results]
        assert texts == ["hi", "hi hello", "hi hello there", "hi hello there"]
        assert [transcript.is_final for transcript in results] == [
            False,
            False,
            False,
            True,
        ]

    def test_unchanged_text_is_not_re_yielded(self) -> None:
        recognizer = _online(["hi"])
        results = list(recognizer.transcribe_stream(iter(_frames(3.0))))

        partials = [transcript for transcript in results if not transcript.is_final]
        assert len(partials) == 1

    def test_transcribe_returns_the_full_text(self) -> None:
        recognizer = _online(["hi", "hello"])
        transcript = recognizer.transcribe(_utterance(2.0, language=ENGLISH))

        assert transcript.text == "hi hello"
        assert transcript.is_final

    def test_rejects_the_wrong_language(self) -> None:
        recognizer = _online(["hi"])
        with pytest.raises(TranscriptionError, match="only supports English"):
            recognizer.transcribe(_utterance(1.0, language=TAMIL))

    def test_supports_only_its_own_language(self) -> None:
        recognizer = _online([])
        assert recognizer.supports(ENGLISH)
        assert not recognizer.supports(TAMIL)

    def test_counts_utterances(self) -> None:
        recognizer = _online(["hi"])
        list(recognizer.transcribe_stream(iter(_frames(2.0))))
        recognizer.transcribe(_utterance(1.0, language=ENGLISH))

        assert recognizer.utterances_decoded == 2


class TestLoading:
    """Missing files fail clearly for both adapters."""

    def test_offline_missing_model_reports_the_path(self, tmp_path: Path) -> None:
        recognizer = SherpaNemoCtcRecognizer(
            model_path=tmp_path / "absent.onnx",
            tokens_path=tmp_path / "absent.txt",
            model_id="indicconformer-ta",
            language=TAMIL,
        )
        with pytest.raises(ModelLoadError, match=r"absent\.onnx"):
            recognizer.warmup()

    def test_online_missing_model_reports_the_path(self, tmp_path: Path) -> None:
        recognizer = SherpaNemoStreamingRecognizer(
            model_path=tmp_path / "absent.onnx",
            tokens_path=tmp_path / "absent.txt",
            model_id="nemo-streaming-en",
            language=ENGLISH,
        )
        with pytest.raises(ModelLoadError, match=r"absent\.onnx"):
            recognizer.warmup()

    def test_close_is_idempotent(self) -> None:
        recognizer = _ctc(lambda samples: ([], []))
        recognizer.close()
        recognizer.close()
