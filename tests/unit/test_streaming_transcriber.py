"""Unit tests for the per-utterance streaming transcriber."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from ai_interpreter.application.services.streaming_transcriber import UtteranceStreamer
from ai_interpreter.domain.entities import AudioFrame, Transcript, UtteranceId
from ai_interpreter.domain.errors import TranscriptionError
from ai_interpreter.domain.value_objects import Confidence, LanguageCode, SampleRate

pytestmark = pytest.mark.unit

RATE = SampleRate(16000)
TAMIL = LanguageCode("ta")


def _frame(index: int = 0) -> AudioFrame:
    return AudioFrame(
        pcm=np.zeros(512, dtype=np.float32),
        sample_rate=RATE,
        timestamp_ms=index * 32.0,
    )


def _transcript(text: str, *, final: bool) -> Transcript:
    return Transcript(
        utterance_id=UtteranceId("stream"),
        text=text,
        language=TAMIL,
        confidence=Confidence(1.0),
        is_final=final,
    )


class FakeStreamingRecognizer:
    """Yields one partial per frame consumed, then a final on exhaustion."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.frames_seen = 0

    def transcribe_stream(
        self,
        frames: Iterator[AudioFrame],
        language: LanguageCode | None = None,
    ) -> Iterator[Transcript]:
        if self.fail:
            raise TranscriptionError("scripted streaming failure")
        for _ in frames:
            self.frames_seen += 1
            yield _transcript(f"word{self.frames_seen}", final=False)
        yield _transcript(" ".join(f"word{i + 1}" for i in range(self.frames_seen)), final=True)


class TestUtteranceStreamer:
    """One utterance, one thread, one final transcript."""

    def test_final_transcript_covers_every_frame(self) -> None:
        recognizer = FakeStreamingRecognizer()
        streamer = UtteranceStreamer(recognizer, TAMIL)  # type: ignore[arg-type]
        for index in range(3):
            streamer.push(_frame(index))
        streamer.finish()

        result = streamer.result(timeout=5.0)

        assert result is not None
        assert result.text == "word1 word2 word3"
        assert streamer.error is None

    def test_partials_reach_the_callback(self) -> None:
        partials: list[str] = []
        streamer = UtteranceStreamer(
            FakeStreamingRecognizer(),  # type: ignore[arg-type]
            TAMIL,
            on_partial=lambda transcript: partials.append(transcript.text),
        )
        streamer.push(_frame(0))
        streamer.push(_frame(1))
        streamer.finish()
        streamer.result(timeout=5.0)

        assert partials == ["word1", "word2"]

    def test_decode_failure_yields_none_and_records_the_error(self) -> None:
        streamer = UtteranceStreamer(FakeStreamingRecognizer(fail=True), TAMIL)  # type: ignore[arg-type]
        streamer.push(_frame())
        streamer.finish()

        assert streamer.result(timeout=5.0) is None
        assert isinstance(streamer.error, TranscriptionError)

    def test_finish_is_idempotent_and_frames_after_finish_are_dropped(self) -> None:
        recognizer = FakeStreamingRecognizer()
        streamer = UtteranceStreamer(recognizer, TAMIL)  # type: ignore[arg-type]
        streamer.push(_frame(0))
        streamer.finish()
        streamer.finish()
        streamer.push(_frame(1))  # late frame; must not deadlock or count

        result = streamer.result(timeout=5.0)

        assert result is not None
        assert recognizer.frames_seen == 1

    def test_result_timeout_returns_none(self) -> None:
        class NeverFinishes:
            def transcribe_stream(
                self,
                frames: Iterator[AudioFrame],
                language: LanguageCode | None = None,
            ) -> Iterator[Transcript]:
                for _ in frames:  # pragma: no cover - consumed forever
                    time.sleep(0.05)
                    yield _transcript("x", final=False)
                yield _transcript("x", final=True)

        streamer = UtteranceStreamer(NeverFinishes(), TAMIL)  # type: ignore[arg-type]
        streamer.push(_frame())
        # finish() is never called: the stream is still waiting for frames.
        assert streamer.result(timeout=0.2) is None
        streamer.finish()  # release the thread

    def test_frames_are_consumed_from_another_thread(self) -> None:
        # The real topology: capture thread pushes, worker thread collects.
        recognizer = FakeStreamingRecognizer()
        streamer = UtteranceStreamer(recognizer, TAMIL)  # type: ignore[arg-type]

        def feed() -> None:
            for index in range(5):
                streamer.push(_frame(index))
            streamer.finish()

        thread = threading.Thread(target=feed)
        thread.start()
        result = streamer.result(timeout=5.0)
        thread.join()

        assert result is not None
        assert recognizer.frames_seen == 5
