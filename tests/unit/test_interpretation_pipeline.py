"""Unit tests for the interpretation pipeline.

Every component is a fake, so ordering, backpressure, retries, barge-in and
the captions fallback are verified deterministically. The real components are
each proven separately; what lives here is the orchestration - the part where
threading bugs breed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from ai_interpreter.application.pipeline.interpretation import (
    InterpretationPipeline,
    PipelineEvents,
    UtteranceTiming,
)
from ai_interpreter.application.services.utterance_segmenter import SegmenterState
from ai_interpreter.domain.entities import (
    SpeechAudio,
    Transcript,
    Translation,
    Utterance,
    UtteranceId,
)
from ai_interpreter.domain.errors import TranscriptionError, TranslationError
from ai_interpreter.domain.value_objects import Confidence, LanguageCode, LanguagePair, SampleRate

pytestmark = pytest.mark.unit

RATE = SampleRate(16000)
PAIR = LanguagePair.of("ta", "en")
TAMIL = LanguageCode("ta")
ENGLISH = LanguageCode("en")


class FakeCapture:
    """Stands in for CaptureSession: the pipeline only wires callbacks."""

    def __init__(self) -> None:
        self._on_utterance: Any = None
        self._on_state_change: Any = None
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True

    @property
    def is_running(self) -> bool:
        return self.started and not self.stopped


class FakeRecognizer:
    """Transcribes by script; can fail a set number of times first."""

    def __init__(self, fail_times: int = 0, text_prefix: str = "heard") -> None:
        self.fail_times = fail_times
        self.text_prefix = text_prefix
        self.calls = 0

    def transcribe(self, utterance: Utterance) -> Transcript:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TranscriptionError("scripted stt failure")
        return Transcript(
            utterance_id=utterance.id,
            text=f"{self.text_prefix} {utterance.id}",
            language=TAMIL,
            confidence=Confidence(0.9),
            is_final=True,
        )

    def supports(self, language: LanguageCode) -> bool:
        return bool(language)

    def warmup(self) -> None: ...
    def close(self) -> None: ...

    @property
    def model_id(self) -> str:
        return "fake-stt"


class EmptyRecognizer(FakeRecognizer):
    """Returns empty transcripts, as silence would."""

    def transcribe(self, utterance: Utterance) -> Transcript:
        self.calls += 1
        return Transcript(
            utterance_id=utterance.id,
            text="",
            language=TAMIL,
            confidence=Confidence(0.0),
            is_final=True,
        )


class FakeTranslator:
    """Translates by prefixing; can fail a set number of times first."""

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def translate(self, text: str, pair: LanguagePair) -> Translation:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TranslationError("scripted mt failure")
        return Translation(
            utterance_id=UtteranceId("mt"),
            source_text=text,
            translated_text=f"T[{text}]",
            pair=pair,
        )

    def supports(self, pair: LanguagePair) -> bool:
        return bool(pair)

    def warmup(self) -> None: ...
    def close(self) -> None: ...

    @property
    def model_id(self) -> str:
        return "fake-mt"


class FakeSynthesizer:
    """Streams two chunks per text."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    @property
    def provider_id(self) -> str:
        return "fake-tts"

    def supports(self, language: LanguageCode) -> bool:
        return bool(language)

    def voices(self, language: LanguageCode | None = None) -> tuple:
        return ()

    def synthesize(
        self, text: str, language: LanguageCode, voice_id: str | None = None, speed: float = 1.0
    ):
        raise AssertionError("streaming path expected")

    def synthesize_stream(
        self,
        text: str,
        language: LanguageCode,
        voice_id: str | None = None,
        speed: float = 1.0,
    ) -> Iterator[SpeechAudio]:
        self.texts.append(text)
        for index in range(2):
            yield SpeechAudio(
                utterance_id=UtteranceId("tts"),
                pcm=np.full(800, 0.1, dtype=np.float32),
                sample_rate=RATE,
                language=language,
                chunk_index=index,
                is_last=index == 1,
                latency_ms=5.0,
            )

    def warmup(self) -> None: ...
    def close(self) -> None: ...


class FakeSink:
    """Records everything written to it."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.cleared = 0
        self.flushed = 0
        self.written: list[SpeechAudio] = []
        self._device = None

    @property
    def device(self):
        return self._device

    @property
    def is_open(self) -> bool:
        return self.opened and not self.closed

    def open(self) -> None:
        self.opened = True

    def write(self, audio: SpeechAudio) -> None:
        self.written.append(audio)

    def flush(self, timeout: float | None = None) -> None:
        self.flushed += 1

    def clear(self) -> None:
        self.cleared += 1

    def close(self) -> None:
        self.closed = True


def _utterance(name: str, seconds: float = 1.0) -> Utterance:
    return Utterance(
        id=UtteranceId(name),
        pcm=np.zeros(int(RATE.hz * seconds), dtype=np.float32),
        sample_rate=RATE,
        started_at_ms=0.0,
        ended_at_ms=seconds * 1000.0,
        language=TAMIL,
    )


def _build(
    recognizer: FakeRecognizer | None = None,
    translator: FakeTranslator | None = None,
    synthesizer: FakeSynthesizer | None = None,
    sink: FakeSink | None = None,
    events: PipelineEvents | None = None,
    **kwargs: Any,
) -> tuple[InterpretationPipeline, FakeCapture, FakeSink | None]:
    capture = FakeCapture()
    sink = sink if sink is not None else FakeSink()
    pipeline = InterpretationPipeline(
        capture=capture,  # type: ignore[arg-type]
        recognizer=recognizer or FakeRecognizer(),
        translator=translator or FakeTranslator(),
        synthesizer=synthesizer if synthesizer is not None else FakeSynthesizer(),
        sink=sink,
        pair=PAIR,
        events=events,
        retry_backoff_s=0.01,
        **kwargs,
    )
    return pipeline, capture, sink


def _wait_done(pipeline: InterpretationPipeline, expected: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = pipeline.stats()
        resolved = (
            stats.utterances_out + stats.dropped_backpressure + stats.dropped_empty + stats.failures
        )
        if resolved >= expected:
            return
        time.sleep(0.01)
    raise AssertionError(f"pipeline did not resolve {expected} utterances in time")


class TestHappyPath:
    """One utterance, all the way through."""

    def test_interprets_and_speaks(self) -> None:
        synthesizer = FakeSynthesizer()
        pipeline, capture, sink = _build(synthesizer=synthesizer)
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        assert capture.started and capture.stopped
        assert synthesizer.texts == ["T[heard u1]"]
        assert sink is not None
        assert len(sink.written) == 2
        assert sink.flushed >= 1
        assert sink.closed

    def test_events_fire_in_order(self) -> None:
        order: list[str] = []
        events = PipelineEvents(
            on_transcript=lambda t: order.append(f"stt:{t.text}"),
            on_translation=lambda t: order.append(f"mt:{t.translated_text}"),
            on_timing=lambda t: order.append("timing"),
        )
        pipeline, _, _ = _build(events=events)
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        assert order == ["stt:heard u1", "mt:T[heard u1]", "timing"]

    def test_multiple_utterances_stay_ordered(self) -> None:
        synthesizer = FakeSynthesizer()
        pipeline, _, _ = _build(synthesizer=synthesizer, queue_maxsize=10)
        pipeline.start()
        try:
            for index in range(4):
                pipeline.submit_utterance(_utterance(f"u{index}"))
            _wait_done(pipeline, 4)
        finally:
            pipeline.stop()

        assert synthesizer.texts == [f"T[heard u{index}]" for index in range(4)]

    def test_timing_records_the_stages(self) -> None:
        timings: list[UtteranceTiming] = []
        pipeline, _, _ = _build(events=PipelineEvents(on_timing=timings.append))
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1", seconds=2.0))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        timing = timings[0]
        assert timing.audio_ms == pytest.approx(2000.0)
        assert timing.eou_to_first_audio_ms >= timing.stt_ms
        assert timing.total_ms >= timing.eou_to_first_audio_ms


class TestBackpressure:
    """The queue must not grow when the machine cannot keep up."""

    def test_oldest_utterance_is_dropped(self) -> None:
        # Submit before start so nothing is consumed yet.
        pipeline, _, _ = _build(queue_maxsize=2)
        for index in range(5):
            pipeline.submit_utterance(_utterance(f"u{index}"))

        stats = pipeline.stats()
        assert stats.utterances_in == 5
        assert stats.dropped_backpressure == 3

        synthesizer = pipeline._synthesizer
        pipeline.start()
        try:
            _wait_done(pipeline, 5)
        finally:
            pipeline.stop()
        assert isinstance(synthesizer, FakeSynthesizer)
        # The two newest survived.
        assert synthesizer.texts == ["T[heard u3]", "T[heard u4]"]


class TestEmptyResults:
    """Silence and empty translations are dropped quietly."""

    def test_empty_transcript_skips_translation(self) -> None:
        translator = FakeTranslator()
        pipeline, _, _ = _build(recognizer=EmptyRecognizer(), translator=translator)
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        assert translator.calls == 0
        assert pipeline.stats().dropped_empty == 1


class TestRetries:
    """Bounded retries, then a clean drop."""

    def test_one_failure_is_retried_and_succeeds(self) -> None:
        recognizer = FakeRecognizer(fail_times=1)
        errors: list[str] = []
        pipeline, _, _ = _build(
            recognizer=recognizer,
            events=PipelineEvents(on_error=lambda stage, exc: errors.append(stage)),
            max_retries=1,
        )
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        assert recognizer.calls == 2
        assert errors == []
        assert pipeline.stats().utterances_out == 1

    def test_exhausted_retries_drop_the_utterance(self) -> None:
        recognizer = FakeRecognizer(fail_times=5)
        errors: list[str] = []
        pipeline, _, _ = _build(
            recognizer=recognizer,
            events=PipelineEvents(on_error=lambda stage, exc: errors.append(stage)),
            max_retries=1,
        )
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        assert recognizer.calls == 2
        assert errors == ["stt"]
        stats = pipeline.stats()
        assert stats.failures == 1
        assert stats.utterances_out == 0

    def test_mt_failure_after_retries_still_reported(self) -> None:
        errors: list[str] = []
        pipeline, _, _ = _build(
            translator=FakeTranslator(fail_times=5),
            events=PipelineEvents(on_error=lambda stage, exc: errors.append(stage)),
            max_retries=1,
        )
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        assert errors == ["mt"]

    def test_pipeline_survives_and_processes_the_next_utterance(self) -> None:
        recognizer = FakeRecognizer(fail_times=2)  # kills u1 entirely
        pipeline, _, _ = _build(recognizer=recognizer, max_retries=1, queue_maxsize=5)
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            pipeline.submit_utterance(_utterance("u2"))
            _wait_done(pipeline, 2)
        finally:
            pipeline.stop()

        stats = pipeline.stats()
        assert stats.failures == 1
        assert stats.utterances_out == 1


class TestBargeIn:
    """New speech silences the interpreter."""

    def test_speech_onset_clears_the_sink(self) -> None:
        pipeline, _, sink = _build()
        pipeline.start()
        try:
            assert sink is not None
            pipeline._on_capture_state(SegmenterState.SPEECH)
        finally:
            pipeline.stop()

        assert sink.cleared == 1
        assert pipeline.stats().barge_ins == 1

    def test_state_events_are_forwarded(self) -> None:
        states: list[SegmenterState] = []
        pipeline, _, _ = _build(events=PipelineEvents(on_state=states.append))
        pipeline.start()
        try:
            pipeline._on_capture_state(SegmenterState.SPEECH)
            pipeline._on_capture_state(SegmenterState.SILENCE)
        finally:
            pipeline.stop()

        assert states == [SegmenterState.SPEECH, SegmenterState.SILENCE]


class TestCaptionsOnly:
    """The designed fallback when the target language has no voice."""

    def test_runs_without_synthesizer_or_sink(self) -> None:
        translations: list[str] = []
        capture = FakeCapture()
        pipeline = InterpretationPipeline(
            capture=capture,  # type: ignore[arg-type]
            recognizer=FakeRecognizer(),
            translator=FakeTranslator(),
            synthesizer=None,
            sink=None,
            pair=PAIR,
            events=PipelineEvents(on_translation=lambda t: translations.append(t.translated_text)),
        )
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        assert translations == ["T[heard u1]"]
        assert pipeline.stats().utterances_out == 1


class TestLifecycle:
    """Start and stop discipline."""

    def test_stop_is_idempotent(self) -> None:
        pipeline, _, _ = _build()
        pipeline.start()
        pipeline.stop()
        pipeline.stop()

    def test_start_is_idempotent(self) -> None:
        pipeline, _, _ = _build()
        pipeline.start()
        worker = pipeline._worker
        pipeline.start()
        assert pipeline._worker is worker
        pipeline.stop()

    def test_worker_runs_off_the_caller_thread(self) -> None:
        threads: list[str] = []
        events = PipelineEvents(
            on_transcript=lambda t: threads.append(threading.current_thread().name)
        )
        pipeline, _, _ = _build(events=events)
        pipeline.start()
        try:
            pipeline.submit_utterance(_utterance("u1"))
            _wait_done(pipeline, 1)
        finally:
            pipeline.stop()

        assert threads == ["interpretation"]
