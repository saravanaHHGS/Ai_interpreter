"""The live interpretation pipeline: utterances in, translated speech out.

This is where the components built in Phases 3-7 become one interpreter::

    CaptureSession ──▶ bounded queue ──▶ worker: STT ▶ MT ▶ TTS ▶ sink
         │ (speech onset)                                      │
         └────────────── sink.clear()  ◀───────────────────────┘  barge-in

Design decisions, each traceable to a measurement or a constraint:

**Threads, not asyncio.** Every stage is a blocking native call (CTranslate2,
sherpa-onnx) that releases the GIL, so plain threads already deliver real
parallelism where it exists. On the 2-core target the profile mandates a
*serial* inference lane anyway - three models contending for two cores was
measured slower than running them in sequence - which reduces the pipeline to
one worker consuming one queue. An event loop would add machinery without
adding overlap. The parallel lane (one worker per stage) is a Phase 10 change
inside this class only; nothing outside it knows the difference.

**Drop-oldest backpressure.** The utterance queue is bounded
(``pipeline.queue_maxsize``); when the speaker outruns the machine, the
*oldest* unprocessed utterance is discarded and counted. In live
interpretation, stale speech is worthless - translating something said eight
seconds ago while the speaker continues is worse than skipping it.

**Barge-in clears audio, not text.** When speech onset is detected while
translated audio is still playing, the sink queue is dropped so the
interpreter stops talking over the speaker. The in-flight utterance's *text*
still completes: transcripts and translations remain useful as captions even
when their audio was pre-empted.

**Retries are bounded and stage-local.** A stage failure is retried once
(``pipeline.max_retries``) after a short backoff; a second failure drops the
utterance with an error event. The pipeline itself never dies from a model
error - a session that silently stops interpreting while the UI still shows
"live" is the worst failure mode, so worker exceptions always surface through
``on_error``.

**Latency is measured per utterance, end to end.** ``UtteranceTiming``
records the wall-clock from end-of-utterance to first audio written to the
sink - the EOU→FTS metric defined in Phase 1 - plus per-stage costs, so the
Performance page and the ``--interpret`` summary report reality rather than
estimates.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TypeVar

from ai_interpreter.application.services.capture_session import CaptureSession
from ai_interpreter.application.services.code_switch import (
    english_phonetic_score,
    flag_english_tokens,
)
from ai_interpreter.application.services.glossary import GlossaryRewriter
from ai_interpreter.application.services.utterance_segmenter import SegmenterState
from ai_interpreter.domain.entities import Transcript, Translation, Utterance
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.domain.ports import (
    AudioSink,
    SpeechRecognizer,
    SpeechSynthesizer,
    StreamingSpeechSynthesizer,
    Translator,
)
from ai_interpreter.domain.value_objects import LanguagePair

__all__ = [
    "InterpretationPipeline",
    "PipelineEvents",
    "PipelineStats",
    "UtteranceTiming",
]

logger = logging.getLogger(__name__)

# How long the worker waits for an utterance before re-checking the stop flag.
_QUEUE_POLL_SECONDS = 0.2

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class UtteranceTiming:
    """Wall-clock accounting for one interpreted utterance.

    Args:
        utterance_id: The utterance measured.
        audio_ms: Length of the spoken audio.
        stt_ms: Speech recognition time.
        mt_ms: Translation time (zero on a cache hit).
        tts_first_chunk_ms: Synthesis time of the first audio chunk.
        eou_to_first_audio_ms: End of utterance to first audio written to the
            sink - the headline latency number.
        total_ms: End of utterance to the last chunk written.
    """

    utterance_id: str
    audio_ms: float
    stt_ms: float
    mt_ms: float
    tts_first_chunk_ms: float
    eou_to_first_audio_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class PipelineStats:
    """Counters describing a pipeline run.

    Args:
        utterances_in: Utterances received from capture.
        utterances_out: Utterances fully interpreted.
        dropped_backpressure: Utterances discarded because the queue was full.
        dropped_empty: Utterances whose transcript or translation was empty.
        failures: Utterances dropped after retries were exhausted.
        barge_ins: Times playing audio was cleared by new speech.
        code_switch_reroutes: Utterances rescued by the English fallback
            because their "Tamil" transcript was phonotactically English.
        timings: Per-utterance latency records, in completion order.
    """

    utterances_in: int
    utterances_out: int
    dropped_backpressure: int
    dropped_empty: int
    failures: int
    barge_ins: int
    code_switch_reroutes: int
    timings: tuple[UtteranceTiming, ...]


@dataclass(slots=True)
class PipelineEvents:
    """Observer callbacks, all optional and all invoked on worker threads.

    The UI marshals these onto its own thread; nothing here may block for
    long or the pipeline stalls.

    Args:
        on_transcript: Final transcript for an utterance.
        on_translation: Translation for an utterance.
        on_timing: Latency record when an utterance completes.
        on_error: A stage failed for an utterance (after retries).
        on_state: Capture state changes (speaking / silence).
    """

    on_transcript: Callable[[Transcript], None] | None = None
    on_translation: Callable[[Translation], None] | None = None
    on_timing: Callable[[UtteranceTiming], None] | None = None
    on_error: Callable[[str, Exception], None] | None = None
    on_state: Callable[[SegmenterState], None] | None = None


@dataclass(slots=True)
class _Counters:
    """Mutable counters behind :class:`PipelineStats`."""

    utterances_in: int = 0
    utterances_out: int = 0
    dropped_backpressure: int = 0
    dropped_empty: int = 0
    failures: int = 0
    barge_ins: int = 0
    code_switch_reroutes: int = 0
    timings: list[UtteranceTiming] = field(default_factory=list)


class InterpretationPipeline:
    """Runs live interpretation over injected components.

    Everything arrives through ports, so the whole pipeline is testable with
    fakes and knows nothing about which models or devices are behind it.

    Args:
        capture: The capture session producing utterances. The pipeline takes
            over its ``on_utterance`` and ``on_state_change`` callbacks.
        recognizer: Speech-to-text for the source language (in practice the
            per-language router).
        translator: Text translation for ``pair``.
        synthesizer: Voice for the target language, or ``None`` for
            captions-only operation - the designed fallback when the target
            language has no voice.
        sink: Audio output, or ``None`` for captions-only.
        pair: Direction being interpreted.
        events: Observer callbacks.
        glossary: Term-recovery rewriter applied to transcripts before
            translation, or ``None``. This is where code-switched English and
            technical terms mangled by the Tamil-only recogniser are repaired.
        english_fallback: Recogniser for the *target* language, used to
            rescue utterances the source-language model rendered as phonetic
            soup - an English sentence spoken into the Tamil-only model. Only
            consulted when the transcript is phonotactically English-heavy;
            its text then bypasses translation entirely.
        fallback_min_score: Flagged-word fraction (with at least two flagged
            words) that triggers the rescue; half the utterance flagged
            always triggers.
        queue_maxsize: Utterances buffered before drop-oldest engages.
        max_retries: Per-stage retries before an utterance is dropped.
        retry_backoff_s: Pause before a retry.
    """

    def __init__(
        self,
        capture: CaptureSession,
        recognizer: SpeechRecognizer,
        translator: Translator,
        synthesizer: SpeechSynthesizer | None,
        sink: AudioSink | None,
        pair: LanguagePair,
        events: PipelineEvents | None = None,
        glossary: GlossaryRewriter | None = None,
        english_fallback: SpeechRecognizer | None = None,
        fallback_min_score: float = 0.3,
        queue_maxsize: int = 2,
        max_retries: int = 1,
        retry_backoff_s: float = 0.25,
    ) -> None:
        self._capture = capture
        self._recognizer = recognizer
        self._translator = translator
        self._synthesizer = synthesizer
        self._sink = sink
        self._pair = pair
        self._events = events or PipelineEvents()
        self._glossary = glossary
        self._english_fallback = english_fallback
        self._fallback_min_score = fallback_min_score
        self._queue_maxsize = queue_maxsize
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s

        self._queue: deque[tuple[Utterance, float]] = deque()
        self._queue_ready = threading.Event()
        self._lock = threading.Lock()
        self._counters = _Counters()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

        # Wire ourselves into the capture session.
        capture._on_utterance = self.submit_utterance
        capture._on_state_change = self._on_capture_state

    # -- lifecycle ---------------------------------------------------------
    @property
    def is_running(self) -> bool:
        """Whether the worker is alive and capture is running."""
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        """Open the sink, start the worker, then start capture. Idempotent.

        Order matters: audio output must be ready before the first utterance
        can possibly complete, and capture starts last so nothing arrives
        into a half-built pipeline.
        """
        if self.is_running:
            return
        self._stop.clear()
        if self._sink is not None:
            self._sink.open()
        self._worker = threading.Thread(target=self._run, name="interpretation", daemon=True)
        self._worker.start()
        self._capture.start()
        logger.info("Interpretation pipeline started (%s)", self._pair)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop capture, drain the worker, close the sink. Idempotent.

        Args:
            timeout: Seconds to wait for the worker.
        """
        self._capture.stop()
        self._stop.set()
        self._queue_ready.set()

        worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
            if worker.is_alive():
                logger.warning("Pipeline worker did not stop within %.1f s", timeout)

        if self._sink is not None:
            # Let audio already queued in the sink finish playing; closing
            # immediately would cut off the final translation mid-word. The
            # wait is sized to the audio actually pending (when the sink can
            # report it), not to the worker-join timeout - a lesson from the
            # first end-to-end run, where seconds of translated speech were
            # silently discarded by a flush that timed out too early.
            pending_ms = float(getattr(self._sink, "pending_ms", 0.0))
            self._sink.flush(timeout=max(timeout, pending_ms / 1000.0 + 2.0))
            self._sink.close()
        logger.info("Interpretation pipeline stopped")

    def stats(self) -> PipelineStats:
        """Return a snapshot of the run so far.

        Returns:
            Counters and per-utterance timings.
        """
        with self._lock:
            return PipelineStats(
                utterances_in=self._counters.utterances_in,
                utterances_out=self._counters.utterances_out,
                dropped_backpressure=self._counters.dropped_backpressure,
                dropped_empty=self._counters.dropped_empty,
                failures=self._counters.failures,
                barge_ins=self._counters.barge_ins,
                code_switch_reroutes=self._counters.code_switch_reroutes,
                timings=tuple(self._counters.timings),
            )

    # -- capture-side entry points ----------------------------------------
    def submit_utterance(self, utterance: Utterance) -> None:
        """Queue an utterance for interpretation.

        Called by the capture worker; also the test entry point.

        Args:
            utterance: The finished utterance.
        """
        with self._lock:
            self._counters.utterances_in += 1
            self._queue.append((utterance, time.perf_counter()))
            while len(self._queue) > self._queue_maxsize:
                dropped, _ = self._queue.popleft()
                self._counters.dropped_backpressure += 1
                logger.warning(
                    "Dropping utterance %s (%.1f s of speech): the pipeline cannot keep up",
                    dropped.id,
                    dropped.duration_ms / 1000.0,
                )
        self._queue_ready.set()

    def _on_capture_state(self, state: SegmenterState) -> None:
        """React to the speaker starting or stopping.

        Args:
            state: New segmenter state.
        """
        # Barge-in: the speaker talks, the interpreter shuts up. Audio is
        # cleared; the in-flight utterance's text still completes.
        if state is SegmenterState.SPEECH and self._sink is not None and self._sink.is_open:
            self._sink.clear()
            with self._lock:
                self._counters.barge_ins += 1
        if self._events.on_state is not None:
            self._events.on_state(state)

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        """Worker body: pull utterances and interpret them, until stopped."""
        while not self._stop.is_set():
            item = self._next_utterance()
            if item is None:
                continue
            utterance, submitted_at = item
            try:
                self._interpret(utterance, submitted_at)
            except Exception as exc:
                logger.exception("Unexpected pipeline failure for %s", utterance.id)
                with self._lock:
                    self._counters.failures += 1
                if self._events.on_error is not None:
                    self._events.on_error("pipeline", exc)

    def _next_utterance(self) -> tuple[Utterance, float] | None:
        """Take the next queued utterance, waiting briefly.

        Returns:
            The utterance and its submission time, or ``None`` on timeout or
            shutdown.
        """
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            self._queue_ready.clear()
        self._queue_ready.wait(_QUEUE_POLL_SECONDS)
        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    def _interpret(self, utterance: Utterance, submitted_at: float) -> None:
        """Run one utterance through STT, MT and TTS.

        Args:
            utterance: The utterance to interpret.
            submitted_at: ``perf_counter`` when it left the segmenter -
                the end-of-utterance reference for latency accounting.
        """
        transcript = self._with_retries("stt", lambda: self._recognizer.transcribe(utterance))
        if transcript is None:
            return
        stt_done = time.perf_counter()

        # Glossary repair happens before the transcript event, so the caption
        # the user sees matches what the translator receives.
        if self._glossary is not None and not transcript.is_empty:
            rewritten = self._glossary.rewrite(transcript.text)
            if rewritten != transcript.text:
                logger.debug("Glossary rewrote transcript for %s", utterance.id)
                transcript = replace(transcript, text=rewritten)

        # Code-switch rescue: an English sentence spoken into the Tamil-only
        # model comes out as phonetic soup. When the transcript is
        # phonotactically English-heavy, re-recognise with the target-language
        # model and use that text directly - no translation needed.
        direct_translation: Translation | None = None
        if (
            self._english_fallback is not None
            and not transcript.is_empty
            and self._pair.target.code == "en"
        ):
            flags = flag_english_tokens(transcript.text)
            score = english_phonetic_score(transcript.text)
            if score >= 0.5 or (len(flags) >= 2 and score >= self._fallback_min_score):
                logger.debug(
                    "Code-switch rescue for %s (score %.2f, flags %d)",
                    utterance.id,
                    score,
                    len(flags),
                )
                english = self._with_retries(
                    "stt-fallback",
                    lambda: self._english_fallback.transcribe(  # type: ignore[union-attr]
                        replace(utterance, language=self._pair.target)
                    ),
                )
                if english is not None and not english.is_empty:
                    with self._lock:
                        self._counters.code_switch_reroutes += 1
                    direct_translation = Translation(
                        utterance_id=utterance.id,
                        source_text=transcript.text,
                        translated_text=english.text,
                        pair=self._pair,
                        model_id=f"{english.model_id}+direct",
                    )
                    transcript = english

        if self._events.on_transcript is not None:
            self._events.on_transcript(transcript)
        if transcript.is_empty:
            with self._lock:
                self._counters.dropped_empty += 1
            return

        translation = direct_translation or self._with_retries(
            "mt", lambda: self._translator.translate(transcript.text, self._pair)
        )
        if translation is None:
            return
        mt_done = time.perf_counter()
        if self._events.on_translation is not None:
            self._events.on_translation(translation)
        if translation.is_empty:
            with self._lock:
                self._counters.dropped_empty += 1
            return

        first_audio_at: float | None = None
        tts_first_ms = 0.0
        if self._synthesizer is not None and self._sink is not None:
            first_audio_at, tts_first_ms = self._speak(translation)
            if first_audio_at is None and tts_first_ms == 0.0:
                return  # synthesis failed after retries; error already sent

        done = time.perf_counter()
        timing = UtteranceTiming(
            utterance_id=str(utterance.id),
            audio_ms=utterance.duration_ms,
            stt_ms=(stt_done - submitted_at) * 1000.0,
            mt_ms=(mt_done - stt_done) * 1000.0,
            tts_first_chunk_ms=tts_first_ms,
            eou_to_first_audio_ms=(((first_audio_at or done) - submitted_at) * 1000.0),
            total_ms=(done - submitted_at) * 1000.0,
        )
        with self._lock:
            self._counters.utterances_out += 1
            self._counters.timings.append(timing)
        if self._events.on_timing is not None:
            self._events.on_timing(timing)

    def _speak(self, translation: Translation) -> tuple[float | None, float]:
        """Synthesise a translation into the sink, chunk by chunk.

        Args:
            translation: Text to speak.

        Returns:
            The ``perf_counter`` of the first chunk written (or ``None``) and
            the synthesis time of the first chunk in milliseconds. ``(None,
            0.0)`` signals failure after retries.
        """
        synthesizer = self._synthesizer
        sink = self._sink
        assert synthesizer is not None and sink is not None

        def synthesise_all() -> tuple[float | None, float]:
            first_at: float | None = None
            first_ms = 0.0
            if isinstance(synthesizer, StreamingSpeechSynthesizer):
                chunk_iter = synthesizer.synthesize_stream(
                    translation.translated_text, self._pair.target
                )
            else:
                chunk_iter = iter(
                    [synthesizer.synthesize(translation.translated_text, self._pair.target)]
                )
            for chunk in chunk_iter:
                sink.write(chunk)
                if first_at is None and chunk.pcm.size:
                    first_at = time.perf_counter()
                    first_ms = chunk.latency_ms
            return first_at, first_ms

        result = self._with_retries("tts", synthesise_all)
        return result if result is not None else (None, 0.0)

    def _with_retries(self, stage: str, operation: Callable[[], _T]) -> _T | None:
        """Run a stage with bounded retries.

        Args:
            stage: Stage name for logging and error events.
            operation: The work to attempt.

        Returns:
            The operation's result, or ``None`` after retries are exhausted
            (the error event has been sent and the failure counted).
        """
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            if self._stop.is_set():
                return None
            try:
                return operation()
            except InterpreterError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    logger.warning(
                        "%s failed (attempt %d/%d): %s - retrying",
                        stage,
                        attempt + 1,
                        attempts,
                        exc,
                    )
                    time.sleep(self._retry_backoff_s)

        assert last_error is not None
        logger.error("%s failed after %d attempt(s): %s", stage, attempts, last_error)
        with self._lock:
            self._counters.failures += 1
        if self._events.on_error is not None:
            self._events.on_error(stage, last_error)
        return None
