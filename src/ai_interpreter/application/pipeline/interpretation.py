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
    has_native_anchor,
)
from ai_interpreter.application.services.glossary import GlossaryRewriter
from ai_interpreter.application.services.streaming_transcriber import UtteranceStreamer
from ai_interpreter.application.services.transcript_fusion import fuse_transcripts
from ai_interpreter.application.services.utterance_segmenter import SegmenterState
from ai_interpreter.domain.entities import AudioFrame, Transcript, Translation, Utterance
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.domain.ports import (
    AudioSink,
    SpeechRecognizer,
    SpeechSynthesizer,
    StreamingSpeechRecognizer,
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

# Pre-roll frames kept for the streaming lane. The segmenter keeps up to
# 300 ms of audio from before speech onset; at 32 ms per VAD frame, twelve
# frames cover it with margin. The streamed decode must see the same leading
# audio the offline decode would, or first syllables go missing.
_STREAM_PREROLL_FRAMES = 12

# Ceiling on waiting for a streamed final transcript: the largest possible
# uncommitted tail (max_buffer_seconds, 12 s) at the measured RTF, doubled.
_STREAM_RESULT_TIMEOUT_SECONDS = 15.0

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
        word_fusions: Mixed utterances whose transliterated English words
            were replaced by the English recogniser's words, time-aligned.
        timings: Per-utterance latency records, in completion order.
    """

    utterances_in: int
    utterances_out: int
    dropped_backpressure: int
    dropped_empty: int
    failures: int
    barge_ins: int
    code_switch_reroutes: int
    word_fusions: int
    timings: tuple[UtteranceTiming, ...]


@dataclass(slots=True)
class PipelineEvents:
    """Observer callbacks, all optional and all invoked on worker threads.

    The UI marshals these onto its own thread; nothing here may block for
    long or the pipeline stalls.

    Args:
        on_transcript: Final transcript for an utterance.
        on_partial: Interim transcript while the speaker is still talking
            (streaming lane only). Text grows as words are committed.
        on_translation: Translation for an utterance.
        on_timing: Latency record when an utterance completes.
        on_error: A stage failed for an utterance (after retries).
        on_state: Capture state changes (speaking / silence).
    """

    on_transcript: Callable[[Transcript], None] | None = None
    on_partial: Callable[[Transcript], None] | None = None
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
    word_fusions: int = 0
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
        english_fallback: Recogniser for the *target* language, consulted
            when the source transcript carries phonotactically English words.
            Two repairs come from it, chosen per utterance: a transcript
            that is English *throughout* (no native anchor word) is replaced
            wholesale and bypasses translation; a *mixed* transcript keeps
            its Tamil and has only the flagged words replaced by the English
            recogniser's words from the same time window (word fusion).
        fallback_min_score: Flagged-word fraction (with at least two flagged
            words) that triggers the wholesale reroute; half the utterance
            flagged always triggers.
        word_fusion: Whether mixed transcripts are repaired word-by-word.
            Requires ``english_fallback`` and word timestamps on both
            recognisers; without them fusion quietly declines per utterance.
        streaming_stt: Feed frames into the recogniser *while the speaker is
            talking* instead of decoding after end-of-utterance. Only worth
            enabling for linear-cost chunked recognisers (the IndicConformer);
            the assembly decides. Any streaming failure falls back to the
            offline decode, so this can never lose an utterance.
        stream_min_seconds: Speech must last this long before the streaming
            lane engages. An utterance shorter than the recogniser's
            chunk-plus-margin can never commit early, so streaming it buys
            nothing - and measured on the WAV regression it *cost* time,
            because the streamed decode contends with the previous
            utterance's translation on 2 cores. Short sentences therefore
            take the offline path exactly as before.
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
        word_fusion: bool = True,
        streaming_stt: bool = False,
        stream_min_seconds: float = 2.8,
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
        self._word_fusion = word_fusion
        self._queue_maxsize = queue_maxsize
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s

        self._queue: deque[tuple[Utterance, float]] = deque()
        self._queue_ready = threading.Event()
        self._lock = threading.Lock()
        self._counters = _Counters()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

        # Streaming lane state, touched only on the capture thread (frames,
        # state changes and utterance submission all arrive from it).
        self._streaming_stt = streaming_stt and isinstance(recognizer, StreamingSpeechRecognizer)
        self._stream_min_ms = stream_min_seconds * 1000.0
        self._streamer: UtteranceStreamer | None = None
        self._stream_ring: deque[AudioFrame] = deque(maxlen=_STREAM_PREROLL_FRAMES)
        self._speech_buffer: list[AudioFrame] = []
        self._speech_ms = 0.0
        self._in_speech = False
        self._streams: dict[str, UtteranceStreamer] = {}

        # Wire ourselves into the capture session.
        capture._on_utterance = self.submit_utterance
        capture._on_state_change = self._on_capture_state
        if self._streaming_stt:
            capture._on_frame = self._on_stream_frame

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
        # A stream still open mid-speech must be released or its thread
        # waits forever for frames that will never come.
        streamer, self._streamer = self._streamer, None
        if streamer is not None:
            streamer.finish()
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
                word_fusions=self._counters.word_fusions,
                timings=tuple(self._counters.timings),
            )

    # -- capture-side entry points ----------------------------------------
    def submit_utterance(self, utterance: Utterance) -> None:
        """Queue an utterance for interpretation.

        Called by the capture worker; also the test entry point.

        Args:
            utterance: The finished utterance.
        """
        # Hand the live stream over to this utterance: end-of-utterance means
        # the streamer can decode its tail while the worker gets scheduled.
        # Short utterances never opened one and decode offline.
        streamer, self._streamer = self._streamer, None
        self._in_speech = False
        self._speech_buffer.clear()
        self._speech_ms = 0.0
        if streamer is not None:
            streamer.finish()
            self._streams[str(utterance.id)] = streamer
            while len(self._streams) > self._queue_maxsize + 1:
                # Utterances dropped by backpressure never collect theirs.
                self._streams.pop(next(iter(self._streams)))

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

        # Streaming lane: speech onset starts buffering (seeded with the
        # pre-roll ring so the decode sees the same leading audio the
        # offline path would); the stream itself only opens once speech has
        # lasted long enough for chunked commitment to pay.
        if self._streaming_stt:
            if state is SegmenterState.SPEECH and self._streamer is None:
                self._speech_buffer.extend(self._stream_ring)
                self._speech_ms = sum(frame.duration_ms for frame in self._speech_buffer)
                self._stream_ring.clear()
                self._in_speech = True
            elif state is SegmenterState.SILENCE:
                self._in_speech = False

        if self._events.on_state is not None:
            self._events.on_state(state)

    def _on_stream_frame(self, frame: AudioFrame, probability: float) -> None:
        """Route one capture frame into the streaming lane.

        Args:
            frame: Processed 16 kHz frame from capture.
            probability: Speech probability (unused; the segmenter decides).
        """
        if self._streamer is not None:
            self._streamer.push(frame)
            return
        if not self._in_speech:
            self._stream_ring.append(frame)
            return

        self._speech_buffer.append(frame)
        self._speech_ms += frame.duration_ms
        if self._speech_ms < self._stream_min_ms:
            return

        # Speech has outlasted the threshold: open the stream and hand it
        # the whole backlog. The decode starts mid-utterance with enough
        # audio buffered for its first committed chunk.
        streamer = UtteranceStreamer(
            self._recognizer,  # type: ignore[arg-type]
            self._pair.source,
            on_partial=self._events.on_partial,
        )
        for buffered in self._speech_buffer:
            streamer.push(buffered)
        self._speech_buffer.clear()
        self._speech_ms = 0.0
        self._streamer = streamer

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
        transcript = self._collect_streamed(utterance)
        if transcript is None:
            transcript = self._with_retries("stt", lambda: self._recognizer.transcribe(utterance))
        if transcript is None:
            return
        stt_done = time.perf_counter()

        # Model-level repair first: when the transcript carries flagged
        # English-in-Tamil-script words, consult the English recogniser.
        transcript, direct_translation = self._repair_code_switch(utterance, transcript)

        # Glossary repair is the LAST resort, after the models have had their
        # say - and it happens before the transcript event, so the caption
        # the user sees matches what the translator receives.
        if self._glossary is not None and not transcript.is_empty:
            rewritten = self._glossary.rewrite(transcript.text)
            if rewritten != transcript.text:
                logger.debug("Glossary rewrote transcript for %s", utterance.id)
                transcript = replace(transcript, text=rewritten, segments=())
                if direct_translation is not None:
                    # The direct path speaks the transcript verbatim; a
                    # glossary fix must reach the voice too.
                    direct_translation = replace(
                        direct_translation, translated_text=transcript.text
                    )

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

    def _collect_streamed(self, utterance: Utterance) -> Transcript | None:
        """Collect the streamed transcript for an utterance, if one exists.

        Args:
            utterance: The utterance being interpreted.

        Returns:
            The final streamed transcript stamped with the utterance's id,
            or ``None`` when no stream ran or it failed - the caller then
            decodes offline, so streaming can never lose an utterance.
        """
        streamer = self._streams.pop(str(utterance.id), None)
        if streamer is None:
            return None
        transcript = streamer.result(timeout=_STREAM_RESULT_TIMEOUT_SECONDS)
        if transcript is None or streamer.error is not None:
            return None
        return replace(transcript, utterance_id=utterance.id)

    def _repair_code_switch(
        self, utterance: Utterance, transcript: Transcript
    ) -> tuple[Transcript, Translation | None]:
        """Repair English words the source-language recogniser transliterated.

        Two tiers, selected by what the transcript actually is:

        **Wholesale reroute** - the transcript is English throughout (flagged
        words, and no long unflagged native word anchoring it as Tamil). The
        English recogniser's text replaces it entirely and translation is
        skipped: the speaker said an English sentence.

        **Word fusion** - the transcript is genuinely mixed (an anchor word
        proves the Tamil is real). The Tamil stays; only the flagged words
        are replaced by the English recogniser's words from the same time
        window. The result still goes through translation, which passes the
        spliced Latin words through untouched.

        Either way the English decode happens at most once per utterance.

        Args:
            utterance: The utterance, re-decoded by the fallback when needed.
            transcript: The source-language transcript.

        Returns:
            The (possibly repaired) transcript, and a ready-made translation
            when the wholesale reroute made translation unnecessary.
        """
        fallback = self._english_fallback
        if fallback is None or transcript.is_empty or self._pair.target.code != "en":
            return transcript, None

        flags = flag_english_tokens(transcript.text)
        if not flags:
            return transcript, None
        score = english_phonetic_score(transcript.text)
        heavily_english = score >= 0.5 or (len(flags) >= 2 and score >= self._fallback_min_score)
        # A native anchor word marks the sentence as genuinely mixed, which
        # makes fusion the right repair even at a high score - a wholesale
        # reroute would throw the real Tamil away. Without fusion available,
        # the score decides alone, as before.
        try_fusion = self._word_fusion and (
            has_native_anchor(transcript.text) or not heavily_english
        )
        if not try_fusion and not heavily_english:
            return transcript, None

        logger.debug(
            "Code-switch repair for %s (score %.2f, flags %d, fusion=%s)",
            utterance.id,
            score,
            len(flags),
            try_fusion,
        )
        english = self._with_retries(
            "stt-fallback",
            lambda: fallback.transcribe(replace(utterance, language=self._pair.target)),
        )
        if english is None or english.is_empty:
            return transcript, None

        if try_fusion:
            fused = fuse_transcripts(transcript, english)
            if fused is not None:
                with self._lock:
                    self._counters.word_fusions += 1
                logger.debug(
                    "Word fusion for %s replaced %d word(s) with %d",
                    utterance.id,
                    len(fused.replaced),
                    len(fused.inserted),
                )
                repaired = replace(
                    transcript,
                    text=fused.text,
                    segments=(),
                    model_id=f"{transcript.model_id}+{english.model_id}+fusion",
                )
                return repaired, None
            # Fusion declined (no word timestamps, or no English words in
            # any flagged window). An English-heavy sentence still deserves
            # the wholesale repair rather than translating soup.

        if not heavily_english:
            return transcript, None
        with self._lock:
            self._counters.code_switch_reroutes += 1
        direct = Translation(
            utterance_id=utterance.id,
            source_text=transcript.text,
            translated_text=english.text,
            pair=self._pair,
            model_id=f"{english.model_id}+direct",
        )
        return english, direct

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
