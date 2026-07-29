"""Assembling the full interpretation pipeline from the container.

Extracted from the ``--interpret`` command the moment a second consumer
appeared (the Phase 8 desktop UI): both front ends must build *exactly* the
same pipeline - same captions-only fallback, same code-switch fallback with
word timestamps, same glossary wiring - or the UI would quietly behave
differently from the CLI that validated everything.

The module lives in ``app`` beside the container because it *composes*:
it decides which concrete pieces exist and how they plug together, which is
the composition root's job, not the application layer's.

Progress reporting is a callback rather than printing, so the CLI renders
rows and the UI renders status lines from the same events.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ai_interpreter.app.container import Container
from ai_interpreter.application.pipeline.interpretation import (
    InterpretationPipeline,
    PipelineEvents,
)
from ai_interpreter.application.services.capture_session import CaptureSession
from ai_interpreter.application.services.glossary import GlossaryRewriter
from ai_interpreter.domain.errors import ConfigurationError, InterpreterError
from ai_interpreter.domain.ports import (
    AudioSink,
    SpeechRecognizer,
    SpeechSynthesizer,
    Translator,
)
from ai_interpreter.domain.value_objects import LanguagePair, SampleRate
from ai_interpreter.infrastructure.audio.capture.microphone import MicrophoneSource
from ai_interpreter.infrastructure.audio.capture.wav_file import WavFileSource
from ai_interpreter.infrastructure.stt.sherpa_nemo import SherpaNemoCtcRecognizer

__all__ = ["InterpretationBundle", "ProgressCallback", "build_interpretation_bundle"]

logger = logging.getLogger(__name__)

# Every speech model in this project expects 16 kHz mono.
_MODEL_RATE: Final[int] = 16000

ProgressCallback = Callable[[str, str], None]
"""Called with a (label, detail) pair as each component becomes ready."""


@dataclass
class InterpretationBundle:
    """Everything a front end needs to run and tear down one session.

    Args:
        pipeline: The assembled pipeline, not yet started.
        capture: Its capture session (owned by the pipeline's callbacks).
        recognizer: Source-language speech recognition.
        translator: Text translation.
        synthesizer: Target-language voice, or ``None`` in captions-only mode.
        sink: Audio output, or ``None`` in captions-only mode.
        english_fallback: The code-switch rescue recogniser, if configured.
        captions_only: Whether the session runs without audio output.
    """

    pipeline: InterpretationPipeline
    capture: CaptureSession
    recognizer: SpeechRecognizer
    translator: Translator
    synthesizer: SpeechSynthesizer | None
    sink: AudioSink | None
    english_fallback: SpeechRecognizer | None
    captions_only: bool

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the session. Safe to call twice.

        Deliberately does NOT close the models: they are owned by the
        container's component cache and stay warm, so the next session
        starts in well under a second instead of paying ~10 s of loading.
        ``Container.shutdown()`` releases them at process exit.

        Args:
            timeout: Seconds to wait for the pipeline worker.
        """
        self.pipeline.stop(timeout=timeout)


def build_interpretation_bundle(
    container: Container,
    pair: LanguagePair,
    events: PipelineEvents,
    *,
    input_device: str | None = None,
    output_device: str | None = None,
    wav: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> InterpretationBundle:
    """Build and warm the complete pipeline for one direction.

    Models are warmed here, during assembly, so the first spoken sentence
    pays no initialisation cost - the same decision every CLI command made.

    Args:
        container: Built application container.
        pair: Direction to interpret.
        events: Observer callbacks (the CLI's printers or the UI's bridge).
        input_device: Microphone name fragment, or ``None`` for configured.
        output_device: Output name fragment (``"CABLE Input"`` for the
            virtual microphone), or ``None`` for configured.
        wav: Replay this recording instead of opening a microphone.
        on_progress: Receives a (label, detail) line per component.

    Returns:
        The assembled, warmed, not-yet-started bundle.

    Raises:
        InterpreterError: If a required component cannot be built. Optional
            components (TTS voice, code-switch fallback) degrade instead.
    """

    def progress(label: str, detail: str) -> None:
        if on_progress is not None:
            on_progress(label, detail)

    settings = container.settings

    if wav is not None:
        # realtime=True: the file must behave like a microphone, or every
        # latency number in the summary would be inflated by queue waits.
        audio_source: WavFileSource | MicrophoneSource = WavFileSource(
            wav,
            frame_ms=settings.audio.input.frame_ms,
            realtime=True,
        )
        progress("Input", f"WAV file: {wav.name} (real-time paced)")
    else:
        microphone = container.resolve_input_device(input_device)
        audio_source = container.create_microphone_source(microphone)
        progress("Input", f"{microphone.name}  [{microphone.host_api}]")

    started = time.perf_counter()
    recognizer = container.create_recognizer(pair.source)
    recognizer.warmup()
    progress(
        "STT",
        f"{container.describe_model_for(pair.source)} "
        f"({(time.perf_counter() - started) * 1000.0:.0f} ms warmup)",
    )

    started = time.perf_counter()
    translator = container.create_translator(pair)
    translator.warmup()
    progress(
        "MT",
        f"{translator.model_id} ({(time.perf_counter() - started) * 1000.0:.0f} ms warmup)",
    )

    synthesizer: SpeechSynthesizer | None
    sink: AudioSink | None
    captions_only = False
    try:
        started = time.perf_counter()
        synthesizer = container.create_synthesizer(pair.target)
        synthesizer.warmup()
        progress(
            "TTS",
            f"{synthesizer.provider_id} ({(time.perf_counter() - started) * 1000.0:.0f} ms warmup)",
        )
        sink = container.create_audio_sink(device_name=output_device)
        progress("Output", f"{sink.device.name}  [{sink.device.host_api}]")
        if sink.device.is_virtual_cable:
            progress("", "-> select 'CABLE Output' as the mic in Teams/Zoom/Meet")
    except ConfigurationError as exc:
        # The designed fallback: no voice for the target language means
        # captions on screen instead of audio.
        logger.warning("Captions-only mode: %s", exc)
        progress("TTS", "none - captions only")
        synthesizer = None
        sink = None
        captions_only = True

    vad = container.create_vad()
    preprocessor = container.create_preprocessor(audio_source.sample_rate)
    segmenter = container.create_segmenter(SampleRate(_MODEL_RATE), language=pair.source)
    capture = CaptureSession(
        source=audio_source,
        preprocessor=preprocessor,
        vad=vad,
        segmenter=segmenter,
    )

    glossary_terms = settings.translation.glossary
    glossary = GlossaryRewriter(glossary_terms) if glossary_terms else None
    if glossary is not None:
        progress("Glossary", f"{glossary.size} term variant(s) active")

    english_fallback: SpeechRecognizer | None = None
    if settings.stt.code_switch_fallback and pair.target.code == "en" and pair.source.code != "en":
        try:
            # Word timestamps regardless of the global setting: transcript
            # fusion splices by time, and without per-word timings it would
            # quietly decline on every mixed sentence.
            english_fallback = container.create_recognizer(pair.target, word_timestamps=True)
            english_fallback.warmup()
            fusion = "word fusion for mixed sentences, reroute for English ones"
            if not settings.stt.word_fusion:
                fusion = "English-heavy utterances are re-recognised in English"
            progress(
                "Code-switch rescue",
                f"{container.describe_model_for(pair.target)} - {fusion}",
            )
        except InterpreterError as exc:
            logger.warning("Code-switch fallback unavailable: %s", exc)
            english_fallback = None

    # Streaming only pays on a linear-cost chunked recogniser: words commit
    # while the speaker talks and only the tail decodes at end-of-utterance.
    # On Whisper the same interface costs full encoder passes for zero
    # latency gain, so the offline path stays.
    streaming_stt = settings.stt.streaming and isinstance(recognizer, SherpaNemoCtcRecognizer)
    if streaming_stt:
        progress("Streaming", "chunked decode during speech; only the tail after you stop")

    pipeline = InterpretationPipeline(
        capture=capture,
        recognizer=recognizer,
        translator=translator,
        synthesizer=synthesizer,
        sink=sink,
        pair=pair,
        glossary=glossary,
        english_fallback=english_fallback,
        fallback_min_score=settings.stt.code_switch_min_score,
        word_fusion=settings.stt.word_fusion,
        streaming_stt=streaming_stt,
        # Chunked commitment cannot commit anything before chunk+margin of
        # audio exists, so shorter speech gains nothing from streaming and
        # takes the offline path (0.8 s is the recogniser's fixed margin).
        stream_min_seconds=settings.stt.chunk_ms / 1000.0 + 0.8,
        events=events,
        queue_maxsize=settings.pipeline.queue_maxsize,
        max_retries=settings.pipeline.max_retries,
        retry_backoff_s=settings.pipeline.retry_backoff_ms / 1000.0,
    )

    return InterpretationBundle(
        pipeline=pipeline,
        capture=capture,
        recognizer=recognizer,
        translator=translator,
        synthesizer=synthesizer,
        sink=sink,
        english_fallback=english_fallback,
        captions_only=captions_only,
    )
