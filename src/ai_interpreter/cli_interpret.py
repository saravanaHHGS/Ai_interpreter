"""The live interpretation command.

``--interpret`` runs the complete pipeline: microphone (or a WAV file) in,
translated speech out of the configured output device. With
``--out "CABLE Input"`` the output is the virtual microphone and a meeting
application hears the translation; with the default output it plays on the
speakers, which is the way to *hear* what the interpreter would say.

``--wav FILE`` replaces the microphone with a recording - the reproducible
path: the same input produces comparable timings run after run, which is how
the pipeline's end-to-end latency is measured honestly.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Final

from ai_interpreter.app.container import Container
from ai_interpreter.application.pipeline.interpretation import (
    InterpretationPipeline,
    PipelineEvents,
    UtteranceTiming,
)
from ai_interpreter.application.services.capture_session import CaptureSession
from ai_interpreter.domain.entities import Transcript, Translation
from ai_interpreter.domain.errors import ConfigurationError, InterpreterError
from ai_interpreter.domain.ports import AudioSink, SpeechSynthesizer
from ai_interpreter.domain.value_objects import LanguagePair, SampleRate
from ai_interpreter.infrastructure.audio.capture.wav_file import WavFileSource
from ai_interpreter.presentation.console import WIDTH, heading, row

__all__ = ["run_interpret"]

logger = logging.getLogger(__name__)

_EXIT_OK: Final[int] = 0
_EXIT_ERROR: Final[int] = 1

_MODEL_RATE: Final[int] = 16000


def run_interpret(
    container: Container,
    seconds: float,
    device: str | None,
    out_device: str | None,
    wav: Path | None,
    source: str | None,
    target: str | None,
) -> int:
    """Run the live pipeline.

    Args:
        container: Built application container.
        seconds: How long to run (ignored for a WAV source, which runs to its
            end).
        device: Microphone name fragment, or ``None`` for configured/default.
        out_device: Output device fragment; ``"CABLE Input"`` for the virtual
            microphone, ``None`` for the configured output.
        wav: Recording to replay instead of the microphone.
        source: Source language override.
        target: Target language override.

    Returns:
        Process exit code.
    """
    print("=" * WIDTH)
    print("  Live interpretation")
    print("=" * WIDTH)

    configured = container.settings.app.language_pair
    try:
        pair = LanguagePair.of(source or configured.source, target or configured.target)
    except ValueError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    heading("Loading (models warm up once, before any audio flows)")
    row("Direction", str(pair))

    try:
        if wav is not None:
            # realtime=True: the file must behave like a microphone, or every
            # latency number in the summary would be inflated by queue waits.
            audio_source: object = WavFileSource(
                wav,
                frame_ms=container.settings.audio.input.frame_ms,
                realtime=True,
            )
            row("Input", f"WAV file: {wav.name} (real-time paced)")
        else:
            mic = container.resolve_input_device(device)
            audio_source = container.create_microphone_source(mic)
            row("Input", f"{mic.name}  [{mic.host_api}]")

        started = time.perf_counter()
        recognizer = container.create_recognizer(pair.source)
        recognizer.warmup()
        row(
            "STT",
            f"{container.describe_model_for(pair.source)} "
            f"({(time.perf_counter() - started) * 1000.0:.0f} ms warmup)",
        )

        started = time.perf_counter()
        translator = container.create_translator(pair)
        translator.warmup()
        row(
            "MT",
            f"{translator.model_id} ({(time.perf_counter() - started) * 1000.0:.0f} ms warmup)",
        )

        synthesizer: SpeechSynthesizer | None
        sink: AudioSink | None
        try:
            started = time.perf_counter()
            synthesizer = container.create_synthesizer(pair.target)
            synthesizer.warmup()
            row(
                "TTS",
                f"{synthesizer.provider_id} "
                f"({(time.perf_counter() - started) * 1000.0:.0f} ms warmup)",
            )
            sink = container.create_audio_sink(device_name=out_device)
            row("Output", f"{sink.device.name}  [{sink.device.host_api}]")
            if sink.device.is_virtual_cable:
                row("", "-> select 'CABLE Output' as the mic in Teams/Zoom/Meet")
        except ConfigurationError as exc:
            # The designed fallback: no voice for the target language means
            # captions on screen instead of audio.
            logger.warning("Captions-only mode: %s", exc)
            row("TTS", "none - captions only")
            synthesizer = None
            sink = None

        vad = container.create_vad()
        preprocessor = container.create_preprocessor(audio_source.sample_rate)  # type: ignore[attr-defined]
        segmenter = container.create_segmenter(SampleRate(_MODEL_RATE), language=pair.source)
        capture = CaptureSession(
            source=audio_source,  # type: ignore[arg-type]
            preprocessor=preprocessor,
            vad=vad,
            segmenter=segmenter,
        )
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    def on_transcript(transcript: Transcript) -> None:
        if not transcript.is_empty:
            print(f"\n  [{pair.source.code}] {transcript.text}")

    def on_translation(translation: Translation) -> None:
        cached = "  (cached)" if translation.from_cache else ""
        print(f"  [{pair.target.code}] {translation.translated_text}{cached}")

    def on_timing(timing: UtteranceTiming) -> None:
        print(
            f"       EOU->audio {timing.eou_to_first_audio_ms:5.0f} ms   "
            f"(stt {timing.stt_ms:.0f} | mt {timing.mt_ms:.0f} | "
            f"tts {timing.tts_first_chunk_ms:.0f})"
        )

    def on_error(stage: str, exc: Exception) -> None:
        print(f"  [error in {stage}] {exc}", file=sys.stderr)

    pipeline = InterpretationPipeline(
        capture=capture,
        recognizer=recognizer,
        translator=translator,
        synthesizer=synthesizer,
        sink=sink,
        pair=pair,
        events=PipelineEvents(
            on_transcript=on_transcript,
            on_translation=on_translation,
            on_timing=on_timing,
            on_error=on_error,
        ),
        queue_maxsize=container.settings.pipeline.queue_maxsize,
        max_retries=container.settings.pipeline.max_retries,
        retry_backoff_s=container.settings.pipeline.retry_backoff_ms / 1000.0,
    )

    heading("Interpreting" + ("" if wav is not None else f" for {seconds:.0f} seconds"))
    if wav is None:
        print("  Speak normally, pausing between sentences. Ctrl+C stops early.")

    try:
        with container.logging_service.quiet_console():
            pipeline.start()
            try:
                _wait(pipeline, capture, seconds, wav is not None)
            except KeyboardInterrupt:
                print("\n  Interrupted.")
    finally:
        pipeline.stop()
        recognizer.close()
        translator.close()
        if synthesizer is not None:
            synthesizer.close()

    return _summarise(pipeline, container)


def _wait(
    pipeline: InterpretationPipeline,
    capture: CaptureSession,
    seconds: float,
    finite_source: bool,
) -> None:
    """Block while the pipeline runs.

    Args:
        pipeline: The running pipeline.
        capture: Its capture session.
        seconds: Time limit for a live microphone.
        finite_source: Whether the source ends on its own (a WAV file). Then
            the wait continues until every submitted utterance is resolved.
    """
    if not finite_source:
        time.sleep(seconds)
        return

    while capture.is_running:
        time.sleep(0.1)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        stats = pipeline.stats()
        resolved = (
            stats.utterances_out + stats.dropped_backpressure + stats.dropped_empty + stats.failures
        )
        if resolved >= stats.utterances_in:
            break
        time.sleep(0.2)


def _summarise(pipeline: InterpretationPipeline, container: Container) -> int:
    """Print the run summary.

    Args:
        pipeline: The stopped pipeline.
        container: For configuration context in the report.

    Returns:
        Process exit code.
    """
    stats = pipeline.stats()
    heading("Summary")
    row("Utterances heard", str(stats.utterances_in))
    row("Interpreted", str(stats.utterances_out))
    row("Dropped (backpressure)", str(stats.dropped_backpressure))
    row("Dropped (empty)", str(stats.dropped_empty))
    row("Failures", str(stats.failures))
    row("Barge-ins", str(stats.barge_ins))

    if stats.timings:
        latencies = sorted(t.eou_to_first_audio_ms for t in stats.timings)
        mean = sum(latencies) / len(latencies)
        endpoint = container.settings.vad.min_silence_ms
        print()
        row("EOU->first audio (mean)", f"{mean:.0f} ms")
        row("EOU->first audio (best)", f"{latencies[0]:.0f} ms")
        row("EOU->first audio (worst)", f"{latencies[-1]:.0f} ms")
        row(
            "Perceived (incl. endpoint)",
            f"{endpoint + mean:.0f} ms after you stop speaking",
        )

    return _EXIT_OK if stats.failures == 0 else _EXIT_ERROR
