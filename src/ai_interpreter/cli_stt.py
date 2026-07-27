"""Speech-to-text commands: transcribe, listen and benchmark.

``--benchmark`` is the important one. Phase 1 estimated Whisper `small` at a
0.55 real-time factor on this class of CPU; the measured figure was roughly
eight times worse. Estimates about inference speed are worth very little, so
the benchmark is a first-class command rather than a one-off script.
"""

from __future__ import annotations

import logging
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Final

import numpy as np

from ai_interpreter.app.container import Container
from ai_interpreter.application.services.capture_session import CaptureSession
from ai_interpreter.domain.entities import Transcript, Utterance, UtteranceId
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.domain.value_objects import LanguageCode, SampleRate
from ai_interpreter.infrastructure.audio.capture.wav_file import WavFileSource
from ai_interpreter.infrastructure.audio.dsp import AudioPreprocessor
from ai_interpreter.infrastructure.stt.faster_whisper import (
    FasterWhisperRecognizer,
    WhisperDecodeOptions,
)
from ai_interpreter.presentation.console import WIDTH, heading, row

__all__ = ["run_benchmark", "run_listen", "run_transcribe"]

logger = logging.getLogger(__name__)

_EXIT_OK: Final[int] = 0
_EXIT_ERROR: Final[int] = 1

_MODEL_RATE: Final[int] = 16000

# Most recent recordings to search for a usable benchmark utterance.
_MAX_BENCHMARK_CANDIDATES: Final[int] = 12


def _load_utterances(
    container: Container,
    wav_path: Path,
    language: LanguageCode | None = None,
) -> list[Utterance]:
    """Segment a WAV file into utterances using the Phase 3 chain.

    Args:
        container: Built application container.
        wav_path: File to process.
        language: Language to tag utterances with, or ``None`` for the
            configured source language.

    Returns:
        Detected utterances, in order.

    Raises:
        InterpreterError: If the file cannot be read or the detector fails.
    """
    source = WavFileSource(wav_path, frame_ms=container.settings.audio.input.frame_ms)
    preprocessor = container.create_preprocessor(source.sample_rate)
    vad = container.create_vad()
    segmenter = container.create_segmenter(SampleRate(_MODEL_RATE), language=language)

    collected: list[Utterance] = []
    session = CaptureSession(
        source=source,
        preprocessor=preprocessor,
        vad=vad,
        segmenter=segmenter,
        on_utterance=collected.append,
    )

    session.start()
    while session.is_running:
        time.sleep(0.01)
    session.stop()

    if session.error is not None:
        raise InterpreterError(str(session.error))
    return collected


def _print_transcript(index: int, utterance: Utterance, transcript: Transcript) -> None:
    """Render one transcript with its timings and confidence.

    Args:
        index: Position in the list, from one.
        utterance: Utterance that was transcribed.
        transcript: Result.
    """
    print()
    print(
        f"  [{index}] {utterance.duration_ms / 1000.0:.2f} s of audio "
        f"at {utterance.started_at_ms / 1000.0:.2f} s"
    )
    row("    decode time", f"{transcript.latency_ms:.0f} ms")
    row("    language", f"{transcript.language.english_name} ({transcript.language.code})")
    row("    confidence", f"{transcript.confidence.value:.2f}")

    if transcript.is_empty:
        row("    text", "(nothing recognised)")
        return

    print(f"\n      {transcript.text}\n")
    for segment in transcript.segments:
        print(
            f"      [{segment.start_ms / 1000.0:5.2f}-{segment.end_ms / 1000.0:5.2f}] "
            f"conf {segment.confidence.value:.2f}  {segment.text}"
        )


def run_transcribe(container: Container, wav_path: Path, language: str | None) -> int:
    """Transcribe a WAV file and report timings.

    Args:
        container: Built application container.
        wav_path: File to transcribe.
        language: Language code override, or ``None`` for the configured one.

    Returns:
        Process exit code.
    """
    print("=" * WIDTH)
    print(f"  Transcribe: {wav_path.name}")
    print("=" * WIDTH)

    if not wav_path.is_file():
        print(f"\nFile not found: {wav_path}\n", file=sys.stderr)
        return _EXIT_ERROR

    stt = container.settings.stt
    chosen = LanguageCode(language) if language else None

    try:
        heading("Loading")
        row("Model", f"whisper-{stt.model} ({stt.compute_type}, {stt.device})")
        row("Threads", str(stt.cpu_threads))

        recognizer = container.create_recognizer(chosen)
        started = time.perf_counter()
        recognizer.warmup()
        row("Warmup", f"{(time.perf_counter() - started) * 1000.0:.0f} ms")
        row("Language", str(recognizer.language or "auto-detect"))

        utterances = _load_utterances(container, wav_path, chosen)
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    heading(f"Utterances detected: {len(utterances)}")
    if not utterances:
        print("  No speech was found in this file.")
        return _EXIT_OK

    try:
        for index, utterance in enumerate(utterances, start=1):
            transcript = recognizer.transcribe(utterance)
            _print_transcript(index, utterance, transcript)
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR
    finally:
        recognizer.close()

    heading("Summary")
    row("Utterances", str(len(utterances)))
    row("Mean decode time", f"{recognizer.mean_decode_ms:.0f} ms")
    return _EXIT_OK


def run_listen(container: Container, seconds: float, device_name: str | None) -> int:
    """Capture from the microphone and transcribe each utterance live.

    Args:
        container: Built application container.
        seconds: How long to listen.
        device_name: Input device name fragment, or ``None`` for configured.

    Returns:
        Process exit code.
    """
    print("=" * WIDTH)
    print(f"  Live transcription - {seconds:.0f} seconds")
    print("=" * WIDTH)

    stt = container.settings.stt

    try:
        device = container.resolve_input_device(device_name)
        heading("Configuration")
        row("Device", f"{device.name}  [{device.host_api}]")
        row("Model", f"whisper-{stt.model} ({stt.compute_type}, {stt.cpu_threads} threads)")

        print("\n  Loading model...")
        recognizer = container.create_recognizer()
        recognizer.warmup()
        row("Language", str(recognizer.language or "auto-detect"))

        source = container.create_microphone_source(device)
        preprocessor = container.create_preprocessor(
            SampleRate(container.settings.audio.input.sample_rate)
        )
        vad = container.create_vad()
        segmenter = container.create_segmenter(
            SampleRate(_MODEL_RATE), language=recognizer.language
        )
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    results: list[tuple[Utterance, Transcript]] = []
    lock = threading.Lock()

    def on_utterance(utterance: Utterance) -> None:
        # Runs on the capture worker thread. Decoding here blocks capture for
        # its duration, which is acceptable for this diagnostic command but is
        # exactly what the Phase 9 pipeline moves to a separate executor.
        try:
            transcript = recognizer.transcribe(utterance)
        except InterpreterError as exc:
            logger.error("Transcription failed: %s", exc)
            return
        with lock:
            results.append((utterance, transcript))
        text = transcript.text or "(nothing recognised)"
        print(
            f"\n  {utterance.duration_ms / 1000.0:5.2f} s "
            f"-> {transcript.latency_ms:6.0f} ms  "
            f"conf {transcript.confidence.value:.2f}  {text}"
        )

    session = CaptureSession(
        source=source,
        preprocessor=preprocessor,
        vad=vad,
        segmenter=segmenter,
        on_utterance=on_utterance,
    )

    heading("Listening")
    print("  Speak normally, pausing between sentences.")

    try:
        with container.logging_service.quiet_console():
            try:
                session.start()
                time.sleep(seconds)
            except KeyboardInterrupt:
                print("\n  Interrupted.")
            finally:
                session.stop()
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR
    finally:
        recognizer.close()

    heading("Summary")
    row("Utterances", str(len(results)))
    if results:
        latencies = [transcript.latency_ms for _, transcript in results]
        row("Mean decode time", f"{statistics.mean(latencies):.0f} ms")
        row("Slowest decode", f"{max(latencies):.0f} ms")
        row("End-of-speech delay", f"{container.settings.vad.min_silence_ms} ms")
        row(
            "Speech to text total",
            f"{container.settings.vad.min_silence_ms + statistics.mean(latencies):.0f} ms",
        )
    return _EXIT_OK


def run_benchmark(
    container: Container,
    wav_path: Path | None,
    repeats: int,
    language: str | None = None,
) -> int:
    """Measure decode time across thread counts, on a single utterance.

    A single utterance is the right unit: it is exactly what the pipeline
    decodes, and the resulting time is exactly what the user waits after
    finishing a sentence. Benchmarking a whole file would average speech with
    silence, and Whisper hallucinates through silence, which inflates the
    result with work that never happens in production.

    Args:
        container: Built application container.
        wav_path: Audio to benchmark with, or ``None`` to pick a recording.
        repeats: Timed runs per configuration; the fastest is reported, since
            slower runs only reflect interference from other processes.
        language: Language to decode, or ``None`` to auto-detect. Forcing the
            wrong language makes Whisper hallucinate and roughly doubles the
            measured time, which would be a measurement artefact rather than
            a property of the model.

    Returns:
        Process exit code.
    """
    print("=" * WIDTH)
    print("  Speech-to-text benchmark")
    print("=" * WIDTH)

    hardware = container.hardware
    heading("Machine")
    row("CPU", hardware.cpu_name)
    row("Cores", f"{hardware.physical_cores} physical / {hardware.logical_cores} logical")
    row("Profile", container.selection.profile.value)

    chosen = LanguageCode(language) if language else None
    audio, label = _benchmark_audio(container, wav_path, chosen)
    duration_s = audio.size / _MODEL_RATE
    row("Test utterance", f"{label} ({duration_s:.2f} s)")
    row("Language", str(chosen) if chosen else "auto-detect")

    stt = container.settings.stt
    thread_counts = sorted({1, hardware.physical_cores, hardware.logical_cores})
    models = [stt.model]

    heading("Results")
    print(f"  {'model':<10} {'threads':>8} {'decode':>10} {'vs audio':>10}   transcript")
    print("  " + "-" * (WIDTH - 4))

    utterance = Utterance(
        id=UtteranceId("benchmark"),
        pcm=audio,
        sample_rate=SampleRate(_MODEL_RATE),
        started_at_ms=0.0,
        ended_at_ms=duration_s * 1000.0,
        language=chosen,
    )

    for model_name in models:
        for threads in thread_counts:
            try:
                recognizer = _build_benchmark_recognizer(container, model_name, threads, chosen)
                recognizer.warmup()
                timings = []
                text = ""
                for _ in range(max(1, repeats)):
                    started = time.perf_counter()
                    transcript = recognizer.transcribe(utterance)
                    timings.append((time.perf_counter() - started) * 1000.0)
                    text = transcript.text
                recognizer.close()
            except InterpreterError as exc:
                print(f"  {model_name:<10} {threads:>8}   failed: {exc}")
                continue

            best = min(timings)
            print(
                f"  {model_name:<10} {threads:>8} {best:>9.0f}ms "
                f"{best / (duration_s * 1000.0):>9.2f}x   {text[:28]}"
            )

    endpoint_ms = container.settings.vad.min_silence_ms
    heading("How to read this")
    print("  'vs audio' is decode time divided by utterance length. It is NOT a")
    print("  fixed property of the model: Whisper pads its encoder to a 30-second")
    print("  window, so decode time barely changes with utterance length and the")
    print("  ratio simply falls as utterances get longer.")
    print()
    print("  The absolute decode time is what matters, because it is paid in full")
    print("  after you stop speaking, on top of the end-of-speech delay:")
    print()
    print(f"      speech-to-text delay  =  {endpoint_ms} ms endpoint  +  decode time")
    print()
    print("  Translation and speech synthesis are added on top in Phases 5 and 6.")
    return _EXIT_OK


def _benchmark_audio(
    container: Container,
    wav_path: Path | None,
    language: LanguageCode | None,
) -> tuple[np.ndarray, str]:
    """Obtain a single utterance to benchmark with.

    Args:
        container: Built application container.
        wav_path: Explicit file, or ``None`` to pick a recording.
        language: Language to tag the segmented utterance with.

    Returns:
        Mono float32 audio at 16 kHz and a label describing its source.
    """
    if wav_path is not None and wav_path.is_file():
        candidates = [wav_path]
    else:
        # Newest first. Benchmarking against silence measures hallucination
        # rather than transcription: Whisper invents text to fill the gap,
        # producing far more decoder tokens than real speech.
        candidates = sorted(container.paths.recordings_dir.glob("*.wav"), reverse=True)[
            :_MAX_BENCHMARK_CANDIDATES
        ]

    # Speech is detected with the voice activity detector, not an amplitude
    # threshold. A recording of a quiet room with one desk knock peaks at 0.15
    # while its RMS is 0.01 - loud enough to pass any peak test, and entirely
    # devoid of speech. The detector is the tool that answers this question.
    for path in candidates:
        try:
            utterances = _load_utterances(container, path, language)
        except InterpreterError as exc:
            logger.debug("Could not segment %s: %s", path.name, exc)
            continue

        if utterances:
            # The longest utterance is the most representative: short ones are
            # dominated by the fixed encoder cost either way, and a longer one
            # also exercises the decoder.
            longest = max(utterances, key=lambda item: item.pcm.size)
            return longest.pcm, f"{path.name}, longest of {len(utterances)}"

        logger.debug("Skipping %s: no speech detected", path.name)

    # No usable recording: a tone still measures the encoder, which dominates,
    # but the decoder will produce nothing meaningful.
    samples = np.arange(_MODEL_RATE * 3) / _MODEL_RATE
    tone = (0.1 * np.sin(2 * np.pi * 220 * samples)).astype(np.float32)
    return tone, "synthetic tone - run '--record 10' and speak for a real measurement"


def _read_as_model_audio(path: Path) -> np.ndarray:
    """Read a WAV file as mono float32 at the model's sample rate.

    Args:
        path: File to read.

    Returns:
        Mono float32 samples at 16 kHz.
    """
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono: np.ndarray = np.asarray(data, dtype=np.float32).mean(axis=1).astype(np.float32)
    if int(rate) != _MODEL_RATE:
        preprocessor = AudioPreprocessor(SampleRate(int(rate)), SampleRate(_MODEL_RATE))
        mono = preprocessor.process(mono, last=True)
    return mono


def _build_benchmark_recognizer(
    container: Container,
    model_name: str,
    threads: int,
    language: LanguageCode | None,
) -> FasterWhisperRecognizer:
    """Build a recogniser with overridden model, thread count and language.

    Args:
        container: Built application container.
        model_name: Whisper model size.
        threads: CPU thread count.
        language: Language to decode, or ``None`` to auto-detect.

    Returns:
        The recogniser.
    """
    descriptor = container.models.get(f"whisper-{model_name}")
    model_dir = container.model_repository.ensure(descriptor)
    stt = container.settings.stt

    return FasterWhisperRecognizer(
        model_dir=model_dir,
        model_id=descriptor.id,
        device=stt.device,
        compute_type=stt.compute_type,
        cpu_threads=threads,
        language=language,
        options=WhisperDecodeOptions(beam_size=stt.beam_size),
    )
