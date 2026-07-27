"""Audio commands: device listing and live capture tests.

Separated from :mod:`ai_interpreter.cli` because these are the first commands
that touch hardware and take time, and mixing them into the argument parsing
module would make both harder to follow.

``--record`` is the phase's real deliverable. It runs the complete Phase 3
chain against your microphone, shows a live level meter and speech indicator,
and writes two WAV files: the raw device audio and the conditioned 16 kHz
audio the speech model will actually receive. Hearing the difference is worth
more than any amount of documentation about resampling.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Final

import numpy as np

from ai_interpreter.app.container import Container
from ai_interpreter.application.services.capture_session import CaptureSession
from ai_interpreter.application.services.utterance_segmenter import SegmenterState
from ai_interpreter.domain.entities import AudioFrame, Utterance
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.domain.value_objects import DeviceKind, SampleRate
from ai_interpreter.infrastructure.audio.recording import WavRecorder
from ai_interpreter.presentation.console import (
    WIDTH,
    format_device_table,
    heading,
    level_bar,
    row,
    terminal_width,
)

__all__ = ["run_list_devices", "run_record"]

logger = logging.getLogger(__name__)

_EXIT_OK: Final[int] = 0
_EXIT_ERROR: Final[int] = 1

# Refresh rate of the live meter. 10 Hz reads as smooth and costs nothing;
# redrawing per 32 ms frame would spend more time on console I/O than on audio.
_METER_INTERVAL_SECONDS: Final[float] = 0.1


def run_list_devices(container: Container) -> int:
    """Print every audio endpoint PortAudio can see.

    Args:
        container: Built application container.

    Returns:
        Process exit code.
    """
    print("=" * WIDTH)
    print("  Audio devices")
    print("=" * WIDTH)

    try:
        inputs = container.devices.list_devices(DeviceKind.INPUT)
        outputs = container.devices.list_devices(DeviceKind.OUTPUT)
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    heading("Input devices (microphones)")
    for line in format_device_table(inputs):
        print(line)

    heading("Output devices (playback)")
    for line in format_device_table(outputs):
        print(line)

    heading("Selection")
    configured = container.settings.audio.input.device
    row("Configured input", configured or "(automatic)")
    row("Preferred host API", container.settings.audio.input.host_api or "(automatic order)")

    try:
        chosen = container.resolve_input_device()
        row("Would capture from", f"{chosen.name}  [{chosen.host_api}]")
    except InterpreterError as exc:
        print()
        print(f"  {exc}")
        return _EXIT_ERROR

    print()
    print("  Host API notes:")
    print("    WASAPI       preferred - full device names, native 48 kHz, lowest latency")
    print("    DirectSound  works, but resamples to 44.1 kHz and adds latency")
    print("    MME          legacy - truncates device names to 31 characters")
    print("    WDM-KS       exclusive access - takes the device from other applications")
    return _EXIT_OK


class _LiveMeter:
    """Renders a single-line level meter that updates in place.

    Args:
        enabled: Whether to draw anything. Disabled when output is redirected,
            because carriage-return redraws turn a log file into noise.
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._last_draw = 0.0
        self._lock = threading.Lock()
        self._level = 0.0
        self._probability = 0.0
        self._speaking = False
        self._drawn_width = 0

    def update(self, level: float, probability: float, speaking: bool) -> None:
        """Record the newest values and redraw if due.

        Called from the capture worker thread.

        Args:
            level: Peak amplitude of the latest frame.
            probability: Its speech probability.
            speaking: Whether an utterance is in progress.
        """
        if not self._enabled:
            return

        with self._lock:
            self._level = level
            self._probability = probability
            self._speaking = speaking

            now = time.monotonic()
            if now - self._last_draw < _METER_INTERVAL_SECONDS:
                return
            self._last_draw = now
            self._draw()

    def _draw(self) -> None:
        """Write the meter over the current console line.

        The bar is sized from the terminal width rather than a fixed constant.
        A line wider than the terminal wraps, and a wrapped line cannot be
        overwritten with a carriage return - the meter would leave a trail of
        half-drawn rows instead of updating in place.
        """
        width = terminal_width()
        status = "SPEAKING" if self._speaking else "silence "
        suffix = f"  peak {self._level:5.3f}  speech {self._probability:4.2f}  {status}"
        bar_width = max(8, width - len(suffix) - 2)

        line = f"  {level_bar(self._level, bar_width)}{suffix}"
        sys.stdout.write("\r" + line[:width])
        sys.stdout.flush()
        self._drawn_width = max(self._drawn_width, min(len(line), width))

    def finish(self) -> None:
        """Move off the meter line so later output is not overwritten."""
        if self._enabled:
            sys.stdout.write("\r" + " " * self._drawn_width + "\r")
            sys.stdout.flush()


def run_record(container: Container, seconds: float, device_name: str | None) -> int:
    """Capture from the microphone and report what the pipeline detected.

    Args:
        container: Built application container.
        seconds: Recording length.
        device_name: Device name fragment overriding configuration.

    Returns:
        Process exit code.
    """
    print("=" * WIDTH)
    print(f"  Microphone capture test - {seconds:.0f} seconds")
    print("=" * WIDTH)

    try:
        device = container.resolve_input_device(device_name)
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    audio_input = container.settings.audio.input
    processing = container.settings.audio.processing
    vad_settings = container.settings.vad

    heading("Configuration")
    row("Device", f"{device.name}  [{device.host_api}]")
    row("Capture format", f"{audio_input.sample_rate} Hz mono, {audio_input.frame_ms} ms blocks")
    row("Model format", f"{processing.target_sample_rate} Hz mono")
    row("High-pass filter", f"{processing.high_pass_hz} Hz" if processing.high_pass_hz else "off")
    row("Input gain", f"{audio_input.gain_db:+.1f} dB")
    row("Detector", vad_settings.provider)
    row("Speech threshold", f"{vad_settings.threshold:.2f}")
    row("End-of-speech silence", f"{vad_settings.min_silence_ms} ms")
    row("Pre-roll kept", f"{vad_settings.pre_roll_ms} ms")

    try:
        print("\n  Preparing the voice activity detector...")
        vad = container.create_vad()
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    source = container.create_microphone_source(device)
    preprocessor = container.create_preprocessor(SampleRate(audio_input.sample_rate))
    segmenter = container.create_segmenter(SampleRate(processing.target_sample_rate))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = container.paths.recordings_dir / f"{timestamp}-raw-{audio_input.sample_rate}.wav"
    processed_path = (
        container.paths.recordings_dir
        / f"{timestamp}-processed-{processing.target_sample_rate}.wav"
    )

    raw_recorder = WavRecorder(raw_path, SampleRate(audio_input.sample_rate))
    processed_recorder = WavRecorder(processed_path, SampleRate(processing.target_sample_rate))
    utterances: list[Utterance] = []
    meter = _LiveMeter(enabled=sys.stdout.isatty())
    write_lock = threading.Lock()

    def on_frame(frame: AudioFrame, probability: float) -> None:
        with write_lock:
            processed_recorder.write(frame.pcm)
        meter.update(frame.peak_amplitude, probability, segmenter.is_speaking)

    def on_utterance(utterance: Utterance) -> None:
        utterances.append(utterance)

    def on_state_change(state: SegmenterState) -> None:
        logger.debug("Segmenter state: %s", state.value)

    session = CaptureSession(
        source=source,
        preprocessor=preprocessor,
        vad=vad,
        segmenter=segmenter,
        on_utterance=on_utterance,
        on_frame=on_frame,
        on_state_change=on_state_change,
    )

    # The raw recorder is fed by wrapping the source's read, so the file holds
    # exactly what the device delivered, before any processing.
    original_read = source.read

    def read_and_record(timeout: float | None = None) -> AudioFrame | None:
        frame = original_read(timeout)
        if frame is not None:
            with write_lock:
                raw_recorder.write(frame.pcm)
        return frame

    source.read = read_and_record  # type: ignore[method-assign]

    heading("Recording")
    print("  Speak normally. Say a few short sentences with pauses between them.\n")

    # Log records arriving on stderr in the middle of a carriage-return meter
    # destroy it. The console is silenced for the duration; the file log keeps
    # every record, so nothing is lost.
    try:
        with container.logging_service.quiet_console():
            try:
                raw_recorder.open()
                processed_recorder.open()
                session.start()
                time.sleep(seconds)
            except KeyboardInterrupt:
                meter.finish()
                print("\n  Interrupted.")
            finally:
                session.stop()
                meter.finish()
                with write_lock:
                    raw_recorder.close()
                    processed_recorder.close()
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    if session.error is not None:
        print(f"\n  Capture failed: {session.error}\n", file=sys.stderr)
        return _EXIT_ERROR

    return _report_results(
        session,
        segmenter,
        utterances,
        raw_path,
        processed_path,
        container.paths.recordings_dir,
    )


def _report_results(
    session: CaptureSession,
    segmenter: object,
    utterances: list[Utterance],
    raw_path: Path,
    processed_path: Path,
    recordings_dir: Path,
) -> int:
    """Print the outcome of a capture test.

    Args:
        session: The finished capture session.
        segmenter: Segmenter used, for its counters.
        utterances: Utterances detected.
        raw_path: Raw recording path.
        processed_path: Processed recording path.
        recordings_dir: Directory both were written to.

    Returns:
        Process exit code.
    """
    stats = session.stats()
    seg_stats = segmenter.stats()  # type: ignore[attr-defined]

    heading("Capture statistics")
    row("Blocks captured", str(stats.frames_captured))
    row("Frames analysed", f"{stats.vad_frames} (32 ms each)")
    row("Peak level", f"{stats.peak_level:.3f}  {level_bar(stats.peak_level, 24)}")
    row("Frames with speech", f"{stats.speech_ratio * 100:.1f}%")
    row("Blocks dropped", str(stats.dropped_blocks))
    row("Short bursts rejected", str(seg_stats.rejected_short_bursts))
    row("Utterances cut at limit", str(seg_stats.forced_cuts))

    heading(f"Utterances detected: {len(utterances)}")
    if utterances:
        for index, utterance in enumerate(utterances, start=1):
            peak = float(np.max(np.abs(utterance.pcm))) if utterance.pcm.size else 0.0
            row(
                f"{index}. {utterance.id}",
                f"{utterance.duration_ms / 1000.0:5.2f} s  "
                f"at {utterance.started_at_ms / 1000.0:6.2f} s  peak {peak:.3f}",
            )
    else:
        print("  None. Nothing was detected as speech.")

    heading("Recordings saved")
    row("Raw from device", raw_path.name)
    row("Processed for models", processed_path.name)
    row("Folder", str(recordings_dir))
    print("\n  Play both. The processed file should sound the same but duller,")
    print("  because it is 16 kHz with frequencies below the high-pass removed.")

    heading("Result")
    problems = _diagnose(stats, len(utterances))
    if problems:
        for problem in problems:
            print(f"  {problem}")
        return _EXIT_ERROR

    print("  Capture is working correctly.")
    return _EXIT_OK


def _diagnose(stats: object, utterance_count: int) -> list[str]:
    """Turn capture statistics into actionable problems.

    Args:
        stats: Capture statistics.
        utterance_count: Utterances detected.

    Returns:
        Problem descriptions, empty when everything looks healthy.
    """
    problems: list[str] = []
    peak = stats.peak_level  # type: ignore[attr-defined]
    dropped = stats.dropped_blocks  # type: ignore[attr-defined]
    captured = stats.frames_captured  # type: ignore[attr-defined]

    if captured == 0:
        problems.append(
            "FAILED   No audio was captured at all. The device may be muted in "
            "Windows Sound settings or in use by another application."
        )
        return problems

    if peak < 0.001:
        problems.append(
            "FAILED   The signal was silent. Check that the microphone is not muted "
            "and that Windows privacy settings allow microphone access."
        )
    elif peak < 0.02:
        problems.append(
            f"WARNING  Very quiet signal (peak {peak:.4f}). Raise the microphone level "
            "in Windows Sound settings, or set audio.input.gain_db to about +10."
        )
    elif peak > 0.99:
        problems.append(
            "WARNING  The signal clipped. Lower the microphone level in Windows Sound "
            "settings, or set a negative audio.input.gain_db."
        )

    if dropped:
        problems.append(
            f"WARNING  {dropped} audio blocks were dropped. The machine could not keep up; "
            "close other applications and try again."
        )

    if utterance_count == 0 and peak >= 0.02:
        problems.append(
            "WARNING  Audio was captured but no speech was detected. If you did speak, "
            "lower vad.threshold, or try vad.provider: energy in a noisy room."
        )

    return problems
