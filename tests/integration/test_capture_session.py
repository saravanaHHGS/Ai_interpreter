"""Integration tests for the capture chain.

Runs the real chain - source, preprocessor, frame assembler, detector,
segmenter, background thread - against a WAV file instead of a microphone.
That is the whole reason ``AudioSource`` is a port: this exercises every line
that runs in production, deterministically, on a machine with no audio
hardware at all.

The detector here is a scripted fake rather than Silero. The neural model is
verified separately; what these tests check is the *plumbing*, and a fake with
a known speech pattern makes the expected output exact rather than
approximate.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ai_interpreter.application.services.capture_session import CaptureSession
from ai_interpreter.application.services.utterance_segmenter import (
    SegmenterState,
    UtteranceSegmenter,
)
from ai_interpreter.domain.entities import AudioFrame, Utterance
from ai_interpreter.domain.value_objects import SampleRate
from ai_interpreter.infrastructure.audio.capture.wav_file import WavFileSource
from ai_interpreter.infrastructure.audio.dsp import AudioPreprocessor
from ai_interpreter.infrastructure.audio.recording import WavRecorder

pytestmark = pytest.mark.integration

RATE_48K = SampleRate(48000)
RATE_16K = SampleRate(16000)


class ScriptedVad:
    """A detector returning speech for pre-arranged frame ranges.

    Satisfies the ``VoiceActivityDetector`` port structurally, with no base
    class and no mock framework - the point of using Protocols.

    Args:
        speech_ranges: Half-open frame index ranges considered speech.
    """

    def __init__(self, speech_ranges: list[tuple[int, int]]) -> None:
        self._ranges = speech_ranges
        self._index = 0
        self.resets = 0

    @property
    def required_frame_samples(self) -> int:
        return 512

    @property
    def sample_rate(self) -> SampleRate:
        return RATE_16K

    def speech_probability(self, frame: AudioFrame) -> float:
        current = self._index
        self._index += 1
        return 1.0 if any(start <= current < end for start, end in self._ranges) else 0.0

    def reset(self) -> None:
        self.resets += 1


def _write_wav(path: Path, seconds: float, sample_rate: SampleRate) -> Path:
    """Write a noise file of a given length.

    Args:
        path: Destination.
        seconds: Duration.
        sample_rate: Rate to write at.

    Returns:
        The path written.
    """
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(sample_rate.hz * seconds)) * 0.2).astype(np.float32)
    with WavRecorder(path, sample_rate) as recorder:
        recorder.write(samples)
    return path


def _run_to_completion(session: CaptureSession, timeout: float = 15.0) -> None:
    """Start a session and wait for the finite source to be exhausted.

    Args:
        session: Session to run.
        timeout: Maximum seconds to wait.

    Raises:
        AssertionError: If the session does not finish in time.
    """
    session.start()
    deadline = time.monotonic() + timeout
    while session.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    session.stop()
    assert not session.is_running, "capture session did not finish"


def _build(
    source: WavFileSource,
    vad: ScriptedVad,
    **segmenter_kwargs: object,
) -> tuple[CaptureSession, UtteranceSegmenter, list[Utterance]]:
    """Assemble a session over a file source.

    Args:
        source: Audio source.
        vad: Detector to use.
        **segmenter_kwargs: Segmenter parameter overrides.

    Returns:
        The session, its segmenter, and a list that collects utterances.
    """
    defaults: dict[str, object] = {
        "sample_rate": RATE_16K,
        "threshold": 0.5,
        "min_speech_ms": 64,
        "min_silence_ms": 96,
        "pre_roll_ms": 64,
        "max_utterance_ms": 12000,
    }
    defaults.update(segmenter_kwargs)
    segmenter = UtteranceSegmenter(**defaults)  # type: ignore[arg-type]

    collected: list[Utterance] = []
    session = CaptureSession(
        source=source,
        preprocessor=AudioPreprocessor(source.sample_rate, RATE_16K, high_pass_hz=80),
        vad=vad,
        segmenter=segmenter,
        on_utterance=collected.append,
    )
    return session, segmenter, collected


class TestCaptureChain:
    """End-to-end behaviour over a file."""

    def test_processes_a_file_and_stops(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 2.0, RATE_48K), frame_ms=20)
        session, _, _ = _build(source, ScriptedVad([]))

        _run_to_completion(session)
        stats = session.stats()

        assert stats.frames_captured == 100  # 2 s of 20 ms blocks
        assert stats.vad_frames == pytest.approx(62, abs=2)  # 2 s of 32 ms frames
        assert stats.dropped_blocks == 0
        assert session.error is None

    def test_resamples_48k_to_16k(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 2.0, RATE_48K), frame_ms=20)
        session, _, _ = _build(source, ScriptedVad([]))

        _run_to_completion(session)

        # 2 s at 16 kHz in 512-sample frames is 62.5, so 62 whole frames plus
        # a padded flush frame.
        assert 62 <= session.stats().vad_frames <= 64

    def test_accepts_a_16k_source_without_resampling(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 2.0, RATE_16K), frame_ms=32)
        session, _, _ = _build(source, ScriptedVad([]))

        _run_to_completion(session)
        assert 62 <= session.stats().vad_frames <= 64

    def test_detects_scripted_utterances(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 4.0, RATE_48K), frame_ms=20)
        # Two speech bursts separated by clear silence.
        vad = ScriptedVad([(10, 30), (60, 80)])
        session, _, collected = _build(source, vad)

        _run_to_completion(session)

        assert len(collected) == 2
        assert session.stats().utterances == 2

    def test_utterance_audio_is_16k_mono_float32(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 3.0, RATE_48K), frame_ms=20)
        session, _, collected = _build(source, ScriptedVad([(10, 30)]))

        _run_to_completion(session)

        assert collected
        utterance = collected[0]
        assert utterance.sample_rate == RATE_16K
        assert utterance.pcm.ndim == 1
        assert utterance.pcm.dtype == np.float32

    def test_detector_state_is_reset_between_utterances(self, tmp_path: Path) -> None:
        # Silero is recurrent; carrying state across utterances degrades the
        # first frames of the next one.
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 4.0, RATE_48K), frame_ms=20)
        vad = ScriptedVad([(10, 30), (60, 80)])
        session, _, _ = _build(source, vad)

        _run_to_completion(session)

        assert vad.resets >= 3  # one at start, one after each utterance

    def test_trailing_speech_is_flushed_at_end_of_input(self, tmp_path: Path) -> None:
        # Without draining, the final utterance of every fixed-length
        # recording would be silently lost.
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 2.0, RATE_48K), frame_ms=20)
        vad = ScriptedVad([(10, 999)])  # still speaking when the file ends
        session, _, collected = _build(source, vad)

        _run_to_completion(session)

        assert len(collected) == 1
        assert collected[0].duration_ms > 0


class TestCallbacks:
    """Observation hooks used by the UI and the recorder."""

    def test_frame_callback_receives_every_frame(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 1.0, RATE_48K), frame_ms=20)
        seen: list[float] = []

        segmenter = UtteranceSegmenter(sample_rate=RATE_16K)
        session = CaptureSession(
            source=source,
            preprocessor=AudioPreprocessor(RATE_48K, RATE_16K),
            vad=ScriptedVad([]),
            segmenter=segmenter,
            on_frame=lambda _frame, probability: seen.append(probability),
        )

        _run_to_completion(session)
        assert len(seen) == session.stats().vad_frames

    def test_state_change_callback_reports_transitions(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 3.0, RATE_48K), frame_ms=20)
        states: list[SegmenterState] = []

        segmenter = UtteranceSegmenter(
            sample_rate=RATE_16K, min_speech_ms=64, min_silence_ms=96, pre_roll_ms=64
        )
        session = CaptureSession(
            source=source,
            preprocessor=AudioPreprocessor(RATE_48K, RATE_16K),
            vad=ScriptedVad([(10, 40)]),
            segmenter=segmenter,
            on_state_change=states.append,
        )

        _run_to_completion(session)

        assert SegmenterState.SPEECH in states
        assert SegmenterState.SILENCE in states

    def test_error_callback_fires_and_stops_the_session(self, tmp_path: Path) -> None:
        # A capture session that has silently died while the UI still shows
        # "listening" is the worst possible failure mode.
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 1.0, RATE_48K), frame_ms=20)

        class ExplodingVad(ScriptedVad):
            def speech_probability(self, frame: AudioFrame) -> float:
                raise RuntimeError("detector exploded")

        errors: list[Exception] = []
        session = CaptureSession(
            source=source,
            preprocessor=AudioPreprocessor(RATE_48K, RATE_16K),
            vad=ExplodingVad([]),
            segmenter=UtteranceSegmenter(sample_rate=RATE_16K),
            on_error=errors.append,
        )

        _run_to_completion(session)

        assert len(errors) == 1
        assert isinstance(session.error, RuntimeError)


class TestLifecycle:
    """Starting and stopping."""

    def test_start_is_idempotent(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 3.0, RATE_48K), frame_ms=20)
        session, _, _ = _build(source, ScriptedVad([]))

        session.start()
        session.start()
        session.stop()

        assert not session.is_running

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 1.0, RATE_48K), frame_ms=20)
        session, _, _ = _build(source, ScriptedVad([]))

        session.start()
        session.stop()
        session.stop()

    def test_works_as_a_context_manager(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 1.0, RATE_48K), frame_ms=20)
        session, _, _ = _build(source, ScriptedVad([]))

        with session:
            assert session.is_running
        assert not session.is_running

    def test_runs_on_a_background_thread(self, tmp_path: Path) -> None:
        source = WavFileSource(_write_wav(tmp_path / "a.wav", 5.0, RATE_48K), frame_ms=20)
        caller = threading.current_thread()
        worker_threads: list[threading.Thread] = []

        session_with_probe = CaptureSession(
            source=source,
            preprocessor=AudioPreprocessor(RATE_48K, RATE_16K),
            vad=ScriptedVad([]),
            segmenter=UtteranceSegmenter(sample_rate=RATE_16K),
            on_frame=lambda *_: worker_threads.append(threading.current_thread()),
        )

        _run_to_completion(session_with_probe)

        assert worker_threads
        assert all(thread is not caller for thread in worker_threads)
        assert worker_threads[0].name == "audio-capture"
