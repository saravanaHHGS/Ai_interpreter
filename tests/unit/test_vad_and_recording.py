"""Unit tests for the energy detector, WAV replay, recording and the registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_interpreter.domain.entities import AudioFrame
from ai_interpreter.domain.errors import AudioCaptureError, ConfigurationError
from ai_interpreter.domain.value_objects import LanguageCode, SampleRate
from ai_interpreter.infrastructure.audio.capture.wav_file import WavFileSource
from ai_interpreter.infrastructure.audio.recording import WavRecorder
from ai_interpreter.infrastructure.audio.vad.energy import EnergyVad
from ai_interpreter.infrastructure.models.registry import ModelRegistry

pytestmark = pytest.mark.unit

RATE = SampleRate(16000)
FRAME = 512


def _frame(samples: np.ndarray, timestamp_ms: float = 0.0) -> AudioFrame:
    """Wrap samples in an AudioFrame.

    Args:
        samples: Mono float32 samples.
        timestamp_ms: Frame timestamp.

    Returns:
        The frame.
    """
    return AudioFrame(pcm=samples, sample_rate=RATE, timestamp_ms=timestamp_ms)


def _noise(level: float, size: int = FRAME, seed: int = 0) -> np.ndarray:
    """Generate white noise at a given RMS level.

    Args:
        level: Target RMS.
        size: Sample count.
        seed: Random seed, so tests are deterministic.

    Returns:
        Float32 samples.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(size) * level).astype(np.float32)


class TestEnergyVad:
    """The model-free fallback detector."""

    def test_reports_its_frame_requirements(self) -> None:
        vad = EnergyVad()
        assert vad.required_frame_samples == 512
        assert vad.sample_rate == RATE

    def test_silence_scores_zero(self) -> None:
        vad = EnergyVad()
        assert vad.speech_probability(_frame(np.zeros(FRAME, dtype=np.float32))) == 0.0

    def test_loud_audio_scores_high_above_a_quiet_floor(self) -> None:
        vad = EnergyVad(snr_db=12.0)
        for _ in range(50):
            vad.speech_probability(_frame(_noise(0.001)))

        assert vad.speech_probability(_frame(_noise(0.3, seed=1))) == pytest.approx(1.0)

    def test_noise_floor_adapts_upward(self) -> None:
        vad = EnergyVad()
        initial = vad.noise_floor
        for index in range(500):
            vad.speech_probability(_frame(_noise(0.02, seed=index)))

        assert vad.noise_floor > initial

    def test_sustained_speech_does_not_blind_the_detector(self) -> None:
        # A symmetric average would drift up during a long sentence until the
        # speaker's own voice became the "noise floor" and the detector
        # stopped hearing them. The invariant that matters is not how far the
        # estimate moved, but that speech is still detected afterwards.
        vad = EnergyVad()
        for index in range(100):
            vad.speech_probability(_frame(_noise(0.001, seed=index)))

        for index in range(300):
            probability = vad.speech_probability(_frame(_noise(0.5, seed=index + 1000)))

        assert probability == pytest.approx(1.0)
        assert vad.noise_floor < 0.05, "noise floor crept up towards the speech level"

    def test_recalibrates_in_a_room_that_is_noisy_from_the_start(self) -> None:
        # Zeroing the adaptation rate during speech would leave a detector
        # started in a noisy room reporting speech forever.
        vad = EnergyVad()
        for index in range(2000):
            vad.speech_probability(_frame(_noise(0.05, seed=index)))

        assert vad.noise_floor > 1e-3

    def test_reset_clears_the_estimate(self) -> None:
        vad = EnergyVad()
        for index in range(200):
            vad.speech_probability(_frame(_noise(0.05, seed=index)))
        vad.reset()

        assert vad.noise_floor == pytest.approx(1e-4)

    def test_rejects_wrong_frame_size(self) -> None:
        vad = EnergyVad(frame_samples=512)
        with pytest.raises(ValueError, match="configured for 512 samples"):
            vad.speech_probability(_frame(np.zeros(256, dtype=np.float32)))

    def test_counts_frames(self) -> None:
        vad = EnergyVad()
        for _ in range(7):
            vad.speech_probability(_frame(np.zeros(FRAME, dtype=np.float32)))
        assert vad.frames_scored == 7


class TestWavRecorder:
    """Writing captured audio to disk."""

    def test_writes_a_readable_file(self, tmp_path: Path) -> None:
        import soundfile as sf

        path = tmp_path / "out.wav"
        with WavRecorder(path, RATE) as recorder:
            recorder.write(_noise(0.1, size=16000))

        assert path.is_file()
        info = sf.info(str(path))
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.duration == pytest.approx(1.0, abs=0.01)

    def test_tracks_duration(self, tmp_path: Path) -> None:
        with WavRecorder(tmp_path / "out.wav", RATE) as recorder:
            recorder.write(np.zeros(8000, dtype=np.float32))
            assert recorder.duration_ms == pytest.approx(500.0)

    def test_appends_across_writes(self, tmp_path: Path) -> None:
        import soundfile as sf

        path = tmp_path / "out.wav"
        with WavRecorder(path, RATE) as recorder:
            for _ in range(4):
                recorder.write(np.zeros(4000, dtype=np.float32))

        assert sf.info(str(path)).duration == pytest.approx(1.0, abs=0.01)

    def test_creates_missing_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "out.wav"
        with WavRecorder(path, RATE) as recorder:
            recorder.write(np.zeros(100, dtype=np.float32))
        assert path.is_file()

    def test_writing_before_open_is_an_error(self, tmp_path: Path) -> None:
        recorder = WavRecorder(tmp_path / "out.wav", RATE)
        with pytest.raises(AudioCaptureError, match="closed recorder"):
            recorder.write(np.zeros(100, dtype=np.float32))

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        recorder = WavRecorder(tmp_path / "out.wav", RATE)
        recorder.open()
        recorder.close()
        recorder.close()

    def test_empty_write_is_ignored(self, tmp_path: Path) -> None:
        with WavRecorder(tmp_path / "out.wav", RATE) as recorder:
            recorder.write(np.empty(0, dtype=np.float32))
            assert recorder.duration_ms == 0.0


class TestWavFileSource:
    """Replaying a recording through the microphone interface."""

    @pytest.fixture
    def wav_path(self, tmp_path: Path) -> Path:
        """Write a one-second test file.

        Args:
            tmp_path: Temporary directory.

        Returns:
            Path to the file.
        """
        path = tmp_path / "input.wav"
        with WavRecorder(path, RATE) as recorder:
            recorder.write(_noise(0.2, size=16000))
        return path

    def test_reports_file_properties(self, wav_path: Path) -> None:
        source = WavFileSource(wav_path, frame_ms=32)
        assert source.sample_rate == RATE
        assert source.duration_ms == pytest.approx(1000.0, abs=1.0)
        assert source.device.host_api == "file"

    def test_reads_fixed_size_blocks(self, wav_path: Path) -> None:
        source = WavFileSource(wav_path, frame_ms=32)
        source.start()
        frame = source.read()

        assert frame is not None
        assert frame.pcm.size == 512

    def test_returns_none_at_end_of_file(self, wav_path: Path) -> None:
        source = WavFileSource(wav_path, frame_ms=32)
        source.start()
        while source.read() is not None:
            pass

        assert source.is_exhausted
        assert source.read() is None

    def test_timestamps_advance(self, wav_path: Path) -> None:
        source = WavFileSource(wav_path, frame_ms=32)
        source.start()
        first = source.read()
        second = source.read()

        assert first is not None
        assert second is not None
        assert second.timestamp_ms == pytest.approx(first.timestamp_ms + 32.0)

    def test_realtime_pacing_slows_delivery(self, wav_path: Path) -> None:
        # A paced source must take roughly wall-clock time; an unpaced one
        # must not. Coarse bounds keep this robust on a loaded machine.
        import time as time_module

        paced = WavFileSource(wav_path, frame_ms=100, realtime=True)
        paced.start()
        started = time_module.monotonic()
        for _ in range(4):  # 400 ms of audio
            assert paced.read() is not None
        elapsed = time_module.monotonic() - started
        assert elapsed >= 0.25

        fast = WavFileSource(wav_path, frame_ms=100)
        fast.start()
        started = time_module.monotonic()
        while fast.read() is not None:
            pass
        assert time_module.monotonic() - started < 0.5

    def test_looping_never_exhausts(self, wav_path: Path) -> None:
        source = WavFileSource(wav_path, frame_ms=32, loop=True)
        source.start()
        for _ in range(100):
            assert source.read() is not None
        assert not source.is_exhausted

    def test_reading_before_start_is_an_error(self, wav_path: Path) -> None:
        with pytest.raises(AudioCaptureError, match="not running"):
            WavFileSource(wav_path).read()

    def test_missing_file_reports_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(AudioCaptureError, match="Could not read audio file"):
            WavFileSource(tmp_path / "absent.wav")

    def test_unsupported_sample_rate_suggests_ffmpeg(self, tmp_path: Path) -> None:
        import soundfile as sf

        path = tmp_path / "odd.wav"
        sf.write(str(path), np.zeros(1000, dtype=np.float32), 12345, subtype="PCM_16")

        with pytest.raises(AudioCaptureError, match="ffmpeg"):
            WavFileSource(path)

    def test_stereo_is_downmixed_to_mono(self, tmp_path: Path) -> None:
        import soundfile as sf

        path = tmp_path / "stereo.wav"
        stereo = np.zeros((16000, 2), dtype=np.float32)
        stereo[:, 0] = 1.0
        stereo[:, 1] = -1.0
        sf.write(str(path), stereo, 16000, subtype="PCM_16")

        source = WavFileSource(path, frame_ms=32)
        source.start()
        frame = source.read()

        assert frame is not None
        assert float(np.max(np.abs(frame.pcm))) == pytest.approx(0.0, abs=1e-4)


class TestModelRegistry:
    """Loading the model catalogue."""

    def test_loads_the_committed_registry(self) -> None:
        registry = ModelRegistry.load(Path("config/models.yaml"))
        descriptor = registry.get("silero-vad")

        assert descriptor.task == "vad"
        assert descriptor.repo_id == "onnx-community/silero-vad"
        assert descriptor.files == ("onnx/model.onnx",)

    def test_revision_is_a_full_commit_hash(self) -> None:
        # A branch name would let upstream change the weights underneath us.
        descriptor = ModelRegistry.load(Path("config/models.yaml")).get("silero-vad")
        assert len(descriptor.revision) == 40

    def test_vad_model_declares_language_independence(self) -> None:
        descriptor = ModelRegistry.load(Path("config/models.yaml")).get("silero-vad")
        assert descriptor.supports(LanguageCode("ta"))
        assert descriptor.supports(LanguageCode("en"))

    def test_filters_by_task(self) -> None:
        registry = ModelRegistry.load(Path("config/models.yaml"))
        assert len(registry.for_task("vad")) >= 1
        assert registry.for_task("nonexistent") == ()

    def test_unknown_model_lists_what_exists(self) -> None:
        registry = ModelRegistry.load(Path("config/models.yaml"))
        with pytest.raises(ConfigurationError, match=r"Declared models: .*silero-vad"):
            registry.get("no-such-model")

    def test_missing_file_reports_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="Model registry not found"):
            ModelRegistry.load(tmp_path / "absent.yaml")

    def test_missing_required_field_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text("models:\n  broken:\n    task: vad\n", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="missing required field"):
            ModelRegistry.load(path)

    def test_rejects_a_file_without_a_models_key(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text("something_else: 1\n", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="top-level 'models' mapping"):
            ModelRegistry.load(path)
