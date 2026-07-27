"""Unit tests for domain entities."""

from __future__ import annotations

import numpy as np
import pytest

from ai_interpreter.domain.entities import (
    AudioFrame,
    DeviceInfo,
    GpuInfo,
    HardwareInfo,
    ModelDescriptor,
    SpeechAudio,
    Transcript,
    Utterance,
    UtteranceId,
)
from ai_interpreter.domain.value_objects import (
    Confidence,
    DeviceKind,
    LanguageCode,
    SampleRate,
)

pytestmark = pytest.mark.unit

RATE_16K = SampleRate(16000)


def _silence(samples: int) -> np.ndarray:
    """Build a silent mono float32 buffer.

    Args:
        samples: Number of samples.

    Returns:
        A zero-filled float32 array.
    """
    return np.zeros(samples, dtype=np.float32)


class TestAudioFrame:
    """Audio frame contract enforcement."""

    def test_computes_duration_from_sample_count(self) -> None:
        frame = AudioFrame(pcm=_silence(320), sample_rate=RATE_16K, timestamp_ms=0.0)
        assert frame.duration_ms == pytest.approx(20.0)

    def test_rejects_stereo_buffer(self) -> None:
        stereo = np.zeros((320, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D mono buffer"):
            AudioFrame(pcm=stereo, sample_rate=RATE_16K, timestamp_ms=0.0)

    def test_rejects_wrong_dtype(self) -> None:
        int_buffer = np.zeros(320, dtype=np.int16)
        with pytest.raises(ValueError, match="must be float32"):
            AudioFrame(pcm=int_buffer, sample_rate=RATE_16K, timestamp_ms=0.0)

    def test_rejects_empty_buffer(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            AudioFrame(pcm=_silence(0), sample_rate=RATE_16K, timestamp_ms=0.0)

    def test_reports_peak_and_rms(self) -> None:
        pcm = np.array([0.0, 0.5, -1.0, 0.5], dtype=np.float32)
        frame = AudioFrame(pcm=pcm, sample_rate=RATE_16K, timestamp_ms=0.0)
        assert frame.peak_amplitude == pytest.approx(1.0)
        assert frame.rms == pytest.approx(0.6123724, rel=1e-5)


class TestUtterance:
    """Utterance contract enforcement."""

    def test_computes_duration(self) -> None:
        utterance = Utterance(
            id=UtteranceId("u1"),
            pcm=_silence(16000),
            sample_rate=RATE_16K,
            started_at_ms=0.0,
            ended_at_ms=1000.0,
        )
        assert utterance.duration_ms == pytest.approx(1000.0)

    def test_rejects_reversed_time_range(self) -> None:
        with pytest.raises(ValueError, match="ends before it starts"):
            Utterance(
                id=UtteranceId("u1"),
                pcm=_silence(160),
                sample_rate=RATE_16K,
                started_at_ms=500.0,
                ended_at_ms=100.0,
            )

    def test_equality_uses_identity_not_array_comparison(self) -> None:
        # A dataclass __eq__ over a numpy field would raise here; eq=False
        # keeps identity semantics and makes this safe.
        first = Utterance(
            id=UtteranceId("u1"),
            pcm=_silence(160),
            sample_rate=RATE_16K,
            started_at_ms=0.0,
            ended_at_ms=10.0,
        )
        second = Utterance(
            id=UtteranceId("u1"),
            pcm=_silence(160),
            sample_rate=RATE_16K,
            started_at_ms=0.0,
            ended_at_ms=10.0,
        )
        assert first != second
        assert first == first


class TestTranscript:
    """Transcript emptiness detection."""

    @pytest.mark.parametrize(("text", "expected"), [("", True), ("   ", True), ("hello", False)])
    def test_detects_empty_text(self, text: str, expected: bool) -> None:
        transcript = Transcript(
            utterance_id=UtteranceId("u1"),
            text=text,
            language=LanguageCode("en"),
            confidence=Confidence(0.9),
            is_final=True,
        )
        assert transcript.is_empty is expected


class TestSpeechAudio:
    """Synthesised speech contract enforcement."""

    def test_computes_duration(self) -> None:
        audio = SpeechAudio(
            utterance_id=UtteranceId("u1"),
            pcm=_silence(22050),
            sample_rate=SampleRate(22050),
            language=LanguageCode("en"),
        )
        assert audio.duration_ms == pytest.approx(1000.0)

    def test_rejects_negative_chunk_index(self) -> None:
        with pytest.raises(ValueError, match="chunk_index cannot be negative"):
            SpeechAudio(
                utterance_id=UtteranceId("u1"),
                pcm=_silence(100),
                sample_rate=SampleRate(22050),
                language=LanguageCode("en"),
                chunk_index=-1,
            )


class TestDeviceInfo:
    """Virtual cable detection heuristic."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("CABLE Input (VB-Audio Virtual Cable)", True),
            ("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", True),
            ("Speakers (Conexant ISST Audio)", False),
            ("Microphone Array", False),
        ],
    )
    def test_identifies_virtual_cables(self, name: str, expected: bool) -> None:
        device = DeviceInfo(
            index=0,
            name=name,
            kind=DeviceKind.OUTPUT,
            max_channels=2,
            default_sample_rate=48000.0,
        )
        assert device.is_virtual_cable is expected


class TestHardwareInfo:
    """GPU capability reporting."""

    def test_reports_no_cuda_without_nvidia_gpu(self, cpu_only_hardware: HardwareInfo) -> None:
        assert cpu_only_hardware.has_cuda_gpu is False
        assert cpu_only_hardware.cuda_memory_gb == 0.0

    def test_reports_largest_nvidia_gpu_memory(self) -> None:
        hardware = HardwareInfo(
            os_name="Windows",
            os_version="10.0",
            cpu_name="test",
            physical_cores=8,
            logical_cores=16,
            total_ram_gb=32.0,
            available_ram_gb=16.0,
            free_disk_gb=100.0,
            python_version="3.12.10",
            gpus=(
                GpuInfo(name="RTX 3060", total_memory_mb=12288, vendor="nvidia"),
                GpuInfo(name="RTX 2060", total_memory_mb=6144, vendor="nvidia"),
            ),
        )
        assert hardware.has_cuda_gpu is True
        assert hardware.cuda_memory_gb == pytest.approx(12.0)


class TestModelDescriptor:
    """Language support declaration."""

    def test_matches_listed_language(self) -> None:
        descriptor = ModelDescriptor(
            id="indictrans2-indic-en",
            task="mt",
            repo_id="ai4bharat/indictrans2-indic-en-dist-200M",
            revision="main",
            languages=("ta", "hi", "en"),
            size_mb=450,
            runtime="ctranslate2",
        )
        assert descriptor.supports(LanguageCode("ta"))
        assert not descriptor.supports(LanguageCode("bn"))

    def test_wildcard_matches_any_language(self) -> None:
        descriptor = ModelDescriptor(
            id="whisper-small",
            task="stt",
            repo_id="openai/whisper-small",
            revision="main",
            languages=("*",),
            size_mb=460,
            runtime="ctranslate2",
        )
        assert descriptor.supports(LanguageCode("ta"))
