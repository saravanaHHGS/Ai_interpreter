"""Domain entities: the data that flows through the interpretation pipeline.

Entities are frozen (immutable). In a multi-threaded pipeline where an
utterance is handed from the capture thread to the event loop to an inference
worker, immutability removes an entire class of race conditions: nothing can
modify an object another thread is reading.

``eq=False`` is set on every entity holding a numpy array. The default
dataclass ``__eq__`` compares fields with ``==``, which on a numpy array
returns an *array* rather than a bool and raises ``ValueError`` when used in a
boolean context. Identity comparison is what we actually want here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NewType

import numpy as np
from numpy.typing import NDArray

from ai_interpreter.domain.value_objects import (
    Confidence,
    DeviceKind,
    LanguageCode,
    LanguagePair,
    SampleRate,
)

__all__ = [
    "AudioFrame",
    "DeviceInfo",
    "GpuInfo",
    "HardwareInfo",
    "ModelDescriptor",
    "SpeechAudio",
    "Transcript",
    "TranscriptSegment",
    "Translation",
    "Utterance",
    "UtteranceId",
    "VoiceInfo",
]

# A plain string wrapped in NewType: zero runtime cost, but mypy rejects
# passing an arbitrary str where an utterance id is expected.
UtteranceId = NewType("UtteranceId", str)

# All audio inside the application is mono float32 in [-1.0, 1.0].
# Integer PCM is converted at the device boundary and nowhere else, so no
# module downstream has to ask "is this int16 or float?".
AudioBuffer = NDArray[np.float32]


def _validate_mono_float32(pcm: AudioBuffer, label: str) -> None:
    """Ensure a buffer matches the application's internal audio contract.

    Args:
        pcm: Buffer to check.
        label: Name used in the error message.

    Raises:
        ValueError: If the buffer is not one-dimensional float32.
    """
    if pcm.ndim != 1:
        msg = f"{label} must be a 1-D mono buffer, got shape {pcm.shape}"
        raise ValueError(msg)
    if pcm.dtype != np.float32:
        msg = f"{label} must be float32, got {pcm.dtype}"
        raise ValueError(msg)


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class AudioFrame:
    """A fixed-length block of captured audio.

    Produced by the capture thread roughly every ``frame_ms`` milliseconds and
    consumed by the voice activity detector.

    Args:
        pcm: Mono float32 samples in ``[-1.0, 1.0]``.
        sample_rate: Rate the samples were captured at.
        timestamp_ms: Milliseconds since capture started, at the frame's start.

    Raises:
        ValueError: If the buffer is empty or not mono float32.
    """

    pcm: AudioBuffer
    sample_rate: SampleRate
    timestamp_ms: float

    def __post_init__(self) -> None:
        _validate_mono_float32(self.pcm, "AudioFrame.pcm")
        if self.pcm.size == 0:
            msg = "AudioFrame.pcm must contain at least one sample"
            raise ValueError(msg)

    @property
    def duration_ms(self) -> float:
        """Length of this frame in milliseconds."""
        return self.sample_rate.ms_for_samples(self.pcm.size)

    @property
    def peak_amplitude(self) -> float:
        """Loudest absolute sample, used by the UI level meter."""
        return float(np.max(np.abs(self.pcm)))

    @property
    def rms(self) -> float:
        """Root-mean-square level, a better loudness proxy than peak."""
        return float(np.sqrt(np.mean(np.square(self.pcm, dtype=np.float64))))


@dataclass(frozen=True, slots=True, eq=False)
class Utterance:
    """One continuous stretch of speech, bounded by silence.

    The unit of work for the whole pipeline: an utterance is transcribed,
    translated and synthesised as a single item, and is the granularity at
    which cancellation happens when the speaker barges in.

    Args:
        id: Unique identifier propagated through every downstream artefact.
        pcm: Mono float32 audio, including the configured pre-roll.
        sample_rate: Rate of ``pcm``.
        started_at_ms: Capture-clock time of the first sample.
        ended_at_ms: Capture-clock time of the last sample.
        language: Expected spoken language, or ``None`` to auto-detect.

    Raises:
        ValueError: If the buffer is invalid or the time range is reversed.
    """

    id: UtteranceId
    pcm: AudioBuffer
    sample_rate: SampleRate
    started_at_ms: float
    ended_at_ms: float
    language: LanguageCode | None = None

    def __post_init__(self) -> None:
        _validate_mono_float32(self.pcm, "Utterance.pcm")
        if self.pcm.size == 0:
            msg = "Utterance.pcm must contain at least one sample"
            raise ValueError(msg)
        if self.ended_at_ms < self.started_at_ms:
            msg = (
                f"Utterance {self.id} ends before it starts "
                f"({self.ended_at_ms} < {self.started_at_ms})"
            )
            raise ValueError(msg)

    @property
    def duration_ms(self) -> float:
        """Audio length in milliseconds, derived from the sample count."""
        return self.sample_rate.ms_for_samples(self.pcm.size)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A timestamped fragment of a transcript.

    Args:
        text: Recognised text for this fragment.
        start_ms: Offset from the start of the utterance.
        end_ms: End offset from the start of the utterance.
        confidence: Model confidence for this fragment.

    Raises:
        ValueError: If the time range is reversed.
    """

    text: str
    start_ms: float
    end_ms: float
    confidence: Confidence

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            msg = f"Segment ends before it starts ({self.end_ms} < {self.start_ms})"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Transcript:
    """Speech-to-text output for an utterance.

    Args:
        utterance_id: Utterance this transcript describes.
        text: Full recognised text.
        language: Language that was recognised (detected or configured).
        confidence: Aggregate confidence across segments.
        is_final: ``False`` for interim results shown live in the UI, ``True``
            for the stable result that is forwarded to translation.
        segments: Timestamped fragments, empty when the model does not
            provide word or segment timings.
        model_id: Identifier of the model that produced this, for metrics.
        latency_ms: Time spent inside the recogniser.
    """

    utterance_id: UtteranceId
    text: str
    language: LanguageCode
    confidence: Confidence
    is_final: bool
    segments: tuple[TranscriptSegment, ...] = ()
    model_id: str = ""
    latency_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Whether the recogniser returned nothing usable (silence, noise)."""
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class Translation:
    """Machine translation output for one transcript.

    Args:
        utterance_id: Utterance this translation belongs to.
        source_text: Text that was translated.
        translated_text: Result in the target language.
        pair: Direction that was translated.
        model_id: Identifier of the translation model, for metrics.
        from_cache: ``True`` when served from the translation cache, which
            makes cache hit rate directly measurable on the Performance page.
        latency_ms: Time spent inside the translator.
    """

    utterance_id: UtteranceId
    source_text: str
    translated_text: str
    pair: LanguagePair
    model_id: str = ""
    from_cache: bool = False
    latency_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Whether the translator returned nothing usable."""
        return not self.translated_text.strip()


# --------------------------------------------------------------------------
# Synthesised speech
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class SpeechAudio:
    """Synthesised speech ready to be written to an audio sink.

    Args:
        utterance_id: Utterance this audio was generated for.
        pcm: Mono float32 samples in ``[-1.0, 1.0]``.
        sample_rate: Rate of ``pcm``, resampled at the sink boundary.
        language: Language that was spoken.
        voice_id: Voice used, for reproducibility and metrics.
        chunk_index: Position in a streamed sequence, starting at zero.
        is_last: ``True`` for the final chunk of an utterance, letting the
            sink know when it may drain and stop.
        latency_ms: Time spent inside the synthesizer for this chunk.

    Raises:
        ValueError: If the buffer is not mono float32.
    """

    utterance_id: UtteranceId
    pcm: AudioBuffer
    sample_rate: SampleRate
    language: LanguageCode
    voice_id: str = ""
    chunk_index: int = 0
    is_last: bool = True
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        _validate_mono_float32(self.pcm, "SpeechAudio.pcm")
        if self.chunk_index < 0:
            msg = f"chunk_index cannot be negative, got {self.chunk_index}"
            raise ValueError(msg)

    @property
    def duration_ms(self) -> float:
        """Length of the synthesised audio in milliseconds."""
        return self.sample_rate.ms_for_samples(self.pcm.size)


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    """A voice offered by a text-to-speech provider.

    Args:
        id: Provider-specific voice identifier.
        name: Human-readable name shown in the UI.
        language: Language this voice speaks.
        gender: Free-form descriptor such as ``"female"`` or ``"neutral"``.
        provider: Provider that owns the voice, e.g. ``"piper"``.
        sample_rate: Native output rate of the voice.
    """

    id: str
    name: str
    language: LanguageCode
    gender: str
    provider: str
    sample_rate: SampleRate


# --------------------------------------------------------------------------
# Devices, models and hardware
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """An audio endpoint reported by the operating system.

    Args:
        index: Driver-assigned index used to open the device.
        name: Human-readable device name.
        kind: Whether this is a capture or playback endpoint.
        max_channels: Highest channel count the device supports.
        default_sample_rate: Rate the driver prefers.
        host_api: Windows audio backend, e.g. ``"Windows WASAPI"``.
        is_default: Whether this is the system default for its direction.
    """

    index: int
    name: str
    kind: DeviceKind
    max_channels: int
    default_sample_rate: float
    host_api: str = ""
    is_default: bool = False

    @property
    def is_virtual_cable(self) -> bool:
        """Heuristic: does this endpoint look like a virtual audio cable?

        Used by the Devices page to highlight the endpoint that should be
        selected as the interpreter's output, and by the environment check to
        report whether VB-CABLE is installed.
        """
        haystack = self.name.casefold()
        return any(marker in haystack for marker in ("cable", "voicemeeter", "virtual"))


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """A GPU detected on the machine.

    Args:
        name: Adapter name as reported by the vendor tool.
        total_memory_mb: Total video memory in megabytes.
        vendor: Vendor identifier, e.g. ``"nvidia"``.
        driver_version: Installed driver version, empty when unknown.
    """

    name: str
    total_memory_mb: int
    vendor: str
    driver_version: str = ""

    @property
    def total_memory_gb(self) -> float:
        """Video memory in gigabytes."""
        return self.total_memory_mb / 1024.0


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Snapshot of the machine, taken once at startup.

    Drives automatic profile selection, and is printed by ``--check`` so a bug
    report always states what the application actually ran on.

    Args:
        os_name: Operating system name.
        os_version: Operating system version string.
        cpu_name: Processor model name.
        physical_cores: Physical core count.
        logical_cores: Logical processor count including hyper-threading.
        total_ram_gb: Installed memory in gigabytes.
        available_ram_gb: Currently free memory in gigabytes.
        free_disk_gb: Free space on the drive holding the project.
        python_version: Running interpreter version.
        gpus: Discrete GPUs detected, empty on integrated-only machines.
    """

    os_name: str
    os_version: str
    cpu_name: str
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    free_disk_gb: float
    python_version: str
    gpus: tuple[GpuInfo, ...] = ()

    @property
    def has_cuda_gpu(self) -> bool:
        """Whether an NVIDIA GPU capable of running CUDA models is present."""
        return any(gpu.vendor == "nvidia" for gpu in self.gpus)

    @property
    def cuda_memory_gb(self) -> float:
        """Video memory of the largest NVIDIA GPU, or ``0.0`` if none."""
        nvidia = [gpu for gpu in self.gpus if gpu.vendor == "nvidia"]
        if not nvidia:
            return 0.0
        return max(gpu.total_memory_gb for gpu in nvidia)


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Identity and provenance of a downloadable model.

    Recording the exact ``revision`` rather than tracking a moving branch is a
    supply-chain safeguard: an upstream repository cannot silently change the
    weights this application runs.

    Args:
        id: Stable local identifier, e.g. ``"whisper-small-int8"``.
        task: Pipeline stage the model serves: ``stt``, ``mt``, ``tts``, ``vad``.
        repo_id: Hugging Face repository identifier.
        revision: Commit hash or tag pinned for reproducibility.
        languages: Language codes the model supports.
        size_mb: Approximate on-disk size, shown before download.
        runtime: Runtime that executes it, e.g. ``"ctranslate2"``, ``"onnx"``.
        files: Repository-relative files required, e.g. ``("onnx/model.onnx",)``.
            Empty means the whole repository snapshot is needed.
        local_path: Resolved path once downloaded, ``None`` beforehand.
    """

    id: str
    task: str
    repo_id: str
    revision: str
    languages: tuple[str, ...]
    size_mb: int
    runtime: str
    files: tuple[str, ...] = ()
    local_path: Path | None = field(default=None)

    def supports(self, language: LanguageCode) -> bool:
        """Whether this model handles a language.

        Args:
            language: Language to check.

        Returns:
            ``True`` when the code is listed, or when the model declares
            ``"*"`` for multilingual support.
        """
        return "*" in self.languages or language.code in self.languages
