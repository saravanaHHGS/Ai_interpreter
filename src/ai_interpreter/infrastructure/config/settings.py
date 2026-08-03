"""Typed configuration schema.

Every section is a Pydantic model with two deliberate settings:

``extra="forbid"``
    An unrecognised key in YAML is an error, not silence. Writing
    ``min_silense_ms`` by mistake stops the application at startup with the
    exact key name, instead of leaving you to wonder for an hour why your
    tuning had no effect.
``frozen=True``
    Settings are immutable after load. Multiple threads read them without
    locking, and nothing can mutate configuration mid-session.

Most fields have **no default**. Defaults live in ``config/default.yaml``,
which is the single source of truth the user can read and edit. Duplicating
them in Python would create two places to change and one place to forget.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AppSection",
    "AppSettings",
    "AudioInputSection",
    "AudioOutputSection",
    "AudioProcessingSection",
    "AudioSection",
    "DropPolicy",
    "InferenceLane",
    "LanguagePairSection",
    "LogLevel",
    "LoggingSection",
    "PipelineSection",
    "PrivacySection",
    "Profile",
    "SttSection",
    "TranslationCacheSection",
    "TranslationSection",
    "TtsSection",
    "VadSection",
]

# Rates the audio stack accepts; mirrors the domain value object so a bad
# config is rejected during validation rather than at device-open time.
_ALLOWED_SAMPLE_RATES: Final[frozenset[int]] = frozenset(
    {8000, 16000, 22050, 24000, 32000, 44100, 48000}
)

# Frame sizes PortAudio handles well. Smaller means more wake-ups per second,
# which a 2-core CPU cannot afford; larger adds avoidable latency.
_ALLOWED_FRAME_MS: Final[frozenset[int]] = frozenset({10, 20, 30, 40})

_STRICT = ConfigDict(extra="forbid", frozen=True, validate_default=True)

Milliseconds = Annotated[int, Field(gt=0, le=120_000)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Profile(StrEnum):
    """Hardware tier selecting a model set from ``config/profiles``."""

    AUTO = "auto"
    CPU_LOW = "cpu_low"
    CPU_HIGH = "cpu_high"
    CUDA = "cuda"


class DropPolicy(StrEnum):
    """What a full pipeline queue does with a new item."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


class InferenceLane(StrEnum):
    """How many models may run at the same time."""

    AUTO = "auto"
    SERIAL = "serial"
    PARALLEL = "parallel"


class LogLevel(StrEnum):
    """Standard library logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
class LanguagePairSection(BaseModel):
    """Default translation direction."""

    model_config = _STRICT

    source: str = Field(min_length=2, max_length=2, description="ISO 639-1 source code")
    target: str = Field(min_length=2, max_length=2, description="ISO 639-1 target code")

    @model_validator(mode="after")
    def _reject_identical(self) -> Self:
        """Reject a pair whose source and target are the same language."""
        if self.source == self.target:
            msg = f"language_pair source and target must differ (both are {self.source!r})"
            raise ValueError(msg)
        return self


class AppSection(BaseModel):
    """Top-level application identity and defaults."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    profile: Profile
    language_pair: LanguagePairSection


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
class AudioInputSection(BaseModel):
    """Microphone capture settings."""

    model_config = _STRICT

    device: str | None
    host_api: str | None
    sample_rate: int
    channels: int = Field(ge=1, le=2)
    frame_ms: int
    gain_db: float = Field(ge=-40.0, le=40.0)

    @model_validator(mode="after")
    def _check_enumerations(self) -> Self:
        """Constrain sample rate and frame size to supported values."""
        if self.sample_rate not in _ALLOWED_SAMPLE_RATES:
            allowed = sorted(_ALLOWED_SAMPLE_RATES)
            msg = f"audio.input.sample_rate must be one of {allowed}, got {self.sample_rate}"
            raise ValueError(msg)
        if self.frame_ms not in _ALLOWED_FRAME_MS:
            allowed_frames = sorted(_ALLOWED_FRAME_MS)
            msg = f"audio.input.frame_ms must be one of {allowed_frames}, got {self.frame_ms}"
            raise ValueError(msg)
        return self


class AudioOutputSection(BaseModel):
    """Playback settings, including the virtual microphone destination."""

    model_config = _STRICT

    device: str | None
    sample_rate: int
    jitter_buffer_ms: Milliseconds
    monitor: bool
    monitor_device: str | None

    @model_validator(mode="after")
    def _check_sample_rate(self) -> Self:
        """Constrain the output rate to a supported value."""
        if self.sample_rate not in _ALLOWED_SAMPLE_RATES:
            allowed = sorted(_ALLOWED_SAMPLE_RATES)
            msg = f"audio.output.sample_rate must be one of {allowed}, got {self.sample_rate}"
            raise ValueError(msg)
        return self


class AudioProcessingSection(BaseModel):
    """Signal conditioning applied between capture and the models."""

    model_config = _STRICT

    target_sample_rate: int
    denoise: bool
    high_pass_hz: int | None = Field(default=None, ge=0, le=500)

    @model_validator(mode="after")
    def _check_sample_rate(self) -> Self:
        """Constrain the model-facing rate to a supported value."""
        if self.target_sample_rate not in _ALLOWED_SAMPLE_RATES:
            allowed = sorted(_ALLOWED_SAMPLE_RATES)
            msg = (
                f"audio.processing.target_sample_rate must be one of {allowed}, "
                f"got {self.target_sample_rate}"
            )
            raise ValueError(msg)
        return self


class AudioSection(BaseModel):
    """All audio settings."""

    model_config = _STRICT

    input: AudioInputSection
    output: AudioOutputSection
    processing: AudioProcessingSection


# --------------------------------------------------------------------------
# vad
# --------------------------------------------------------------------------
class VadSection(BaseModel):
    """Voice activity detection and utterance segmentation."""

    model_config = _STRICT

    provider: str = Field(min_length=1)
    threshold: Probability
    min_speech_ms: Milliseconds
    min_silence_ms: Milliseconds
    pre_roll_ms: Milliseconds
    max_utterance_ms: Milliseconds

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        """Reject timings that cannot produce a usable utterance."""
        if self.min_silence_ms >= self.max_utterance_ms:
            msg = (
                f"vad.min_silence_ms ({self.min_silence_ms}) must be smaller than "
                f"vad.max_utterance_ms ({self.max_utterance_ms})"
            )
            raise ValueError(msg)
        if self.min_speech_ms >= self.max_utterance_ms:
            msg = (
                f"vad.min_speech_ms ({self.min_speech_ms}) must be smaller than "
                f"vad.max_utterance_ms ({self.max_utterance_ms})"
            )
            raise ValueError(msg)
        return self


# --------------------------------------------------------------------------
# stt / translation / tts
# --------------------------------------------------------------------------
class SttSection(BaseModel):
    """Speech-to-text engine settings."""

    model_config = _STRICT

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    device: str = Field(min_length=1)
    compute_type: str = Field(min_length=1)
    cpu_threads: int = Field(ge=1, le=64)
    beam_size: int = Field(ge=1, le=10)
    streaming: bool
    chunk_ms: Milliseconds
    partial_interval_ms: Milliseconds
    download_root: str | None
    # null = use app.language_pair.source. Passing the known language is both
    # faster and more reliable than Whisper's auto-detection.
    language: str | None
    word_timestamps: bool
    min_confidence: Probability
    # ISO 639-1 code -> model name. Empty means every language uses `model`.
    language_models: dict[str, str]
    # Technical terms, product names and abbreviations you actually speak.
    # Fed to Whisper-based recognisers as a decoder-biasing prompt, so the
    # model recognises them DIRECTLY instead of drifting to dictionary words
    # ("GamePlan" instead of "game plan"). The glossary's canonical terms are
    # added automatically; list extras here.
    hotwords: list[str]
    # When a "Tamil" transcript is phonotactically mostly English (the
    # Tamil-only recogniser transliterating an English sentence), re-recognise
    # the utterance with the English model and use its text directly.
    code_switch_fallback: bool
    # Minimum flagged fraction (with at least two flagged words) to reroute.
    code_switch_min_score: Probability
    # Word-level repair for MIXED sentences: Tamil with English words inside
    # ("மேட்சிங் மட்டும் பெண்டிங் ல இருக்கு"). The flagged words are replaced
    # by the English recogniser's words from the same time window; the Tamil
    # around them is kept. Requires code_switch_fallback.
    word_fusion: bool


class TranslationCacheSection(BaseModel):
    """Translation cache behaviour."""

    model_config = _STRICT

    enabled: bool
    max_entries: int = Field(ge=0, le=1_000_000)
    persist: bool


class TranslationOnlineSection(BaseModel):
    """Opt-in cloud LLM translation (NVIDIA NIM).

    The one feature that sends meeting text off the machine, so it is
    gated twice: this flag AND an API key in ``.env``. The local engine
    always remains as an automatic fallback.
    """

    model_config = _STRICT

    enabled: bool
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    timeout_ms: Milliseconds


class TranslationSection(BaseModel):
    """Machine translation engine settings."""

    model_config = _STRICT

    provider: str = Field(min_length=1)
    # Direction key ("indic-en" / "en-indic") -> model registry identifier.
    # IndicTrans2 ships one checkpoint per direction, so this cannot be a
    # single model name.
    models: dict[str, str]
    device: str = Field(min_length=1)
    compute_type: str = Field(min_length=1)
    beam_size: int = Field(ge=1, le=10)
    max_input_chars: int = Field(ge=16, le=8192)
    cache: TranslationCacheSection
    online: TranslationOnlineSection
    # Intended term -> transcript forms that should become it. Applied to
    # transcripts before translation, recovering code-switched English and
    # technical terms the Tamil-only recogniser renders phonetically.
    glossary: dict[str, list[str]]


class TtsSection(BaseModel):
    """Text-to-speech engine settings."""

    model_config = _STRICT

    provider: str = Field(min_length=1)
    # ISO 639-1 code -> model registry identifier. One voice model per
    # language, mirroring stt.language_models: no single engine serves both
    # sides of a Tamil/English interpreter. The output sample rate comes from
    # the voice model itself, not from configuration.
    voices: dict[str, str]
    device: str = Field(min_length=1)
    speed: float = Field(ge=0.5, le=2.0)
    streaming: bool
    sentence_split: bool


# --------------------------------------------------------------------------
# pipeline / logging / privacy
# --------------------------------------------------------------------------
class PipelineSection(BaseModel):
    """Streaming pipeline scheduling and resilience."""

    model_config = _STRICT

    queue_maxsize: int = Field(ge=1, le=64)
    drop_policy: DropPolicy
    inference_lane: InferenceLane
    max_retries: int = Field(ge=0, le=5)
    retry_backoff_ms: Milliseconds
    utterance_timeout_ms: Milliseconds


class LoggingSection(BaseModel):
    """Logging destinations, levels and rotation."""

    model_config = _STRICT

    level: LogLevel
    console_level: LogLevel
    file_level: LogLevel
    directory: str = Field(min_length=1)
    max_bytes: int = Field(ge=64_000, le=1_000_000_000)
    backup_count: int = Field(ge=0, le=100)
    error_log: bool


class PrivacySection(BaseModel):
    """Controls over what the application records about a conversation."""

    model_config = _STRICT

    log_transcripts: bool
    persist_history: bool
    cache_translations: bool
    telemetry: bool

    @model_validator(mode="after")
    def _reject_telemetry(self) -> Self:
        """Refuse to start with telemetry enabled.

        No telemetry backend exists and none is planned. Rejecting the value
        outright means the setting can never be flipped on by an edited config
        file and silently do something unexpected with meeting audio.
        """
        if self.telemetry:
            msg = "privacy.telemetry must be false: this application sends no telemetry"
            raise ValueError(msg)
        return self


# --------------------------------------------------------------------------
# root
# --------------------------------------------------------------------------
class AppSettings(BaseModel):
    """Complete, validated application configuration.

    Constructed once during startup and injected into every component that
    needs it. Nothing else reads YAML.
    """

    model_config = _STRICT

    app: AppSection
    audio: AudioSection
    vad: VadSection
    stt: SttSection
    translation: TranslationSection
    tts: TtsSection
    pipeline: PipelineSection
    logging: LoggingSection
    privacy: PrivacySection

    @model_validator(mode="after")
    def _check_cross_section_consistency(self) -> Self:
        """Validate rules that span more than one section."""
        frame_samples = self.audio.input.sample_rate * self.audio.input.frame_ms / 1000
        if frame_samples != int(frame_samples):
            msg = (
                f"audio.input.frame_ms ({self.audio.input.frame_ms}) does not divide "
                f"evenly into audio.input.sample_rate ({self.audio.input.sample_rate}); "
                "this produces fractional frames and drifting timestamps"
            )
            raise ValueError(msg)

        if self.stt.chunk_ms > self.vad.max_utterance_ms:
            msg = (
                f"stt.chunk_ms ({self.stt.chunk_ms}) cannot exceed "
                f"vad.max_utterance_ms ({self.vad.max_utterance_ms})"
            )
            raise ValueError(msg)

        if self.translation.cache.enabled and not self.privacy.cache_translations:
            msg = (
                "translation.cache.enabled is true but privacy.cache_translations is "
                "false; the privacy setting must permit caching before it is enabled"
            )
            raise ValueError(msg)

        return self
