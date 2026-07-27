"""Immutable value objects.

A value object has no identity: two ``LanguageCode("ta")`` instances are the
same thing. They are frozen dataclasses so they can be shared freely across
threads without defensive copying, and they validate themselves on
construction, so an invalid value cannot exist anywhere in the system.

This is the "parse, don't validate" principle: once you hold a
:class:`SampleRate`, you never have to check it again.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = [
    "SUPPORTED_LANGUAGES",
    "ComputeDevice",
    "Confidence",
    "DeviceKind",
    "LanguageCode",
    "LanguagePair",
    "SampleRate",
    "StageTiming",
]


# --------------------------------------------------------------------------
# Languages
# --------------------------------------------------------------------------
# ISO 639-1 code -> (English name, native name).
# Adding a language here is the only change required to make the rest of the
# system aware of it; the actual capability then depends on which adapters
# report support for it.
SUPPORTED_LANGUAGES: Final[dict[str, tuple[str, str]]] = {
    "ta": ("Tamil", "தமிழ்"),
    "en": ("English", "English"),
    "hi": ("Hindi", "हिन्दी"),
    "te": ("Telugu", "తెలుగు"),
    "ml": ("Malayalam", "മലയാളം"),
    "kn": ("Kannada", "ಕನ್ನಡ"),
    "bn": ("Bengali", "বাংলা"),
    "mr": ("Marathi", "मराठी"),
    "gu": ("Gujarati", "ગુજરાતી"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ"),
    "ur": ("Urdu", "اردو"),
}


@dataclass(frozen=True, slots=True)
class LanguageCode:
    """An ISO 639-1 language code known to the application.

    Args:
        code: Two-letter lowercase ISO 639-1 code, e.g. ``"ta"``.

    Raises:
        ValueError: If the code is not present in :data:`SUPPORTED_LANGUAGES`.
    """

    code: str

    def __post_init__(self) -> None:
        normalised = self.code.strip().lower()
        if normalised not in SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            msg = f"Unsupported language code {self.code!r}. Supported: {supported}"
            raise ValueError(msg)
        # Frozen dataclasses forbid normal assignment, so write through
        # object.__setattr__ to store the normalised form.
        object.__setattr__(self, "code", normalised)

    @property
    def english_name(self) -> str:
        """Language name in English, e.g. ``"Tamil"``."""
        return SUPPORTED_LANGUAGES[self.code][0]

    @property
    def native_name(self) -> str:
        """Language name in its own script, e.g. ``"தமிழ்"``."""
        return SUPPORTED_LANGUAGES[self.code][1]

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class LanguagePair:
    """A translation direction, e.g. Tamil to English.

    Args:
        source: Language spoken by the person.
        target: Language the meeting should hear.

    Raises:
        ValueError: If source and target are the same language.
    """

    source: LanguageCode
    target: LanguageCode

    def __post_init__(self) -> None:
        if self.source == self.target:
            msg = f"Source and target languages must differ (both are {self.source})"
            raise ValueError(msg)

    @classmethod
    def of(cls, source: str, target: str) -> LanguagePair:
        """Build a pair from two raw ISO codes.

        Args:
            source: Source language code.
            target: Target language code.

        Returns:
            The validated language pair.
        """
        return cls(LanguageCode(source), LanguageCode(target))

    def reversed(self) -> LanguagePair:
        """Return the opposite direction, used by the reverse pipeline."""
        return LanguagePair(self.target, self.source)

    @property
    def key(self) -> str:
        """Stable identifier suitable for cache keys and metrics labels."""
        return f"{self.source.code}-{self.target.code}"

    def __str__(self) -> str:
        return f"{self.source.code}→{self.target.code}"


# --------------------------------------------------------------------------
# Audio primitives
# --------------------------------------------------------------------------
# Rates the audio stack is allowed to use. Restricting the set prevents
# obscure resampling bugs and matches what Windows audio endpoints negotiate.
_ALLOWED_SAMPLE_RATES: Final[frozenset[int]] = frozenset(
    {8000, 16000, 22050, 24000, 32000, 44100, 48000}
)


@dataclass(frozen=True, slots=True)
class SampleRate:
    """An audio sample rate in hertz.

    Args:
        hz: Sample rate, which must be one of the supported rates.

    Raises:
        ValueError: If the rate is not supported.
    """

    hz: int

    def __post_init__(self) -> None:
        if self.hz not in _ALLOWED_SAMPLE_RATES:
            allowed = ", ".join(str(rate) for rate in sorted(_ALLOWED_SAMPLE_RATES))
            msg = f"Unsupported sample rate {self.hz}. Allowed: {allowed}"
            raise ValueError(msg)

    def samples_for_ms(self, milliseconds: float) -> int:
        """Number of samples representing a duration at this rate.

        Args:
            milliseconds: Duration in milliseconds.

        Returns:
            Sample count, rounded to the nearest whole sample.
        """
        return round(self.hz * milliseconds / 1000.0)

    def ms_for_samples(self, samples: int) -> float:
        """Duration in milliseconds represented by a sample count.

        Args:
            samples: Number of samples.

        Returns:
            Duration in milliseconds.
        """
        return samples * 1000.0 / self.hz

    def __int__(self) -> int:
        return self.hz

    def __str__(self) -> str:
        return f"{self.hz} Hz"


@dataclass(frozen=True, slots=True)
class Confidence:
    """A model confidence score constrained to ``[0.0, 1.0]``.

    Args:
        value: Score between 0 and 1 inclusive.

    Raises:
        ValueError: If the score falls outside the valid range.
    """

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            msg = f"Confidence must be within [0.0, 1.0], got {self.value}"
            raise ValueError(msg)

    def is_below(self, threshold: float) -> bool:
        """Whether this score is under a rejection threshold.

        Args:
            threshold: Minimum acceptable confidence.

        Returns:
            ``True`` when the transcript should be suppressed rather than
            translated. Speaking a confident wrong sentence is worse than
            saying nothing.
        """
        return self.value < threshold

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{self.value:.2f}"


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------
class DeviceKind(StrEnum):
    """Direction of an audio endpoint."""

    INPUT = "input"
    OUTPUT = "output"


class ComputeDevice(StrEnum):
    """Hardware a model runs on."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Latency measurement for one pipeline stage.

    These are the raw samples aggregated by the metrics collector and shown on
    the Performance page. Recording per-stage rather than end-to-end timings is
    what makes it possible to answer "which stage should I optimise?".

    Args:
        stage: Stage name, e.g. ``"stt"``.
        duration_ms: Wall-clock duration in milliseconds.
        utterance_id: Utterance the measurement belongs to.
    """

    stage: str
    duration_ms: float
    utterance_id: str

    def __post_init__(self) -> None:
        if self.duration_ms < 0:
            msg = f"Stage duration cannot be negative, got {self.duration_ms}"
            raise ValueError(msg)
