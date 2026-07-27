"""Domain error hierarchy.

Every failure the application can produce derives from :class:`InterpreterError`.
That single root lets the pipeline distinguish "a component of ours failed in a
way we modelled" from "something genuinely unexpected happened", which are
handled very differently: the former is retried or degraded, the latter is
logged with a full traceback and surfaced to the user.

Errors live in the domain layer because the *meaning* of a failure
("this language is not supported") is domain knowledge, even though the
*cause* ("CTranslate2 raised RuntimeError") is an infrastructure detail.
"""

from __future__ import annotations

__all__ = [
    "AudioCaptureError",
    "AudioOutputError",
    "ConfigurationError",
    "DeviceError",
    "DeviceNotFoundError",
    "InterpreterError",
    "ModelDownloadError",
    "ModelLoadError",
    "PipelineError",
    "SynthesisError",
    "TranscriptionError",
    "TranslationError",
    "UnsupportedLanguageError",
    "UtteranceCancelledError",
]


class InterpreterError(Exception):
    """Base class for every error raised by AI Interpreter."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class ConfigurationError(InterpreterError):
    """Configuration is missing, malformed, or internally inconsistent.

    Raised during startup. The application must not continue with a partially
    valid configuration: silently falling back to a default is how production
    systems end up running with the wrong microphone for a week.
    """


# --------------------------------------------------------------------------
# Audio devices
# --------------------------------------------------------------------------
class DeviceError(InterpreterError):
    """An audio device could not be opened, configured, or used."""


class DeviceNotFoundError(DeviceError):
    """The configured audio device does not exist on this machine."""


class AudioCaptureError(DeviceError):
    """Capturing audio from the input device failed."""


class AudioOutputError(DeviceError):
    """Writing audio to the output device (e.g. the virtual cable) failed."""


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class ModelDownloadError(InterpreterError):
    """A model could not be downloaded, or failed its integrity check."""


class ModelLoadError(InterpreterError):
    """A model could not be loaded into memory or onto the compute device."""


# --------------------------------------------------------------------------
# Processing stages
# --------------------------------------------------------------------------
class TranscriptionError(InterpreterError):
    """Speech-to-text failed for an utterance."""


class TranslationError(InterpreterError):
    """Machine translation failed for a piece of text."""


class SynthesisError(InterpreterError):
    """Text-to-speech failed for a piece of text."""


class UnsupportedLanguageError(InterpreterError):
    """A component was asked to handle a language it does not support.

    This is the error that makes the Tamil text-to-speech gap explicit rather
    than silent: a synthesizer without a Tamil voice raises this, and the
    pipeline degrades to on-screen captions instead of emitting nothing.
    """


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class PipelineError(InterpreterError):
    """The processing pipeline entered an invalid state."""


class UtteranceCancelledError(InterpreterError):
    """Processing was cancelled before it completed.

    Normal and expected: raised when the speaker barges in and the in-flight
    utterance is abandoned. Callers treat this as control flow, not a fault.
    """
