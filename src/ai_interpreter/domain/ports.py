"""Ports: the interfaces the application layer is allowed to depend on.

Every port is a :class:`typing.Protocol`, which gives *structural* typing: an
adapter satisfies a port by having the right methods, without inheriting from
anything. Two practical consequences:

* Infrastructure never imports the domain in order to subclass it, so the
  dependency arrow keeps pointing inward.
* A test fake is an ordinary class with three methods - no mock framework, no
  base class, no registration step.

Ports are split by capability rather than bundled into one fat interface
(Interface Segregation Principle). A synthesizer that cannot stream implements
:class:`SpeechSynthesizer` only; one that can also implements
:class:`StreamingSpeechSynthesizer`. The pipeline checks with ``isinstance``
and takes the fast path when available, so adding a streaming-capable Tamil
engine later requires no change anywhere else in the application.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from ai_interpreter.domain.entities import (
    AudioFrame,
    DeviceInfo,
    ModelDescriptor,
    SpeechAudio,
    Transcript,
    Translation,
    Utterance,
    VoiceInfo,
)
from ai_interpreter.domain.value_objects import (
    DeviceKind,
    LanguageCode,
    LanguagePair,
    SampleRate,
)

__all__ = [
    "AudioSink",
    "AudioSource",
    "Component",
    "DeviceEnumerator",
    "ModelRepository",
    "SessionHistoryRepository",
    "SettingsRepository",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "StreamingSpeechRecognizer",
    "StreamingSpeechSynthesizer",
    "TranslationCacheRepository",
    "Translator",
    "VoiceActivityDetector",
]


# --------------------------------------------------------------------------
# Shared lifecycle
# --------------------------------------------------------------------------
@runtime_checkable
class Component(Protocol):
    """Anything holding expensive resources (a model, a device handle).

    Separating ``warmup`` from construction matters for latency: the first
    inference of any model is several times slower than the rest because
    kernels are compiled and buffers allocated. Paying that cost during
    startup rather than on the user's first sentence is the difference between
    a 3-second and a 0.8-second first response.
    """

    def warmup(self) -> None:
        """Load resources and run a throwaway inference to prime caches."""
        ...

    def close(self) -> None:
        """Release all resources. Must be safe to call more than once."""
        ...


# --------------------------------------------------------------------------
# Audio input
# --------------------------------------------------------------------------
@runtime_checkable
class DeviceEnumerator(Protocol):
    """Discovers the audio endpoints available on this machine."""

    def list_devices(self, kind: DeviceKind | None = None) -> Sequence[DeviceInfo]:
        """List audio endpoints.

        Args:
            kind: Restrict to capture or playback endpoints, or ``None`` for all.

        Returns:
            Endpoints in driver order.
        """
        ...

    def default_device(self, kind: DeviceKind) -> DeviceInfo | None:
        """Return the system default endpoint for a direction.

        Args:
            kind: Capture or playback.

        Returns:
            The default endpoint, or ``None`` if the machine has none.
        """
        ...

    def find_device(self, name_fragment: str, kind: DeviceKind) -> DeviceInfo | None:
        """Find an endpoint by case-insensitive name fragment.

        Configuration stores device *names*, not indices, because indices are
        reassigned whenever a USB headset is unplugged.

        Args:
            name_fragment: Substring to match, e.g. ``"CABLE Input"``.
            kind: Capture or playback.

        Returns:
            The first matching endpoint, or ``None``.
        """
        ...


@runtime_checkable
class AudioSource(Protocol):
    """A source of captured audio frames.

    Implemented by the microphone adapter, the WASAPI loopback adapter used
    for reverse translation, and a WAV-file adapter that makes the whole
    pipeline testable with no hardware at all.
    """

    @property
    def device(self) -> DeviceInfo:
        """Endpoint this source reads from."""
        ...

    @property
    def sample_rate(self) -> SampleRate:
        """Rate frames are delivered at."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether capture is currently active."""
        ...

    def start(self) -> None:
        """Begin capturing. Idempotent."""
        ...

    def stop(self) -> None:
        """Stop capturing and release the device handle. Idempotent."""
        ...

    def read(self, timeout: float | None = None) -> AudioFrame | None:
        """Take the next frame from the capture buffer.

        Blocking, and therefore called from a worker thread rather than the
        event loop. The real-time driver callback only copies samples into a
        ring buffer; all other work happens on this side of the boundary.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            The next frame, or ``None`` if the timeout expired.
        """
        ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Decides whether a frame contains speech."""

    def speech_probability(self, frame: AudioFrame) -> float:
        """Probability that a frame contains speech.

        Args:
            frame: Frame to score.

        Returns:
            Probability in ``[0.0, 1.0]``.
        """
        ...

    def reset(self) -> None:
        """Clear internal state between utterances.

        Neural detectors are recurrent; leaving state from a previous speaker
        degrades the first few frames of the next one.
        """
        ...


# --------------------------------------------------------------------------
# Speech to text
# --------------------------------------------------------------------------
@runtime_checkable
class SpeechRecognizer(Protocol):
    """Converts an utterance into text."""

    @property
    def model_id(self) -> str:
        """Identifier of the loaded model, used in metrics and logs."""
        ...

    def supports(self, language: LanguageCode) -> bool:
        """Whether this recogniser handles a language.

        Args:
            language: Language to check.

        Returns:
            ``True`` when supported.
        """
        ...

    def transcribe(self, utterance: Utterance) -> Transcript:
        """Produce the final transcript for a complete utterance.

        Args:
            utterance: Audio to transcribe.

        Returns:
            A transcript with ``is_final`` set to ``True``.

        Raises:
            TranscriptionError: If recognition fails.
            UnsupportedLanguageError: If the utterance language is not handled.
        """
        ...

    def warmup(self) -> None:
        """Load the model and prime it with a throwaway inference."""
        ...

    def close(self) -> None:
        """Release the model. Must be safe to call more than once."""
        ...


@runtime_checkable
class StreamingSpeechRecognizer(SpeechRecognizer, Protocol):
    """A recogniser that emits interim results while speech is ongoing.

    This is the optimisation that brings end-of-utterance latency on a
    low-power CPU from roughly 2 seconds down to under 1: audio is decoded in
    chunks as it arrives, so when the speaker stops only the final chunk is
    still outstanding.
    """

    def transcribe_stream(
        self,
        frames: Iterator[AudioFrame],
        language: LanguageCode | None = None,
    ) -> Iterator[Transcript]:
        """Consume frames and yield progressively refined transcripts.

        Args:
            frames: Frames of an in-progress utterance.
            language: Expected language, or ``None`` to auto-detect.

        Yields:
            Interim transcripts (``is_final`` false) followed by exactly one
            final transcript (``is_final`` true).

        Raises:
            TranscriptionError: If recognition fails.
        """
        ...


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------
@runtime_checkable
class Translator(Protocol):
    """Translates text between two languages."""

    @property
    def model_id(self) -> str:
        """Identifier of the loaded model, used in metrics and logs."""
        ...

    def supports(self, pair: LanguagePair) -> bool:
        """Whether this translator handles a direction.

        Args:
            pair: Direction to check.

        Returns:
            ``True`` when supported.
        """
        ...

    def translate(self, text: str, pair: LanguagePair) -> Translation:
        """Translate a piece of text.

        Args:
            text: Source text.
            pair: Direction to translate in.

        Returns:
            The translation result.

        Raises:
            TranslationError: If translation fails.
            UnsupportedLanguageError: If the direction is not handled.
        """
        ...

    def warmup(self) -> None:
        """Load the model and prime it with a throwaway inference."""
        ...

    def close(self) -> None:
        """Release the model. Must be safe to call more than once."""
        ...


# --------------------------------------------------------------------------
# Text to speech
# --------------------------------------------------------------------------
@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Converts text into speech audio.

    Deliberately the smallest possible interface. Adding a Tamil engine later
    means writing one class with these five methods and registering it in the
    composition root: no other file changes.
    """

    @property
    def provider_id(self) -> str:
        """Identifier of the provider, e.g. ``"piper"``."""
        ...

    def supports(self, language: LanguageCode) -> bool:
        """Whether a voice exists for a language.

        Args:
            language: Language to check.

        Returns:
            ``True`` when at least one voice is available.
        """
        ...

    def voices(self, language: LanguageCode | None = None) -> Sequence[VoiceInfo]:
        """List available voices.

        Args:
            language: Restrict to one language, or ``None`` for all.

        Returns:
            The available voices.
        """
        ...

    def synthesize(
        self,
        text: str,
        language: LanguageCode,
        voice_id: str | None = None,
        speed: float = 1.0,
    ) -> SpeechAudio:
        """Synthesise a complete piece of text.

        Args:
            text: Text to speak.
            language: Language of the text.
            voice_id: Voice to use, or ``None`` for the provider default.
            speed: Speaking rate multiplier where ``1.0`` is natural pace.

        Returns:
            A single audio chunk with ``is_last`` set to ``True``.

        Raises:
            SynthesisError: If synthesis fails.
            UnsupportedLanguageError: If no voice exists for the language.
        """
        ...

    def warmup(self) -> None:
        """Load the voice and prime it with a throwaway inference."""
        ...

    def close(self) -> None:
        """Release resources. Must be safe to call more than once."""
        ...


@runtime_checkable
class StreamingSpeechSynthesizer(SpeechSynthesizer, Protocol):
    """A synthesizer that emits audio sentence by sentence.

    Lets the first sentence reach the virtual microphone while the second is
    still being generated, removing most of the synthesis cost from the
    perceived latency of a multi-sentence reply.
    """

    def synthesize_stream(
        self,
        text: str,
        language: LanguageCode,
        voice_id: str | None = None,
        speed: float = 1.0,
    ) -> Iterator[SpeechAudio]:
        """Synthesise text, yielding audio as each fragment completes.

        Args:
            text: Text to speak.
            language: Language of the text.
            voice_id: Voice to use, or ``None`` for the provider default.
            speed: Speaking rate multiplier where ``1.0`` is natural pace.

        Yields:
            Chunks in playback order; the last has ``is_last`` set to ``True``.

        Raises:
            SynthesisError: If synthesis fails.
        """
        ...


# --------------------------------------------------------------------------
# Audio output
# --------------------------------------------------------------------------
@runtime_checkable
class AudioSink(Protocol):
    """Destination for synthesised speech.

    The virtual-cable adapter implements this, and so does a WAV-writing
    adapter used in tests, which is how the pipeline can be verified
    end-to-end on a machine with no audio hardware at all.
    """

    @property
    def device(self) -> DeviceInfo:
        """Endpoint this sink writes to."""
        ...

    @property
    def is_open(self) -> bool:
        """Whether the sink is currently accepting audio."""
        ...

    def open(self) -> None:
        """Acquire the output device. Idempotent."""
        ...

    def write(self, audio: SpeechAudio) -> None:
        """Queue audio for playback, resampling if the rates differ.

        Args:
            audio: Chunk to play.

        Raises:
            AudioOutputError: If the device rejected the audio.
        """
        ...

    def flush(self, timeout: float | None = None) -> None:
        """Block until queued audio has been played.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely.
        """
        ...

    def clear(self) -> None:
        """Discard queued audio immediately.

        Called on barge-in: when the speaker interrupts, the half-played
        previous translation must stop rather than talk over them.
        """
        ...

    def close(self) -> None:
        """Release the device. Must be safe to call more than once."""
        ...


# --------------------------------------------------------------------------
# Repositories
# --------------------------------------------------------------------------
@runtime_checkable
class SettingsRepository(Protocol):
    """Persists user configuration overrides."""

    def load_overrides(self) -> dict[str, object]:
        """Read the user's saved overrides.

        Returns:
            A nested mapping, empty when the user has saved nothing.
        """
        ...

    def save_overrides(self, overrides: dict[str, object]) -> None:
        """Write the user's overrides.

        Args:
            overrides: Nested mapping to persist.

        Raises:
            ConfigurationError: If the file cannot be written.
        """
        ...


@runtime_checkable
class ModelRepository(Protocol):
    """Resolves model descriptors to files on disk."""

    def is_available(self, descriptor: ModelDescriptor) -> bool:
        """Whether a model is already downloaded and verified.

        Args:
            descriptor: Model to check.

        Returns:
            ``True`` when the local copy is present and intact.
        """
        ...

    def ensure(self, descriptor: ModelDescriptor) -> Path:
        """Download a model if needed and return its local path.

        Args:
            descriptor: Model to obtain.

        Returns:
            Directory containing the model files.

        Raises:
            ModelDownloadError: If download or verification fails.
        """
        ...

    def list_installed(self) -> Sequence[ModelDescriptor]:
        """List models present on disk.

        Returns:
            Descriptors of every locally available model.
        """
        ...


@runtime_checkable
class TranslationCacheRepository(Protocol):
    """Caches translations of repeated phrases.

    Conversational speech is highly repetitive - greetings, confirmations,
    "can you hear me" - so this turns a meaningful share of translations into
    a dictionary lookup.
    """

    def get(self, text: str, pair: LanguagePair) -> str | None:
        """Look up a cached translation.

        Args:
            text: Source text.
            pair: Direction.

        Returns:
            The cached translation, or ``None`` on a miss.
        """
        ...

    def put(self, text: str, pair: LanguagePair, translation: str) -> None:
        """Store a translation.

        Args:
            text: Source text.
            pair: Direction.
            translation: Result to cache.
        """
        ...

    def clear(self) -> None:
        """Remove every cached entry."""
        ...

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache since startup."""
        ...


@runtime_checkable
class SessionHistoryRepository(Protocol):
    """Optionally records what was said during a session.

    Disabled by default: meeting content is sensitive, so persistence is an
    explicit user choice rather than a silent side effect.
    """

    def append(self, transcript: Transcript, translation: Translation) -> None:
        """Record one completed exchange.

        Args:
            transcript: Recognised source text.
            translation: Translated text.
        """
        ...

    def recent(self, limit: int) -> Sequence[tuple[Transcript, Translation]]:
        """Return the most recent exchanges, newest first.

        Args:
            limit: Maximum number of exchanges to return.

        Returns:
            Recorded exchanges.
        """
        ...

    def clear(self) -> None:
        """Delete all recorded history."""
        ...
