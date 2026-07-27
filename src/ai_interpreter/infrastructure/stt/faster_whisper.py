"""Speech recognition with faster-whisper (CTranslate2).

Whisper is used rather than NVIDIA Parakeet because Parakeet is English-only,
and the required language pairs are Tamil and Hindi to English. See
``docs/phases/phase-01-architecture.md`` for that analysis.

What the measurements changed
-----------------------------

Benchmarked on the target machine (Intel i5-7200U, int8, 2 threads, warmed
up), decoding one utterance:

======  ==========  ===========================================
Model   Time        Notes
======  ==========  ===========================================
tiny    0.85 s      noticeably error-prone
base    1.70 s      correct on clear English; the cpu_low default
small   5.88 s      unusable on this CPU
======  ==========  ===========================================

Two findings from those runs shape this module.

**Decode cost is constant, not proportional to audio length.** The same model
took 0.82 s on 1 second of audio and 0.96 s on 9.9 seconds. Whisper pads its
mel spectrogram to a fixed 30-second window, so the encoder does identical
work whatever the utterance length, and the encoder dominates on CPU.

The consequence is the important one: **chunked streaming cannot reduce
end-of-utterance latency on Whisper.** Decoding a 1-second chunk costs a full
encoder pass, so re-decoding every second would multiply the work rather than
spreading it. Phase 1 assumed otherwise; the measurement says no.

Streaming is still implemented, because live captions are worth having - but
it buys *feedback during a long sentence*, not speed. That distinction is
stated wherever the option appears, so nobody enables it expecting latency to
drop. On an architecture whose cost scales with input length - Parakeet and
other Conformer models - the same interface would buy both.

**More threads made it slower.** base took 1.70 s on 2 threads and 2.16 s on
4. The CPU has two physical cores; hyperthread siblings contend for the same
execution units and cache, and CTranslate2's matrix kernels lose more to that
contention than they gain from parallelism. ``stt.cpu_threads`` should track
*physical* cores.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ai_interpreter.domain.entities import (
    AudioFrame,
    Transcript,
    TranscriptSegment,
    Utterance,
    UtteranceId,
)
from ai_interpreter.domain.errors import ModelLoadError, TranscriptionError
from ai_interpreter.domain.value_objects import Confidence, LanguageCode, SampleRate

__all__ = ["FasterWhisperRecognizer", "WhisperDecodeOptions"]

logger = logging.getLogger(__name__)

# Whisper is trained on 16 kHz mono and resamples anything else internally.
# Feeding it the right rate avoids a hidden conversion on every utterance.
_REQUIRED_SAMPLE_RATE: Final[int] = 16000

# Languages Whisper supports that this application also knows about. Whisper
# handles ~99; this is the intersection with SUPPORTED_LANGUAGES.
_SUPPORTED_LANGUAGES: Final[frozenset[str]] = frozenset(
    {"ta", "en", "hi", "te", "ml", "kn", "bn", "mr", "gu", "pa", "ur"}
)

# Below this, a transcript is treated as noise rather than speech. Speaking a
# confident wrong sentence into a meeting is worse than saying nothing.
_DEFAULT_MIN_CONFIDENCE: Final[float] = 0.0


@dataclass(frozen=True, slots=True)
class WhisperDecodeOptions:
    """Decoding parameters, chosen for real-time use.

    Args:
        beam_size: Beam width. ``1`` (greedy) is the only realistic choice
            here: beam search costs two to four times more for an accuracy
            gain that a live conversation cannot wait for.
        temperature: Sampling temperature. A single value disables Whisper's
            temperature-fallback retries, which can triple decode time on a
            difficult utterance.
        condition_on_previous_text: Feeding the previous transcript back as
            context improves coherence but lets one hallucination poison every
            following utterance in a runaway loop. Off by default.
        no_speech_threshold: Above this probability the segment is discarded
            as silence.
        log_prob_threshold: Segments below this average log probability are
            treated as failed decodes.
        word_timestamps: Per-word timings. Costs roughly 10 % more time and is
            only needed for caption highlighting.
        vad_filter: Whisper's own voice activity filter. Off, because Silero
            has already segmented the audio in Phase 3 - running a second
            detector would trim speech the first one deliberately kept.
        min_confidence: Transcripts below this confidence are returned empty.
        repetition_penalty: Discourages the decoder from repeating tokens.
            Slightly above 1.0 suppresses runaway loops at negligible cost to
            normal speech.
        compression_ratio_threshold: Segments whose text compresses better
            than this are discarded as repetitive. This is the measure
            Whisper itself uses to detect its own loops.
        min_word_diversity: Fraction of distinct words a transcript must have.
            A backstop for loops that slip past the compression check.
    """

    beam_size: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    word_timestamps: bool = False
    vad_filter: bool = False
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE
    repetition_penalty: float = 1.1
    compression_ratio_threshold: float = 2.4
    min_word_diversity: float = 0.4


# Below this word count, a low diversity ratio is normal rather than a loop:
# "yes yes" and "no no no" are things people actually say.
_MIN_WORDS_FOR_DIVERSITY_CHECK: Final[int] = 6


def is_repetitive(text: str, min_word_diversity: float) -> bool:
    """Detect a decoder repetition loop.

    Whisper sometimes falls into repeating one phrase for the whole output -
    ``"பண்டும் பண்டும் பண்டும் ..."`` thirty-two times. Measured on the target
    machine, those loops scored confidence **0.91 and 0.93** while correct
    transcripts of the same session scored 0.46 to 0.67.

    That inversion is the reason this check exists and the reason confidence
    alone must never be used as a quality gate: a loop is *more* certain by
    construction, because every repeated token is trivially predictable from
    the one before it. The loops also took 3.7 s to decode against a normal
    1.7 s, so catching them helps latency as well as correctness.

    Args:
        text: Transcript text.
        min_word_diversity: Minimum fraction of distinct words. A loop of one
            word repeated thirty-two times scores 0.03; ordinary speech scores
            well above 0.5.

    Returns:
        ``True`` when the text looks like a loop rather than speech.
    """
    words = text.split()
    if len(words) < _MIN_WORDS_FOR_DIVERSITY_CHECK:
        return False
    return (len(set(words)) / len(words)) < min_word_diversity


def _confidence_from_log_prob(avg_log_prob: float) -> float:
    """Convert an average token log probability into a ``[0, 1]`` confidence.

    ``exp(avg_log_prob)`` is the geometric mean of the per-token
    probabilities, which is a defensible and monotonic reading of the model's
    own certainty. It is a *relative* signal for thresholding, not a
    calibrated probability that the transcript is correct.

    Args:
        avg_log_prob: Mean log probability over the segment's tokens.

    Returns:
        Confidence in ``[0.0, 1.0]``.
    """
    if not math.isfinite(avg_log_prob):
        return 0.0
    return float(min(1.0, max(0.0, math.exp(avg_log_prob))))


class FasterWhisperRecognizer:
    """Whisper speech recognition, satisfying the ``SpeechRecognizer`` port.

    Args:
        model_dir: Directory holding the CTranslate2 model files.
        model_id: Identifier recorded on transcripts, for metrics.
        device: ``"cpu"`` or ``"cuda"``.
        compute_type: CTranslate2 quantisation, e.g. ``"int8"``, ``"float16"``.
        cpu_threads: Threads for CPU inference. Should match *physical* cores;
            see the module docstring.
        language: Language to decode, or ``None`` to auto-detect. Passing the
            known language is both faster and more reliable than detection.
        options: Decoding parameters.
        supported_languages: Language codes this checkpoint handles, or
            ``None`` for the full multilingual set. Fine-tuned checkpoints
            are usually single-language, and asking one for another language
            produces confident nonsense rather than an error - so the
            restriction is declared in the model registry and enforced here.
    """

    def __init__(
        self,
        model_dir: Path,
        model_id: str,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 2,
        language: LanguageCode | None = None,
        options: WhisperDecodeOptions | None = None,
        supported_languages: frozenset[str] | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._model_id = model_id
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._language = language
        self._options = options or WhisperDecodeOptions()
        self._supported = supported_languages or _SUPPORTED_LANGUAGES
        self._model: Any = None
        self._utterances_decoded = 0
        self._total_decode_ms = 0.0

    # -- port interface ----------------------------------------------------
    @property
    def model_id(self) -> str:
        """Identifier of the loaded model."""
        return self._model_id

    @property
    def language(self) -> LanguageCode | None:
        """Language being decoded, or ``None`` for auto-detection."""
        return self._language

    @property
    def utterances_decoded(self) -> int:
        """Utterances transcribed since the recogniser was created."""
        return self._utterances_decoded

    @property
    def mean_decode_ms(self) -> float:
        """Mean decode time so far, in milliseconds."""
        if not self._utterances_decoded:
            return 0.0
        return self._total_decode_ms / self._utterances_decoded

    @property
    def supported_languages(self) -> frozenset[str]:
        """Language codes this checkpoint handles."""
        return self._supported

    def supports(self, language: LanguageCode) -> bool:
        """Whether this recogniser handles a language.

        Args:
            language: Language to check.

        Returns:
            ``True`` when this checkpoint supports it. A Tamil fine-tune
            returns ``False`` for English, even though the underlying
            architecture is multilingual, because the fine-tune has lost that
            ability and would return confident nonsense.
        """
        return language.code in self._supported

    def warmup(self) -> None:
        """Load the model and run one throwaway decode.

        The first decode allocates buffers and initialises kernels, and is
        several times slower than the rest. Paying that during startup rather
        than on the user's first sentence is the difference between a clipped
        first response and a clean one.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        self._ensure_model()
        started = time.perf_counter()

        # Warm up through the same path real decoding takes. Calling
        # model.transcribe() directly here was a real bug: it skipped this
        # class's decode options, so faster-whisper's default temperature
        # fallback applied - up to six decoding passes - and warmup on silence
        # took 56 seconds instead of under two.
        silence = np.zeros(_REQUIRED_SAMPLE_RATE // 2, dtype=np.float32)
        try:
            self._decode(silence, self._language or LanguageCode("en"))
        except TranscriptionError as exc:
            msg = f"Whisper warmup failed for {self._model_id}: {exc}"
            raise ModelLoadError(msg) from exc

        logger.info(
            "Whisper %s warmed up in %.0f ms (%s, %s, %d threads)",
            self._model_id,
            (time.perf_counter() - started) * 1000.0,
            self._device,
            self._compute_type,
            self._cpu_threads,
        )

    def transcribe(self, utterance: Utterance) -> Transcript:
        """Produce the final transcript for a complete utterance.

        Args:
            utterance: Audio to transcribe, at 16 kHz mono.

        Returns:
            A transcript with ``is_final`` set to ``True``. Empty text is a
            valid result: it means the audio contained no recognisable speech.

        Raises:
            TranscriptionError: If decoding fails or the rate is wrong.
        """
        if utterance.sample_rate.hz != _REQUIRED_SAMPLE_RATE:
            msg = (
                f"Whisper requires {_REQUIRED_SAMPLE_RATE} Hz audio, "
                f"got {utterance.sample_rate.hz} Hz for utterance {utterance.id}"
            )
            raise TranscriptionError(msg)

        language = utterance.language or self._language
        if language is not None and not self.supports(language):
            supported = ", ".join(sorted(self._supported))
            msg = (
                f"Model {self._model_id} does not support {language.english_name} "
                f"({language.code}). It supports: {supported}. Choose a different "
                "stt.model, or a language this checkpoint was trained on."
            )
            raise TranscriptionError(msg)

        started = time.perf_counter()
        segments, detected = self._decode(utterance.pcm, language)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self._utterances_decoded += 1
        self._total_decode_ms += elapsed_ms

        return self._build_transcript(
            utterance_id=utterance.id,
            segments=segments,
            detected_language=detected,
            fallback_language=language,
            latency_ms=elapsed_ms,
            is_final=True,
        )

    def transcribe_stream(
        self,
        frames: Iterator[AudioFrame],
        language: LanguageCode | None = None,
    ) -> Iterator[Transcript]:
        """Consume frames and yield progressively refined transcripts.

        **This provides live feedback, not lower latency.** Each interim
        decode costs a full Whisper encoder pass - the same as decoding the
        whole utterance - because the encoder input is padded to a fixed
        30-second window regardless of how much audio it contains. Interim
        results are therefore emitted on a timer while the speaker is still
        talking, where spare CPU is available, and the final decode still
        costs what it always did.

        The interface is kept because a recogniser whose cost scales with
        input length would make it a latency win too, with no change to any
        caller.

        Args:
            frames: Frames of an in-progress utterance, at 16 kHz.
            language: Language to decode, or ``None`` for the configured one.

        Yields:
            Interim transcripts followed by exactly one final transcript.

        Raises:
            TranscriptionError: If decoding fails.
        """
        chosen = language or self._language
        collected: list[NDArray[np.float32]] = []
        total_samples = 0
        next_partial_samples = self._partial_interval_samples
        utterance_id = UtteranceId("stream")

        for frame in frames:
            collected.append(frame.pcm)
            total_samples += frame.pcm.size

            if total_samples >= next_partial_samples:
                next_partial_samples = total_samples + self._partial_interval_samples
                audio = np.concatenate(collected)
                started = time.perf_counter()
                segments, detected = self._decode(audio, chosen)
                yield self._build_transcript(
                    utterance_id=utterance_id,
                    segments=segments,
                    detected_language=detected,
                    fallback_language=chosen,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    is_final=False,
                )

        audio = np.concatenate(collected) if collected else np.zeros(1, dtype=np.float32)
        started = time.perf_counter()
        segments, detected = self._decode(audio, chosen)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._utterances_decoded += 1
        self._total_decode_ms += elapsed_ms

        yield self._build_transcript(
            utterance_id=utterance_id,
            segments=segments,
            detected_language=detected,
            fallback_language=chosen,
            latency_ms=elapsed_ms,
            is_final=True,
        )

    def close(self) -> None:
        """Release the model. Safe to call more than once."""
        self._model = None

    # -- internals ---------------------------------------------------------
    @property
    def _partial_interval_samples(self) -> int:
        """Samples between interim decodes during streaming.

        Deliberately long. Each interim decode costs a full encoder pass, so a
        short interval would saturate the CPU and starve the final decode.
        """
        return _REQUIRED_SAMPLE_RATE * 3

    def _ensure_model(self) -> None:
        """Load the CTranslate2 model on first use.

        Raises:
            ModelLoadError: If the directory is missing or loading fails.
        """
        if self._model is not None:
            return

        if not self._model_dir.is_dir():
            msg = f"Whisper model directory not found: {self._model_dir}"
            raise ModelLoadError(msg)

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                str(self._model_dir),
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
                num_workers=1,
            )
        except Exception as exc:
            msg = (
                f"Could not load the Whisper model {self._model_id} from "
                f"{self._model_dir} on {self._device} with compute type "
                f"{self._compute_type}: {exc}"
            )
            raise ModelLoadError(msg) from exc

        logger.info(
            "Whisper %s loaded (%s, %s, %d threads)",
            self._model_id,
            self._device,
            self._compute_type,
            self._cpu_threads,
        )

    def _decode(
        self,
        audio: NDArray[np.float32],
        language: LanguageCode | None,
    ) -> tuple[list[Any], str | None]:
        """Run one decode pass.

        Args:
            audio: Mono float32 samples at 16 kHz.
            language: Language to decode, or ``None`` to auto-detect.

        Returns:
            The decoded segments and the detected language code.

        Raises:
            TranscriptionError: If decoding fails.
        """
        self._ensure_model()
        options = self._options

        try:
            segments, info = self._model.transcribe(
                audio,
                language=language.code if language else None,
                beam_size=options.beam_size,
                temperature=options.temperature,
                condition_on_previous_text=options.condition_on_previous_text,
                no_speech_threshold=options.no_speech_threshold,
                log_prob_threshold=options.log_prob_threshold,
                compression_ratio_threshold=options.compression_ratio_threshold,
                repetition_penalty=options.repetition_penalty,
                word_timestamps=options.word_timestamps,
                vad_filter=options.vad_filter,
            )
            # The generator is lazy: decoding only happens on iteration, so
            # the timing around this call would be meaningless without it.
            materialised = list(segments)
        except Exception as exc:
            msg = f"Whisper decoding failed with model {self._model_id}: {exc}"
            raise TranscriptionError(msg) from exc

        detected = getattr(info, "language", None)
        return materialised, detected

    def _build_transcript(
        self,
        utterance_id: UtteranceId,
        segments: Sequence[Any],
        detected_language: str | None,
        fallback_language: LanguageCode | None,
        latency_ms: float,
        is_final: bool,
    ) -> Transcript:
        """Convert Whisper segments into a domain transcript.

        Args:
            utterance_id: Utterance the transcript belongs to.
            segments: Raw segments from faster-whisper.
            detected_language: Language code Whisper reported.
            fallback_language: Language to record if detection gave nothing.
            latency_ms: Time the decode took.
            is_final: Whether this is the final result.

        Returns:
            The transcript.
        """
        domain_segments: list[TranscriptSegment] = []
        log_probs: list[float] = []

        for segment in segments:
            text = str(segment.text).strip()
            if not text:
                continue

            # Whisper's own repetition measure. A segment whose text gzips
            # better than this is a loop, whatever its confidence says.
            compression_ratio = float(getattr(segment, "compression_ratio", 0.0))
            if compression_ratio > self._options.compression_ratio_threshold:
                logger.warning(
                    "Discarding a repetitive segment (compression ratio %.2f > %.2f): %d chars",
                    compression_ratio,
                    self._options.compression_ratio_threshold,
                    len(text),
                )
                continue

            avg_log_prob = float(getattr(segment, "avg_logprob", 0.0))
            log_probs.append(avg_log_prob)
            domain_segments.append(
                TranscriptSegment(
                    text=text,
                    start_ms=float(segment.start) * 1000.0,
                    end_ms=float(segment.end) * 1000.0,
                    confidence=Confidence(_confidence_from_log_prob(avg_log_prob)),
                )
            )

        full_text = " ".join(segment.text for segment in domain_segments).strip()
        mean_log_prob = sum(log_probs) / len(log_probs) if log_probs else float("-inf")
        confidence = Confidence(_confidence_from_log_prob(mean_log_prob))

        # Backstop for loops that span segments, where no single segment
        # compresses badly enough to be caught above.
        if full_text and is_repetitive(full_text, self._options.min_word_diversity):
            logger.warning(
                "Discarding a repetitive transcript (confidence %.2f, %d words): %s",
                confidence.value,
                len(full_text.split()),
                full_text[:60],
            )
            full_text = ""
            domain_segments = []

        if full_text and confidence.is_below(self._options.min_confidence):
            logger.debug(
                "Discarding transcript below the confidence floor (%.2f < %.2f): %d characters",
                confidence.value,
                self._options.min_confidence,
                len(full_text),
            )
            full_text = ""
            domain_segments = []

        language = self._resolve_language(detected_language, fallback_language)

        return Transcript(
            utterance_id=utterance_id,
            text=full_text,
            language=language,
            confidence=confidence,
            is_final=is_final,
            segments=tuple(domain_segments),
            model_id=self._model_id,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _resolve_language(
        detected: str | None,
        fallback: LanguageCode | None,
    ) -> LanguageCode:
        """Pick the language to record on a transcript.

        Whisper can report a language this application does not model, so the
        detected value is only used when it is one we know.

        Args:
            detected: Code Whisper reported, if any.
            fallback: Configured language, if any.

        Returns:
            The language to record, defaulting to English when nothing else
            is available.
        """
        if detected and detected in _SUPPORTED_LANGUAGES:
            return LanguageCode(detected)
        if fallback is not None:
            return fallback
        return LanguageCode("en")


def frames_from_audio(
    audio: NDArray[np.float32],
    sample_rate: SampleRate,
    frame_ms: int = 32,
) -> Iterator[AudioFrame]:
    """Split a buffer into frames, for driving :meth:`transcribe_stream`.

    Args:
        audio: Mono float32 samples.
        sample_rate: Rate of the audio.
        frame_ms: Frame length in milliseconds.

    Yields:
        Frames covering the buffer, with the last zero-padded if needed.
    """
    block = sample_rate.samples_for_ms(frame_ms)
    for index, start in enumerate(range(0, audio.size, block)):
        chunk = audio[start : start + block]
        if chunk.size < block:
            padded = np.zeros(block, dtype=np.float32)
            padded[: chunk.size] = chunk
            chunk = padded
        yield AudioFrame(
            pcm=chunk,
            sample_rate=sample_rate,
            timestamp_ms=index * frame_ms,
        )
