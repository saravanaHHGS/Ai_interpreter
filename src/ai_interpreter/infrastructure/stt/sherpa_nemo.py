"""Speech recognition with sherpa-onnx NeMo conformer models.

Why these exist alongside Whisper
---------------------------------

Whisper pads its encoder to a fixed 30-second window, so on the target CPU a
partial decode costs the same as a full one - 1.66 s for ``base`` and 6-10 s
for the Tamil fine-tune. CTC conformers cost time *proportional to the audio
they are given*, which is what makes streaming possible on 2 cores. Measured
on the target machine against a written Tamil reference:

======================  ==========  ==========================================
Model                   RTF          Notes
======================  ==========  ==========================================
IndicConformer-ta int8  **0.55**    ~0-3 % word error - better than the
                                    Whisper Tamil fine-tune, at 2-3x the speed
NeMo streaming en int8  **0.11**    genuinely incremental partial results
======================  ==========  ==========================================

Two adapters:

:class:`SherpaNemoCtcRecognizer`
    Offline CTC (the Tamil IndicConformer). ``transcribe`` decodes a whole
    utterance; ``transcribe_stream`` decodes **incrementally in committed
    chunks** using the model's own token timestamps, so that when the speaker
    stops, only the final chunk remains - a tail of roughly
    ``chunk_seconds x RTF`` instead of the whole utterance.
:class:`SherpaNemoStreamingRecognizer`
    Online CTC (the NeMo streaming fast-conformer). The model itself is
    stateful and incremental; the adapter is a thin wrapper.

A note on confidence: sherpa-onnx's greedy CTC decoding does not expose token
probabilities through its Python API. Transcripts report a confidence of 1.0
when text was produced and 0.0 when it was not - an *availability* flag, not a
model-certainty estimate. The Whisper adapters remain the only source of real
confidence values, and Phase 4 showed how little even those can be trusted.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
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

__all__ = ["SherpaNemoCtcRecognizer", "SherpaNemoStreamingRecognizer"]

logger = logging.getLogger(__name__)

_REQUIRED_SAMPLE_RATE: Final[int] = 16000

# BPE tokens beginning a new word carry the sentencepiece marker.
_WORD_MARKER: Final[str] = "▁"  # '▁'


def _join_tokens(tokens: Sequence[str]) -> str:
    """Join BPE tokens back into readable text.

    Args:
        tokens: Sentencepiece tokens, e.g. ``["▁என்", "▁பெ", "யர்"]``.

    Returns:
        The joined text with word markers converted to spaces.
    """
    return "".join(tokens).replace(_WORD_MARKER, " ").strip()


def _availability_confidence(text: str) -> Confidence:
    """Confidence stand-in for a runtime that reports none.

    Args:
        text: Recognised text.

    Returns:
        1.0 for non-empty text, 0.0 otherwise. See the module docstring.
    """
    return Confidence(1.0 if text.strip() else 0.0)


class SherpaNemoCtcRecognizer:
    """Offline NeMo CTC recognition, satisfying both recogniser ports.

    Args:
        model_path: The ONNX model, already carrying sherpa metadata.
        tokens_path: The token table (``token id`` per line).
        model_id: Identifier recorded on transcripts.
        language: The single language this checkpoint was trained on.
        cpu_threads: Inference threads; physical cores, per Phase 4.
        chunk_seconds: Audio accumulated before an incremental decode.
        commit_margin_seconds: Trailing audio excluded from commitment. CTC
            output near the right edge of a window has not seen its right
            context yet and is still allowed to change; committing it would
            bake in errors. 0.8 s covers the conformer's receptive field.
        max_buffer_seconds: Force-commit ceiling for speech with no word
            boundary in the committable region (rare, but a speaker who never
            pauses must not grow the buffer without bound).
    """

    def __init__(
        self,
        model_path: Path,
        tokens_path: Path,
        model_id: str,
        language: LanguageCode,
        cpu_threads: int = 2,
        chunk_seconds: float = 4.0,
        commit_margin_seconds: float = 0.8,
        max_buffer_seconds: float = 12.0,
    ) -> None:
        self._model_path = model_path
        self._tokens_path = tokens_path
        self._model_id = model_id
        self._language = language
        self._cpu_threads = cpu_threads
        self._chunk_s = chunk_seconds
        self._margin_s = commit_margin_seconds
        self._max_buffer_s = max_buffer_seconds
        self._engine: Any = None
        self._utterances_decoded = 0
        self._total_decode_ms = 0.0

    # -- port interface ----------------------------------------------------
    @property
    def model_id(self) -> str:
        """Identifier of the loaded model."""
        return self._model_id

    @property
    def language(self) -> LanguageCode:
        """The single language this checkpoint serves."""
        return self._language

    @property
    def utterances_decoded(self) -> int:
        """Final transcripts produced since creation."""
        return self._utterances_decoded

    @property
    def mean_decode_ms(self) -> float:
        """Mean final-decode time so far, in milliseconds."""
        if not self._utterances_decoded:
            return 0.0
        return self._total_decode_ms / self._utterances_decoded

    def supports(self, language: LanguageCode) -> bool:
        """Whether this recogniser handles a language.

        Args:
            language: Language to check.

        Returns:
            ``True`` only for the language the checkpoint was trained on.
        """
        return language == self._language

    def warmup(self) -> None:
        """Load the model and prime it with a short silent decode.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        self._ensure_engine()
        started = time.perf_counter()
        self._decode(np.zeros(_REQUIRED_SAMPLE_RATE // 2, dtype=np.float32))
        logger.info(
            "%s warmed up in %.0f ms (%d threads)",
            self._model_id,
            (time.perf_counter() - started) * 1000.0,
            self._cpu_threads,
        )

    def transcribe(self, utterance: Utterance) -> Transcript:
        """Decode a complete utterance.

        Args:
            utterance: Audio at 16 kHz mono.

        Returns:
            The final transcript.

        Raises:
            TranscriptionError: If the rate is wrong, the language is not
                served, or decoding fails.
        """
        self._require_language(utterance.language)
        self._require_rate(utterance.sample_rate)

        started = time.perf_counter()
        text, tokens, stamps = self._decode(utterance.pcm)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._utterances_decoded += 1
        self._total_decode_ms += elapsed_ms

        return self._build_transcript(utterance.id, text, tokens, stamps, elapsed_ms, is_final=True)

    def transcribe_stream(
        self,
        frames: Iterator[AudioFrame],
        language: LanguageCode | None = None,
    ) -> Iterator[Transcript]:
        """Decode incrementally, committing text as its audio becomes stable.

        This is the mechanism that turns a linear-cost model into low
        end-of-utterance latency. Audio is decoded in windows; tokens whose
        timestamps fall safely before the window's right edge are *committed*
        and their audio discarded, so each decode covers roughly
        ``chunk_seconds`` rather than the whole utterance. When the speaker
        stops, only the final uncommitted window remains - a tail of about
        ``chunk_seconds x RTF`` (~2 s of decode for a 4 s chunk at RTF 0.55)
        rather than the 6-10 s a whole-utterance Whisper decode costs.

        Commitment happens only at word boundaries (the sentencepiece ``▁``
        marker), because committing half a Tamil word would split it across
        two partials and corrupt the final join.

        Args:
            frames: Frames of an in-progress utterance at 16 kHz.
            language: Must be the checkpoint's language when given.

        Yields:
            Interim transcripts after each incremental decode, then exactly
            one final transcript.

        Raises:
            TranscriptionError: If the language or sample rate is wrong, or
                decoding fails.
        """
        self._require_language(language)

        utterance_id = UtteranceId("stream")
        committed: list[str] = []
        buffer = np.empty(0, dtype=np.float32)
        chunk_samples = int(self._chunk_s * _REQUIRED_SAMPLE_RATE)
        margin_samples = int(self._margin_s * _REQUIRED_SAMPLE_RATE)
        # Once the buffer passes the threshold, decoding on *every* incoming
        # frame would burn CPU for near-identical results. A decode is only
        # worthwhile after meaningfully more audio has arrived.
        growth_samples = _REQUIRED_SAMPLE_RATE // 2
        last_decoded_size = 0
        started = time.perf_counter()

        for frame in frames:
            self._require_rate(frame.sample_rate)
            buffer = np.concatenate((buffer, frame.pcm))
            if buffer.size < chunk_samples + margin_samples:
                continue
            if buffer.size < last_decoded_size + growth_samples:
                continue

            text, tokens, stamps = self._decode(buffer)
            limit_s = (buffer.size - margin_samples) / _REQUIRED_SAMPLE_RATE
            cut = self._commit_index(tokens, stamps, limit_s)

            if cut is not None:
                piece = _join_tokens(tokens[:cut])
                if piece:
                    committed.append(piece)
                consumed = int(stamps[cut] * _REQUIRED_SAMPLE_RATE)
                buffer = buffer[consumed:]
                tail = _join_tokens(tokens[cut:])
            elif buffer.size > self._max_buffer_s * _REQUIRED_SAMPLE_RATE:
                # No boundary found and the buffer is at its ceiling: commit
                # everything rather than grow without bound.
                if text:
                    committed.append(text)
                buffer = np.empty(0, dtype=np.float32)
                tail = ""
            else:
                tail = text
            last_decoded_size = buffer.size

            partial = " ".join((*committed, tail)).strip()
            yield self._partial_transcript(utterance_id, partial, started)

        if buffer.size:
            text, _, _ = self._decode(buffer)
            if text:
                committed.append(text)

        final_text = " ".join(committed).strip()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._utterances_decoded += 1
        self._total_decode_ms += elapsed_ms
        yield Transcript(
            utterance_id=utterance_id,
            text=final_text,
            language=self._language,
            confidence=_availability_confidence(final_text),
            is_final=True,
            model_id=self._model_id,
            latency_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Release the model. Safe to call more than once."""
        self._engine = None

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _commit_index(
        tokens: Sequence[str],
        stamps: Sequence[float],
        limit_s: float,
    ) -> int | None:
        """Find how many tokens are safely committable.

        Args:
            tokens: Decoded BPE tokens.
            stamps: Start time of each token, in seconds.
            limit_s: Times at or beyond this are too close to the right edge.

        Returns:
            An index ``j`` such that ``tokens[:j]`` may be committed - the
            last word boundary strictly before the limit - or ``None`` when
            no whole word fits.
        """
        best: int | None = None
        for index, (token, stamp) in enumerate(zip(tokens, stamps, strict=True)):
            if stamp >= limit_s:
                break
            if token.startswith(_WORD_MARKER) and index > 0:
                best = index
        return best

    def _partial_transcript(
        self, utterance_id: UtteranceId, text: str, started: float
    ) -> Transcript:
        """Build an interim transcript.

        Args:
            utterance_id: Stream identifier.
            text: Committed plus tentative text so far.
            started: Stream start time, for latency accounting.

        Returns:
            A transcript with ``is_final`` false.
        """
        return Transcript(
            utterance_id=utterance_id,
            text=text,
            language=self._language,
            confidence=_availability_confidence(text),
            is_final=False,
            model_id=self._model_id,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _build_transcript(
        self,
        utterance_id: UtteranceId,
        text: str,
        tokens: Sequence[str],
        stamps: Sequence[float],
        latency_ms: float,
        *,
        is_final: bool,
    ) -> Transcript:
        """Build a transcript with a single covering segment.

        Args:
            utterance_id: Utterance the transcript belongs to.
            text: Recognised text.
            tokens: Decoded tokens, for the segment time range.
            stamps: Token start times.
            latency_ms: Decode duration.
            is_final: Whether this is the final result.

        Returns:
            The transcript.
        """
        segments: tuple[TranscriptSegment, ...] = ()
        if text and stamps:
            segments = (
                TranscriptSegment(
                    text=text,
                    start_ms=stamps[0] * 1000.0,
                    end_ms=stamps[-1] * 1000.0,
                    confidence=_availability_confidence(text),
                ),
            )
        return Transcript(
            utterance_id=utterance_id,
            text=text,
            language=self._language,
            confidence=_availability_confidence(text),
            is_final=is_final,
            segments=segments,
            model_id=self._model_id,
            latency_ms=latency_ms,
        )

    def _require_language(self, language: LanguageCode | None) -> None:
        """Reject a language this checkpoint was not trained on.

        Args:
            language: Requested language, or ``None`` for the default.

        Raises:
            TranscriptionError: If another language was requested.
        """
        if language is not None and language != self._language:
            msg = (
                f"Model {self._model_id} only supports "
                f"{self._language.english_name} ({self._language.code}); "
                f"{language.english_name} was requested."
            )
            raise TranscriptionError(msg)

    @staticmethod
    def _require_rate(rate: SampleRate) -> None:
        """Reject audio at the wrong sample rate.

        Args:
            rate: Rate of the incoming audio.

        Raises:
            TranscriptionError: If it is not 16 kHz.
        """
        if rate.hz != _REQUIRED_SAMPLE_RATE:
            msg = f"NeMo CTC models require {_REQUIRED_SAMPLE_RATE} Hz audio, got {rate.hz} Hz"
            raise TranscriptionError(msg)

    def _ensure_engine(self) -> None:
        """Create the sherpa-onnx recogniser on first use.

        Raises:
            ModelLoadError: If files are missing or loading fails.
        """
        if self._engine is not None:
            return
        for path in (self._model_path, self._tokens_path):
            if not path.is_file():
                msg = f"Model file not found: {path}"
                raise ModelLoadError(msg)
        try:
            import sherpa_onnx

            self._engine = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(self._model_path),
                tokens=str(self._tokens_path),
                num_threads=self._cpu_threads,
                sample_rate=_REQUIRED_SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                provider="cpu",
            )
        except Exception as exc:
            msg = f"Could not load {self._model_id} from {self._model_path}: {exc}"
            raise ModelLoadError(msg) from exc
        logger.info("%s loaded (%d threads)", self._model_id, self._cpu_threads)

    def _decode(self, samples: NDArray[np.float32]) -> tuple[str, list[str], list[float]]:
        """Run one decode over a buffer.

        Args:
            samples: Mono float32 samples at 16 kHz.

        Returns:
            The text, its BPE tokens, and each token's start time in seconds.

        Raises:
            TranscriptionError: If decoding fails.
        """
        self._ensure_engine()
        try:
            stream = self._engine.create_stream()
            stream.accept_waveform(_REQUIRED_SAMPLE_RATE, samples)
            self._engine.decode_stream(stream)
            result = stream.result
        except Exception as exc:
            msg = f"Decoding failed with {self._model_id}: {exc}"
            raise TranscriptionError(msg) from exc
        return (
            str(result.text).strip(),
            list(result.tokens),
            [float(stamp) for stamp in result.timestamps],
        )


class SherpaNemoStreamingRecognizer:
    """Online NeMo streaming CTC recognition.

    The model itself is stateful and incremental - it was *trained* to emit
    text with bounded right context - so unlike the offline adapter there is
    no windowing logic here at all.

    Endpoint detection is disabled: Silero VAD already segments the audio in
    Phase 3, and two competing endpointing mechanisms would fight over where
    utterances end.

    Args:
        model_path: The streaming CTC ONNX model.
        tokens_path: The token table.
        model_id: Identifier recorded on transcripts.
        language: The single language the model serves.
        cpu_threads: Inference threads.
    """

    def __init__(
        self,
        model_path: Path,
        tokens_path: Path,
        model_id: str,
        language: LanguageCode,
        cpu_threads: int = 2,
    ) -> None:
        self._model_path = model_path
        self._tokens_path = tokens_path
        self._model_id = model_id
        self._language = language
        self._cpu_threads = cpu_threads
        self._engine: Any = None
        self._utterances_decoded = 0
        self._total_decode_ms = 0.0

    # -- port interface ----------------------------------------------------
    @property
    def model_id(self) -> str:
        """Identifier of the loaded model."""
        return self._model_id

    @property
    def language(self) -> LanguageCode:
        """The single language this model serves."""
        return self._language

    @property
    def utterances_decoded(self) -> int:
        """Final transcripts produced since creation."""
        return self._utterances_decoded

    @property
    def mean_decode_ms(self) -> float:
        """Mean wall-clock decode time per utterance."""
        if not self._utterances_decoded:
            return 0.0
        return self._total_decode_ms / self._utterances_decoded

    def supports(self, language: LanguageCode) -> bool:
        """Whether this model handles a language.

        Args:
            language: Language to check.

        Returns:
            ``True`` only for the model's own language.
        """
        return language == self._language

    def warmup(self) -> None:
        """Load the model and push a short silent stream through it.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        self._ensure_engine()
        started = time.perf_counter()
        stream = self._engine.create_stream()
        stream.accept_waveform(
            _REQUIRED_SAMPLE_RATE, np.zeros(_REQUIRED_SAMPLE_RATE // 2, dtype=np.float32)
        )
        stream.input_finished()
        while self._engine.is_ready(stream):
            self._engine.decode_stream(stream)
        logger.info(
            "%s warmed up in %.0f ms (%d threads)",
            self._model_id,
            (time.perf_counter() - started) * 1000.0,
            self._cpu_threads,
        )

    def transcribe(self, utterance: Utterance) -> Transcript:
        """Decode a complete utterance by streaming it through the model.

        Args:
            utterance: Audio at 16 kHz mono.

        Returns:
            The final transcript.

        Raises:
            TranscriptionError: If the rate or language is wrong, or decoding
                fails.
        """
        if utterance.language is not None and not self.supports(utterance.language):
            msg = (
                f"Model {self._model_id} only supports {self._language.english_name}; "
                f"{utterance.language.english_name} was requested."
            )
            raise TranscriptionError(msg)
        if utterance.sample_rate.hz != _REQUIRED_SAMPLE_RATE:
            msg = (
                f"Streaming NeMo models require {_REQUIRED_SAMPLE_RATE} Hz audio, "
                f"got {utterance.sample_rate.hz} Hz"
            )
            raise TranscriptionError(msg)

        self._ensure_engine()
        started = time.perf_counter()
        try:
            stream = self._engine.create_stream()
            stream.accept_waveform(_REQUIRED_SAMPLE_RATE, utterance.pcm)
            stream.input_finished()
            while self._engine.is_ready(stream):
                self._engine.decode_stream(stream)
            text = str(self._engine.get_result(stream)).strip()
        except Exception as exc:
            msg = f"Decoding failed with {self._model_id}: {exc}"
            raise TranscriptionError(msg) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._utterances_decoded += 1
        self._total_decode_ms += elapsed_ms
        return Transcript(
            utterance_id=utterance.id,
            text=text,
            language=self._language,
            confidence=_availability_confidence(text),
            is_final=True,
            model_id=self._model_id,
            latency_ms=elapsed_ms,
        )

    def transcribe_stream(
        self,
        frames: Iterator[AudioFrame],
        language: LanguageCode | None = None,
    ) -> Iterator[Transcript]:
        """Feed frames as they arrive, yielding a partial on every change.

        Args:
            frames: Frames of an in-progress utterance at 16 kHz.
            language: Must be the model's language when given.

        Yields:
            Interim transcripts whenever the text grows, then one final.

        Raises:
            TranscriptionError: If the language or rate is wrong, or decoding
                fails.
        """
        if language is not None and not self.supports(language):
            msg = (
                f"Model {self._model_id} only supports {self._language.english_name}; "
                f"{language.english_name} was requested."
            )
            raise TranscriptionError(msg)

        self._ensure_engine()
        utterance_id = UtteranceId("stream")
        started = time.perf_counter()
        try:
            stream = self._engine.create_stream()
            previous = ""
            for frame in frames:
                if frame.sample_rate.hz != _REQUIRED_SAMPLE_RATE:
                    msg = (
                        f"Streaming NeMo models require {_REQUIRED_SAMPLE_RATE} Hz "
                        f"audio, got {frame.sample_rate.hz} Hz"
                    )
                    raise TranscriptionError(msg)
                stream.accept_waveform(_REQUIRED_SAMPLE_RATE, frame.pcm)
                while self._engine.is_ready(stream):
                    self._engine.decode_stream(stream)
                text = str(self._engine.get_result(stream)).strip()
                if text and text != previous:
                    previous = text
                    yield Transcript(
                        utterance_id=utterance_id,
                        text=text,
                        language=self._language,
                        confidence=_availability_confidence(text),
                        is_final=False,
                        model_id=self._model_id,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                    )

            stream.input_finished()
            while self._engine.is_ready(stream):
                self._engine.decode_stream(stream)
            final_text = str(self._engine.get_result(stream)).strip()
        except TranscriptionError:
            raise
        except Exception as exc:
            msg = f"Decoding failed with {self._model_id}: {exc}"
            raise TranscriptionError(msg) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._utterances_decoded += 1
        self._total_decode_ms += elapsed_ms
        yield Transcript(
            utterance_id=utterance_id,
            text=final_text,
            language=self._language,
            confidence=_availability_confidence(final_text),
            is_final=True,
            model_id=self._model_id,
            latency_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Release the model. Safe to call more than once."""
        self._engine = None

    # -- internals ---------------------------------------------------------
    def _ensure_engine(self) -> None:
        """Create the sherpa-onnx online recogniser on first use.

        Raises:
            ModelLoadError: If files are missing or loading fails.
        """
        if self._engine is not None:
            return
        for path in (self._model_path, self._tokens_path):
            if not path.is_file():
                msg = f"Model file not found: {path}"
                raise ModelLoadError(msg)
        try:
            import sherpa_onnx

            self._engine = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
                model=str(self._model_path),
                tokens=str(self._tokens_path),
                num_threads=self._cpu_threads,
                sample_rate=_REQUIRED_SAMPLE_RATE,
                feature_dim=80,
                enable_endpoint_detection=False,
                decoding_method="greedy_search",
                provider="cpu",
            )
        except Exception as exc:
            msg = f"Could not load {self._model_id} from {self._model_path}: {exc}"
            raise ModelLoadError(msg) from exc
        logger.info("%s loaded (%d threads)", self._model_id, self._cpu_threads)
