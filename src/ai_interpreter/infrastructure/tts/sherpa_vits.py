"""VITS speech synthesis through sherpa-onnx.

One engine serves every voice in the project - Piper English, MMS Tamil, and
later Kokoro - because sherpa-onnx runs them all, is already installed for
speech recognition, and is proven to load under Smart App Control. The
alternative (the ``piper-tts`` package plus ``piper_phonemize``) would add
another native DLL for the policy to block.

Measured on the target machine (2 threads, warm):

==================  =========  ================================================
Voice               RTF        Notes
==================  =========  ================================================
piper-en-lessac     **0.29**   22.05 kHz; a 3 s sentence costs ~0.9 s
piper-en-amy-low    **0.18**   16 kHz; flatter voice, lowest latency
mms-tam             **1.47**   16 kHz; **slower than real time** - see below
==================  =========  ================================================

The Tamil reality, stated plainly: MMS is the only Tamil voice that runs
under constraint C6, and at RTF 1.47 it generates one second of speech in
about 1.5 seconds. Sentence-streamed playback therefore *starts* after
``first-sentence-duration x 1.47`` and cannot sustain continuous speech
without gaps. Quantisation was tried and measured worse (RTF 4.2 - this CPU
lacks VNNI, so int8 convolutions fall back to slow paths). English output -
the primary Tamil→English direction - is unaffected and fast.

An instance wraps exactly one voice model, mirroring the per-language pattern
of the recognisers; the container builds one per target language in use.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from ai_interpreter.domain.entities import SpeechAudio, UtteranceId, VoiceInfo
from ai_interpreter.domain.errors import ModelLoadError, SynthesisError
from ai_interpreter.domain.value_objects import LanguageCode, SampleRate

__all__ = ["SherpaVitsSynthesizer", "split_sentences"]

logger = logging.getLogger(__name__)

# Sentence terminators across the project's scripts: Latin punctuation plus
# the Devanagari danda family (retained in some Indic text).
_SENTENCE_END: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?।॥])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked synthesis.

    Splitting lets the first sentence reach the audio sink while later ones
    are still being generated, which removes most of the synthesis cost from
    perceived latency on multi-sentence replies.

    Args:
        text: Text to split.

    Returns:
        Non-empty sentences in order; a single-element list when no sentence
        boundary is found.
    """
    parts = [part.strip() for part in _SENTENCE_END.split(text.strip())]
    return [part for part in parts if part]


class SherpaVitsSynthesizer:
    """One VITS voice, satisfying both synthesizer ports.

    Args:
        model_path: The VITS ONNX model.
        tokens_path: Its token table.
        model_id: Registry identifier, recorded on output and shown in the UI.
        language: The single language this voice speaks.
        voice_name: Human-readable name for the Voices page.
        data_dir: Bundled ``espeak-ng-data`` directory for Piper voices, or
            ``None`` for character-frontend models such as MMS.
        cpu_threads: Inference threads; physical cores, per Phase 4.
        speed: Default speaking-rate multiplier.
        sentence_split: Whether :meth:`synthesize_stream` splits sentences.
    """

    def __init__(
        self,
        model_path: Path,
        tokens_path: Path,
        model_id: str,
        language: LanguageCode,
        voice_name: str = "",
        data_dir: Path | None = None,
        cpu_threads: int = 2,
        speed: float = 1.0,
        sentence_split: bool = True,
    ) -> None:
        self._model_path = model_path
        self._tokens_path = tokens_path
        self._model_id = model_id
        self._language = language
        self._voice_name = voice_name or model_id
        self._data_dir = data_dir
        self._cpu_threads = cpu_threads
        self._speed = speed
        self._sentence_split = sentence_split
        self._engine: Any = None
        self._warmed = False
        self._chunks_generated = 0
        self._total_ms = 0.0

    # -- port interface ----------------------------------------------------
    @property
    def provider_id(self) -> str:
        """Identifier of this voice's model."""
        return self._model_id

    @property
    def language(self) -> LanguageCode:
        """The single language this voice speaks."""
        return self._language

    @property
    def chunks_generated(self) -> int:
        """Audio chunks produced since creation."""
        return self._chunks_generated

    @property
    def mean_chunk_ms(self) -> float:
        """Mean synthesis time per chunk, in milliseconds."""
        if not self._chunks_generated:
            return 0.0
        return self._total_ms / self._chunks_generated

    def supports(self, language: LanguageCode) -> bool:
        """Whether a voice exists for a language.

        Args:
            language: Language to check.

        Returns:
            ``True`` only for this voice's own language.
        """
        return language == self._language

    def voices(self, language: LanguageCode | None = None) -> Sequence[VoiceInfo]:
        """List this synthesizer's voice.

        Args:
            language: Restrict to one language, or ``None`` for all.

        Returns:
            A single-element sequence, or empty when the language differs.
        """
        if language is not None and language != self._language:
            return ()
        self._ensure_engine()
        return (
            VoiceInfo(
                id=self._model_id,
                name=self._voice_name,
                language=self._language,
                gender="",
                provider="sherpa-vits",
                sample_rate=SampleRate(int(self._engine.sample_rate)),
            ),
        )

    def warmup(self) -> None:
        """Load the voice and run one short synthesis.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        if self._warmed and self._engine is not None:
            return
        self._ensure_engine()
        started = time.perf_counter()
        sample = "வணக்கம்" if self._language.code == "ta" else "Hello."
        self._generate(sample, self._speed)
        self._chunks_generated = 0
        self._total_ms = 0.0
        self._warmed = True
        logger.info(
            "%s warmed up in %.0f ms (%d threads)",
            self._model_id,
            (time.perf_counter() - started) * 1000.0,
            self._cpu_threads,
        )

    def synthesize(
        self,
        text: str,
        language: LanguageCode,
        voice_id: str | None = None,
        speed: float = 1.0,
    ) -> SpeechAudio:
        """Synthesise a complete piece of text as one chunk.

        Args:
            text: Text to speak.
            language: Language of the text; must be this voice's language.
            voice_id: Ignored beyond validation - an instance is one voice.
            speed: Speaking-rate multiplier, combined with the default.

        Returns:
            One audio chunk with ``is_last`` set. Empty text yields an empty
            chunk rather than an error.

        Raises:
            SynthesisError: If synthesis fails.
            ModelLoadError: If the model cannot be loaded.
        """
        self._require_language(language)
        cleaned = text.strip()
        if not cleaned:
            return self._empty_chunk()

        started = time.perf_counter()
        samples, rate = self._generate(cleaned, self._effective_speed(speed))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._chunks_generated += 1
        self._total_ms += elapsed_ms

        return SpeechAudio(
            utterance_id=UtteranceId("tts"),
            pcm=samples,
            sample_rate=rate,
            language=self._language,
            voice_id=self._model_id,
            chunk_index=0,
            is_last=True,
            latency_ms=elapsed_ms,
        )

    def synthesize_stream(
        self,
        text: str,
        language: LanguageCode,
        voice_id: str | None = None,
        speed: float = 1.0,
    ) -> Iterator[SpeechAudio]:
        """Synthesise sentence by sentence, yielding each as it completes.

        The first sentence's audio is available after only its own synthesis
        time, while later sentences are generated as earlier ones play. For
        the English voice at RTF 0.29 this makes multi-sentence replies feel
        immediate; for MMS Tamil at RTF 1.47 it shortens the wait but cannot
        eliminate inter-sentence gaps.

        Args:
            text: Text to speak.
            language: Language of the text; must be this voice's language.
            voice_id: Ignored beyond validation.
            speed: Speaking-rate multiplier, combined with the default.

        Yields:
            Chunks in playback order; the final chunk has ``is_last`` set.

        Raises:
            SynthesisError: If synthesis fails.
        """
        self._require_language(language)
        cleaned = text.strip()
        if not cleaned:
            yield self._empty_chunk()
            return

        sentences = split_sentences(cleaned) if self._sentence_split else [cleaned]
        effective = self._effective_speed(speed)

        for index, sentence in enumerate(sentences):
            started = time.perf_counter()
            samples, rate = self._generate(sentence, effective)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._chunks_generated += 1
            self._total_ms += elapsed_ms

            yield SpeechAudio(
                utterance_id=UtteranceId("tts"),
                pcm=samples,
                sample_rate=rate,
                language=self._language,
                voice_id=self._model_id,
                chunk_index=index,
                is_last=index == len(sentences) - 1,
                latency_ms=elapsed_ms,
            )

    def close(self) -> None:
        """Release the model. Safe to call more than once."""
        self._engine = None
        self._warmed = False

    # -- internals ---------------------------------------------------------
    def _effective_speed(self, speed: float) -> float:
        """Combine the per-call multiplier with the configured default.

        Args:
            speed: Per-call speaking-rate multiplier.

        Returns:
            The multiplier handed to the engine.
        """
        return self._speed * speed

    def _empty_chunk(self) -> SpeechAudio:
        """Build the zero-length chunk returned for empty input.

        Returns:
            An empty, final chunk at the voice's native rate.
        """
        self._ensure_engine()
        return SpeechAudio(
            utterance_id=UtteranceId("tts"),
            pcm=np.empty(0, dtype=np.float32),
            sample_rate=SampleRate(int(self._engine.sample_rate)),
            language=self._language,
            voice_id=self._model_id,
            chunk_index=0,
            is_last=True,
        )

    def _require_language(self, language: LanguageCode) -> None:
        """Reject a language this voice does not speak.

        Args:
            language: Requested language.

        Raises:
            SynthesisError: If it differs from the voice's language.
        """
        if language != self._language:
            msg = (
                f"Voice {self._model_id} speaks {self._language.english_name}, "
                f"not {language.english_name}. Configure tts.voices for the "
                "target language."
            )
            raise SynthesisError(msg)

    def _ensure_engine(self) -> None:
        """Create the sherpa-onnx synthesizer on first use.

        Raises:
            ModelLoadError: If files are missing or loading fails.
        """
        if self._engine is not None:
            return
        for path in (self._model_path, self._tokens_path):
            if not path.is_file():
                msg = f"Voice model file not found: {path}"
                raise ModelLoadError(msg)

        try:
            import sherpa_onnx

            vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(self._model_path),
                tokens=str(self._tokens_path),
                data_dir=str(self._data_dir) if self._data_dir else "",
            )
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=vits,
                    num_threads=self._cpu_threads,
                    provider="cpu",
                ),
                max_num_sentences=1,
            )
            self._engine = sherpa_onnx.OfflineTts(config)
        except Exception as exc:
            msg = f"Could not load voice {self._model_id} from {self._model_path}: {exc}"
            raise ModelLoadError(msg) from exc

        logger.info(
            "%s loaded (%s, %d Hz, %d threads)",
            self._model_id,
            self._language.code,
            int(self._engine.sample_rate),
            self._cpu_threads,
        )

    def _generate(self, text: str, speed: float) -> tuple[np.ndarray, SampleRate]:
        """Run one synthesis.

        Args:
            text: Text to speak.
            speed: Speaking-rate multiplier.

        Returns:
            The samples as mono float32 and their sample rate.

        Raises:
            SynthesisError: If the engine fails or produces nothing.
        """
        self._ensure_engine()
        try:
            audio = self._engine.generate(text, sid=0, speed=speed)
            samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
            rate = SampleRate(int(audio.sample_rate))
        except Exception as exc:
            msg = f"Synthesis failed with {self._model_id}: {exc}"
            raise SynthesisError(msg) from exc

        if not samples.size:
            msg = f"Voice {self._model_id} produced no audio for a {len(text)}-character input"
            raise SynthesisError(msg)
        return samples, rate
