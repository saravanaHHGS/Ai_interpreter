"""IndicTrans2 translation on CTranslate2.

Why this stack: IndicTrans2-200M is purpose-built for the 22 scheduled Indic
languages and, measured on the target machine (2 threads, int8, beam 4),
translates a sentence in **0.19-1.3 s** with quality the Phase 1 analysis
hoped for and Phase 5 verified:

    என் பெயர் சரவணகுமார்            -> My name is Saravanakumar.
    நாளைக்கு என்ன திட்டம்?           -> What's the plan for tomorrow?
    The meeting will start in five   -> ஐந்து நிமிடங்களில் கூட்டம்
    minutes.                            தொடங்கும்.

Constraint C6 (PyTorch blocked) rules out the reference implementation; the
CTranslate2 export runs on the runtime already proven for Whisper.

The model is direction-specific - one checkpoint translates Indic to English,
another English to Indic - so an instance serves the pairs of one direction
and the container builds one per direction in use.

Input format, learned the hard way (see ``transliteration.py``): NFC
normalise, transliterate Indic text to Devanagari, sentencepiece-encode, and
prefix the two language tags::

    ["tam_Taml", "eng_Latn", "▁ऎऩ्", "▁पॆयर्", ...]

CTranslate2 appends EOS itself (``add_source_eos`` in the export's config).
For English->Indic the decoder emits Devanagari, which is shifted back into
the target script afterwards.

Not implemented, deliberately: the official IndicProcessor also wraps URLs,
emails and long numbers in placeholder tokens so they survive translation
byte-for-byte. Spoken utterances rarely contain them; revisit if numeric
mangling shows up in practice.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Final

from ai_interpreter.domain.entities import Translation, UtteranceId
from ai_interpreter.domain.errors import ModelLoadError, TranslationError
from ai_interpreter.domain.value_objects import LanguagePair
from ai_interpreter.infrastructure.translation.transliteration import (
    INDICTRANS2_TAGS,
    from_devanagari,
    supports_script_mapping,
    to_devanagari,
)

__all__ = ["IndicTrans2Translator"]

logger = logging.getLogger(__name__)

_DIRECTIONS: Final[frozenset[str]] = frozenset({"indic-en", "en-indic"})

# Sentencepiece detokenisation leaves spaces around punctuation and clitics
# ("What 's the plan ?"). These rules restore ordinary typography.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:%\)\]\}])")
_SPACE_AFTER_OPEN = re.compile(r"([\(\[\{])\s+")
_SPACE_AROUND_APOSTROPHE = re.compile(r"\s+'\s*(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)


def _detokenize(text: str) -> str:
    """Tidy sentencepiece output into ordinary typography.

    Args:
        text: Decoded text with detokenisation artefacts.

    Returns:
        The cleaned text.
    """
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _SPACE_AROUND_APOSTROPHE.sub(r"'\1", text)
    return text.strip()


class IndicTrans2Translator:
    """IndicTrans2 on CTranslate2, satisfying the ``Translator`` port.

    Args:
        model_dir: Directory holding the CTranslate2 export (``model.bin``,
            vocabularies, and ``vocab/model.SRC`` / ``vocab/model.TGT``).
        model_id: Identifier recorded on translations.
        direction: ``"indic-en"`` or ``"en-indic"``.
        cpu_threads: Intra-op threads; physical cores, per Phase 4.
        beam_size: Beam width. Unlike speech recognition - where beam search
            was measured too slow for real time - a 200M text model at beam 4
            still answers in well under a second, and the quality gain on
            morphologically rich Tamil is worth having.
        max_input_chars: Inputs longer than this are truncated with a warning
            rather than silently degrading latency.
    """

    def __init__(
        self,
        model_dir: Path,
        model_id: str,
        direction: str,
        cpu_threads: int = 2,
        beam_size: int = 4,
        max_input_chars: int = 512,
    ) -> None:
        if direction not in _DIRECTIONS:
            msg = f"direction must be one of {sorted(_DIRECTIONS)}, got {direction!r}"
            raise ValueError(msg)
        self._model_dir = model_dir
        self._model_id = model_id
        self._direction = direction
        self._cpu_threads = cpu_threads
        self._beam_size = beam_size
        self._max_input_chars = max_input_chars
        self._engine: Any = None
        self._src_spm: Any = None
        self._tgt_spm: Any = None
        self._warmed = False
        self._translations = 0
        self._total_ms = 0.0

    # -- port interface ----------------------------------------------------
    @property
    def model_id(self) -> str:
        """Identifier of the loaded model."""
        return self._model_id

    @property
    def direction(self) -> str:
        """Which way this instance translates."""
        return self._direction

    @property
    def translations_done(self) -> int:
        """Translations produced since creation."""
        return self._translations

    @property
    def mean_latency_ms(self) -> float:
        """Mean translation time so far, in milliseconds."""
        if not self._translations:
            return 0.0
        return self._total_ms / self._translations

    def supports(self, pair: LanguagePair) -> bool:
        """Whether this instance translates a direction.

        Args:
            pair: Direction to check.

        Returns:
            ``True`` when the pair matches this instance's direction and the
            Indic side has both an IndicTrans2 tag and an aligned script
            block (Urdu, in Perso-Arabic script, has neither mapping here).
        """
        if self._direction == "indic-en":
            indic, target = pair.source, pair.target
        else:
            target, indic = pair.source, pair.target
        return (
            target.code == "en"
            and indic.code != "en"
            and indic.code in INDICTRANS2_TAGS
            and supports_script_mapping(indic)
        )

    def warmup(self) -> None:
        """Load the model and run one throwaway translation.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        if self._warmed:
            return
        self._ensure_engine()
        started = time.perf_counter()
        pair = (
            LanguagePair.of("ta", "en")
            if self._direction == "indic-en"
            else LanguagePair.of("en", "ta")
        )
        sample = "வணக்கம்" if self._direction == "indic-en" else "Hello"
        try:
            self.translate(sample, pair)
        except TranslationError as exc:
            msg = f"Warmup translation failed for {self._model_id}: {exc}"
            raise ModelLoadError(msg) from exc
        # Warmup must not skew the running statistics.
        self._translations = 0
        self._total_ms = 0.0
        self._warmed = True
        logger.info(
            "%s warmed up in %.0f ms (%d threads, beam %d)",
            self._model_id,
            (time.perf_counter() - started) * 1000.0,
            self._cpu_threads,
            self._beam_size,
        )

    def translate(self, text: str, pair: LanguagePair) -> Translation:
        """Translate one piece of text.

        Args:
            text: Source text.
            pair: Direction to translate in.

        Returns:
            The translation. Empty input yields an empty translation rather
            than an error - silence is not a fault.

        Raises:
            TranslationError: If the pair is not supported or decoding fails.
        """
        if not self.supports(pair):
            msg = (
                f"Model {self._model_id} ({self._direction}) does not translate "
                f"{pair}. Check translation.models in configuration."
            )
            raise TranslationError(msg)

        cleaned = unicodedata.normalize("NFC", text.strip())
        if not cleaned:
            return Translation(
                utterance_id=UtteranceId("mt"),
                source_text=text,
                translated_text="",
                pair=pair,
                model_id=self._model_id,
            )
        if len(cleaned) > self._max_input_chars:
            logger.warning(
                "Truncating a %d-character input to %d for translation",
                len(cleaned),
                self._max_input_chars,
            )
            cleaned = cleaned[: self._max_input_chars]

        self._ensure_engine()
        started = time.perf_counter()

        source_for_model = (
            to_devanagari(cleaned, pair.source) if pair.source.code != "en" else cleaned
        )
        try:
            pieces = self._src_spm.encode(source_for_model, out_type=str)
            tokens = [
                INDICTRANS2_TAGS[pair.source.code],
                INDICTRANS2_TAGS[pair.target.code],
                *pieces,
            ]
            results = self._engine.translate_batch(
                [tokens],
                beam_size=self._beam_size,
                max_decoding_length=self._max_input_chars,
            )
            hypothesis = results[0].hypotheses[0]
            raw = self._tgt_spm.decode([piece for piece in hypothesis if not piece.startswith("<")])
        except Exception as exc:
            msg = f"Translation failed with {self._model_id}: {exc}"
            raise TranslationError(msg) from exc

        output = raw if pair.target.code == "en" else from_devanagari(raw, pair.target)
        output = _detokenize(output)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._translations += 1
        self._total_ms += elapsed_ms

        return Translation(
            utterance_id=UtteranceId("mt"),
            source_text=text,
            translated_text=output,
            pair=pair,
            model_id=self._model_id,
            latency_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Release the model. Safe to call more than once."""
        self._engine = None
        self._src_spm = None
        self._tgt_spm = None
        self._warmed = False

    # -- internals ---------------------------------------------------------
    def _ensure_engine(self) -> None:
        """Load CTranslate2 and both sentencepiece models on first use.

        Raises:
            ModelLoadError: If any file is missing or loading fails.
        """
        if self._engine is not None:
            return

        required = [
            self._model_dir / "model.bin",
            self._model_dir / "vocab" / "model.SRC",
            self._model_dir / "vocab" / "model.TGT",
        ]
        for path in required:
            if not path.is_file():
                msg = f"Translation model file not found: {path}"
                raise ModelLoadError(msg)

        try:
            import ctranslate2
            import sentencepiece

            self._engine = ctranslate2.Translator(
                str(self._model_dir),
                device="cpu",
                compute_type="int8",
                inter_threads=1,
                intra_threads=self._cpu_threads,
            )
            self._src_spm = sentencepiece.SentencePieceProcessor(
                model_file=str(self._model_dir / "vocab" / "model.SRC")
            )
            self._tgt_spm = sentencepiece.SentencePieceProcessor(
                model_file=str(self._model_dir / "vocab" / "model.TGT")
            )
        except Exception as exc:
            msg = f"Could not load {self._model_id} from {self._model_dir}: {exc}"
            raise ModelLoadError(msg) from exc

        logger.info(
            "%s loaded (%s, int8, %d threads)",
            self._model_id,
            self._direction,
            self._cpu_threads,
        )
