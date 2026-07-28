"""Unit tests for the translation stack.

The CTranslate2 engine and sentencepiece models are replaced with fakes, as in
the STT tests: what is under test is the adapter's request assembly - the
transliteration, tags and detokenisation whose absence produced garbage on the
first live attempt - plus the cache semantics. Real-model quality is verified
through ``--translate`` against reference sentences.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from ai_interpreter.application.services.cached_translator import CachedTranslator
from ai_interpreter.domain.errors import ModelLoadError, TranslationError
from ai_interpreter.domain.value_objects import LanguageCode, LanguagePair
from ai_interpreter.infrastructure.translation.cache import LruTranslationCache
from ai_interpreter.infrastructure.translation.indictrans2 import (
    IndicTrans2Translator,
    _detokenize,
)
from ai_interpreter.infrastructure.translation.transliteration import (
    INDICTRANS2_TAGS,
    from_devanagari,
    supports_script_mapping,
    to_devanagari,
)

pytestmark = pytest.mark.unit

TAMIL = LanguageCode("ta")
HINDI = LanguageCode("hi")
ENGLISH = LanguageCode("en")
TA_EN = LanguagePair.of("ta", "en")
EN_TA = LanguagePair.of("en", "ta")


class TestTransliteration:
    """The script-unification step IndicTrans2 requires."""

    def test_tamil_shifts_onto_devanagari(self) -> None:
        # க (0x0B95) and क (0x0915) sit at the same offset in their blocks.
        assert to_devanagari("க", TAMIL) == "क"

    def test_round_trip_preserves_tamil(self) -> None:
        text = "என் பெயர் சரவணகுமார்"
        assert from_devanagari(to_devanagari(text, TAMIL), TAMIL) == text

    def test_devanagari_languages_pass_through(self) -> None:
        text = "मेरा नाम"
        assert to_devanagari(text, HINDI) == text
        assert from_devanagari(text, HINDI) == text

    def test_punctuation_digits_and_latin_are_untouched(self) -> None:
        text = "வணக்கம், 25% ok?"
        shifted = to_devanagari(text, TAMIL)
        for kept in (",", " ", "2", "5", "%", "o", "k", "?"):
            assert kept in shifted

    def test_mapping_support_excludes_urdu(self) -> None:
        # Urdu is Perso-Arabic script: no aligned block, no offset mapping.
        assert not supports_script_mapping(LanguageCode("ur"))
        assert supports_script_mapping(TAMIL)
        assert supports_script_mapping(HINDI)

    def test_every_mapped_language_has_a_model_tag(self) -> None:
        for code in ("ta", "hi", "te", "ml", "kn", "bn", "mr", "gu", "pa", "en"):
            assert code in INDICTRANS2_TAGS


class TestDetokenize:
    """Sentencepiece output cleanup."""

    def test_removes_space_before_punctuation(self) -> None:
        assert _detokenize("What is the plan ?") == "What is the plan?"

    def test_fixes_clitics(self) -> None:
        assert _detokenize("What 's today your plan ?") == "What's today your plan?"

    def test_plain_text_is_unchanged(self) -> None:
        assert _detokenize("My name is Saravanakumar") == "My name is Saravanakumar"


# ---------------------------------------------------------------------------
# Translator with fake engine
# ---------------------------------------------------------------------------
class FakeSpm:
    """Splits on whitespace; decodes by joining."""

    def encode(self, text: str, out_type: type = str) -> list[str]:
        return [f"▁{word}" for word in text.split()]

    def decode(self, pieces: list[str]) -> str:
        return " ".join(piece.lstrip("▁") for piece in pieces)


class FakeHypothesisResult:
    def __init__(self, tokens: list[str]) -> None:
        self.hypotheses = [tokens]


class FakeCt2Engine:
    """Records requests and returns scripted output tokens."""

    def __init__(self, output_tokens: list[str] | None = None) -> None:
        self.output = output_tokens if output_tokens is not None else ["▁translated"]
        self.requests: list[dict[str, Any]] = []

    def translate_batch(self, batch: list[list[str]], **kwargs: Any) -> list[Any]:
        self.requests.append({"tokens": batch[0], **kwargs})
        return [FakeHypothesisResult(list(self.output))]


def _translator(
    direction: str = "indic-en",
    engine: FakeCt2Engine | None = None,
    **kwargs: Any,
) -> tuple[IndicTrans2Translator, FakeCt2Engine]:
    """Build a translator with fakes injected.

    Args:
        direction: Model direction.
        engine: Fake engine, or ``None`` for a fresh one.
        **kwargs: Constructor overrides.

    Returns:
        The translator and its fake engine.
    """
    options: dict[str, Any] = {
        "model_dir": Path("unused"),
        "model_id": f"indictrans2-{direction}",
        "direction": direction,
        "beam_size": 4,
    }
    options.update(kwargs)
    translator = IndicTrans2Translator(**options)
    fake = engine or FakeCt2Engine()
    translator._engine = fake
    translator._src_spm = FakeSpm()
    translator._tgt_spm = FakeSpm()
    return translator, fake


class TestRequestAssembly:
    """The exact input format the model requires."""

    def test_prefixes_source_and_target_tags(self) -> None:
        translator, engine = _translator()
        translator.translate("வணக்கம்", TA_EN)

        tokens = engine.requests[0]["tokens"]
        assert tokens[0] == "tam_Taml"
        assert tokens[1] == "eng_Latn"

    def test_tamil_source_is_transliterated_to_devanagari(self) -> None:
        # The bug the first live run exposed: without this, sentencepiece
        # falls back to characters and the decoder produces gibberish.
        translator, engine = _translator()
        translator.translate("க", TA_EN)

        pieces = engine.requests[0]["tokens"][2:]
        assert pieces == ["▁क"]

    def test_english_source_is_not_transliterated(self) -> None:
        translator, engine = _translator(direction="en-indic")
        translator.translate("Hello", EN_TA)

        assert engine.requests[0]["tokens"] == ["eng_Latn", "tam_Taml", "▁Hello"]

    def test_beam_size_is_passed_through(self) -> None:
        translator, engine = _translator(beam_size=7)
        translator.translate("வணக்கம்", TA_EN)

        assert engine.requests[0]["beam_size"] == 7

    def test_devanagari_output_is_shifted_to_the_target_script(self) -> None:
        translator, _ = _translator(direction="en-indic", engine=FakeCt2Engine(["▁क"]))
        result = translator.translate("Hello", EN_TA)

        assert result.translated_text == "க"

    def test_english_output_is_detokenized(self) -> None:
        translator, _ = _translator(engine=FakeCt2Engine(["▁What", "▁'s", "▁the", "▁plan", "▁?"]))
        result = translator.translate("என்ன திட்டம்", TA_EN)

        assert result.translated_text == "What's the plan?"

    def test_special_tokens_are_stripped_from_output(self) -> None:
        translator, _ = _translator(engine=FakeCt2Engine(["▁hello", "</s>"]))
        result = translator.translate("வணக்கம்", TA_EN)

        assert "</s>" not in result.translated_text

    def test_long_input_is_truncated(self) -> None:
        translator, engine = _translator(max_input_chars=10)
        translator.translate("வணக்கம் " * 50, TA_EN)

        source_pieces = engine.requests[0]["tokens"][2:]
        assert len("".join(source_pieces)) < 30


class TestTranslateBehaviour:
    """Result semantics."""

    def test_empty_input_returns_empty_without_calling_the_engine(self) -> None:
        translator, engine = _translator()
        result = translator.translate("   ", TA_EN)

        assert result.is_empty
        assert engine.requests == []

    def test_records_latency_and_model_id(self) -> None:
        translator, _ = _translator()
        result = translator.translate("வணக்கம்", TA_EN)

        assert result.model_id == "indictrans2-indic-en"
        assert result.latency_ms >= 0.0
        assert result.from_cache is False

    def test_tracks_statistics(self) -> None:
        translator, _ = _translator()
        translator.translate("ஒன்று", TA_EN)
        translator.translate("இரண்டு", TA_EN)

        assert translator.translations_done == 2

    def test_engine_failure_is_wrapped(self) -> None:
        class ExplodingEngine(FakeCt2Engine):
            def translate_batch(self, batch: list[list[str]], **kwargs: Any) -> list[Any]:
                raise RuntimeError("ct2 exploded")

        translator, _ = _translator(engine=ExplodingEngine())
        with pytest.raises(TranslationError, match="Translation failed"):
            translator.translate("வணக்கம்", TA_EN)


class TestDirectionSupport:
    """One instance serves one direction."""

    def test_indic_en_supports_only_indic_to_english(self) -> None:
        translator, _ = _translator(direction="indic-en")

        assert translator.supports(TA_EN)
        assert translator.supports(LanguagePair.of("hi", "en"))
        assert not translator.supports(EN_TA)

    def test_en_indic_supports_only_english_to_indic(self) -> None:
        translator, _ = _translator(direction="en-indic")

        assert translator.supports(EN_TA)
        assert not translator.supports(TA_EN)

    def test_urdu_is_not_supported(self) -> None:
        # Perso-Arabic script has no aligned block to transliterate.
        translator, _ = _translator(direction="indic-en")
        assert not translator.supports(LanguagePair.of("ur", "en"))

    def test_wrong_direction_raises_clearly(self) -> None:
        translator, _ = _translator(direction="indic-en")
        with pytest.raises(TranslationError, match="does not translate"):
            translator.translate("Hello", EN_TA)

    def test_invalid_direction_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="direction must be one of"):
            IndicTrans2Translator(model_dir=Path("unused"), model_id="x", direction="sideways")


class TestModelLoading:
    """Missing files fail clearly."""

    def test_missing_model_reports_the_path(self, tmp_path: Path) -> None:
        translator = IndicTrans2Translator(
            model_dir=tmp_path, model_id="indictrans2-indic-en", direction="indic-en"
        )
        with pytest.raises(ModelLoadError, match=r"model\.bin"):
            translator.warmup()

    def test_close_is_idempotent(self) -> None:
        translator, _ = _translator()
        translator.close()
        translator.close()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class TestLruTranslationCache:
    """Bounded LRU semantics."""

    def test_miss_then_hit(self) -> None:
        cache = LruTranslationCache(max_entries=10)
        assert cache.get("வணக்கம்", TA_EN) is None

        cache.put("வணக்கம்", TA_EN, "Hello")
        assert cache.get("வணக்கம்", TA_EN) == "Hello"

    def test_directions_do_not_collide(self) -> None:
        cache = LruTranslationCache(max_entries=10)
        cache.put("test", TA_EN, "result-a")
        cache.put("test", EN_TA, "result-b")

        assert cache.get("test", TA_EN) == "result-a"
        assert cache.get("test", EN_TA) == "result-b"

    def test_lookup_ignores_case_and_extra_whitespace(self) -> None:
        cache = LruTranslationCache(max_entries=10)
        cache.put("Hello there", EN_TA, "வணக்கம்")

        assert cache.get("  hello   THERE ", EN_TA) == "வணக்கம்"

    def test_evicts_least_recently_used(self) -> None:
        cache = LruTranslationCache(max_entries=2)
        cache.put("a", TA_EN, "1")
        cache.put("b", TA_EN, "2")
        cache.get("a", TA_EN)  # refresh a
        cache.put("c", TA_EN, "3")  # evicts b

        assert cache.get("a", TA_EN) == "1"
        assert cache.get("b", TA_EN) is None
        assert cache.get("c", TA_EN) == "3"

    def test_empty_translations_are_not_stored(self) -> None:
        cache = LruTranslationCache(max_entries=10)
        cache.put("text", TA_EN, "   ")

        assert cache.size == 0

    def test_zero_capacity_disables_storage(self) -> None:
        cache = LruTranslationCache(max_entries=0)
        cache.put("text", TA_EN, "result")

        assert cache.get("text", TA_EN) is None

    def test_hit_rate(self) -> None:
        cache = LruTranslationCache(max_entries=10)
        cache.put("a", TA_EN, "1")
        cache.get("a", TA_EN)
        cache.get("missing", TA_EN)

        assert cache.hit_rate == pytest.approx(0.5)

    def test_clear_resets_everything(self) -> None:
        cache = LruTranslationCache(max_entries=10)
        cache.put("a", TA_EN, "1")
        cache.get("a", TA_EN)
        cache.clear()

        assert cache.size == 0
        assert cache.hit_rate == 0.0

    def test_negative_capacity_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            LruTranslationCache(max_entries=-1)

    def test_concurrent_access_survives(self) -> None:
        cache = LruTranslationCache(max_entries=100)
        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for index in range(200):
                    cache.put(f"text-{worker_id}-{index % 20}", TA_EN, f"r{index}")
                    cache.get(f"text-{worker_id}-{index % 20}", TA_EN)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert cache.size <= 100


class TestCachedTranslator:
    """The decorator that makes repeats free."""

    def test_first_call_translates_second_call_hits(self) -> None:
        inner, _ = _translator()
        cached = CachedTranslator(inner, LruTranslationCache(max_entries=10))

        first = cached.translate("வணக்கம்", TA_EN)
        second = cached.translate("வணக்கம்", TA_EN)

        assert first.from_cache is False
        assert second.from_cache is True
        assert second.translated_text == first.translated_text
        assert inner.translations_done == 1

    def test_empty_results_are_not_cached(self) -> None:
        inner, _ = _translator(engine=FakeCt2Engine([]))
        cached = CachedTranslator(inner, LruTranslationCache(max_entries=10))

        cached.translate("வணக்கம்", TA_EN)
        cached.translate("வணக்கம்", TA_EN)

        assert inner.translations_done == 2

    def test_delegates_support_and_model_id(self) -> None:
        inner, _ = _translator()
        cached = CachedTranslator(inner, LruTranslationCache(max_entries=10))

        assert cached.supports(TA_EN)
        assert not cached.supports(EN_TA)
        assert cached.model_id == inner.model_id
