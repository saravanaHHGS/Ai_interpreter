"""Unit tests for the opt-in online translator and its fallback composition."""

from __future__ import annotations

from typing import Any

import pytest

from ai_interpreter.application.services.fallback_translator import FallbackTranslator
from ai_interpreter.domain.entities import Translation, UtteranceId
from ai_interpreter.domain.errors import TranslationError
from ai_interpreter.domain.value_objects import LanguagePair
from ai_interpreter.infrastructure.translation.nim_llm import NimLlmTranslator

pytestmark = pytest.mark.unit

TA_EN = LanguagePair.of("ta", "en")
EN_TA = LanguagePair.of("en", "ta")


def _response(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _nim(
    reply: str = "There is a meeting with Nate tomorrow.",
    terms: tuple[str, ...] = ("VALD", "Nate"),
    fail: Exception | None = None,
) -> tuple[NimLlmTranslator, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(payload)
        if fail is not None:
            raise fail
        return _response(reply)

    translator = NimLlmTranslator(
        api_key="test-key",
        model="meta/llama-3.3-70b-instruct",
        context_terms=terms,
        transport=transport,
    )
    return translator, requests


class TestNimTranslator:
    """The LLM adapter in isolation, transport faked."""

    def test_translates_and_records_the_model(self) -> None:
        translator, requests = _nim()

        result = translator.translate("நேட்டு கூட meeting இருக்கு", TA_EN)

        assert result.translated_text == "There is a meeting with Nate tomorrow."
        assert result.model_id == "nim:meta/llama-3.3-70b-instruct"
        assert requests[0]["messages"][1]["content"] == "நேட்டு கூட meeting இருக்கு"

    def test_context_terms_reach_the_prompt(self) -> None:
        # The mechanism that lets the model know நேட்டு is Nate.
        translator, requests = _nim(terms=("VALD", "Nate", "GamePlan"))
        translator.translate("வணக்கம்", TA_EN)

        system = requests[0]["messages"][0]["content"]
        assert "VALD, Nate, GamePlan" in system
        assert "ONLY the English translation" in system

    def test_direction_switches_the_target_language(self) -> None:
        translator, requests = _nim()
        translator.translate("See you tomorrow", EN_TA)

        assert "Tamil translation" in requests[0]["messages"][0]["content"]

    def test_surrounding_quotes_are_stripped(self) -> None:
        translator, _ = _nim(reply='"Hello there."')
        result = translator.translate("வணக்கம்", TA_EN)

        assert result.translated_text == "Hello there."

    def test_empty_input_never_touches_the_network(self) -> None:
        translator, requests = _nim()
        result = translator.translate("   ", TA_EN)

        assert result.is_empty
        assert requests == []

    def test_transport_failure_raises_translation_error(self) -> None:
        translator, _ = _nim(fail=TranslationError("timed out"))

        with pytest.raises(TranslationError):
            translator.translate("வணக்கம்", TA_EN)

    def test_malformed_response_raises_translation_error(self) -> None:
        def transport(payload: dict[str, Any]) -> dict[str, Any]:
            return {"unexpected": True}

        translator = NimLlmTranslator(api_key="k", model="m", transport=transport)

        with pytest.raises(TranslationError, match="response shape"):
            translator.translate("வணக்கம்", TA_EN)

    def test_unsupported_pair_is_rejected(self) -> None:
        translator, _ = _nim()

        assert not translator.supports(LanguagePair.of("hi", "en"))


class _ScriptedTranslator:
    """Local-engine stand-in."""

    def __init__(self, text: str = "local answer", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0
        self.warmed = 0

    @property
    def model_id(self) -> str:
        return "local"

    def supports(self, pair: LanguagePair) -> bool:
        return True

    def warmup(self) -> None:
        self.warmed += 1

    def translate(self, text: str, pair: LanguagePair) -> Translation:
        self.calls += 1
        if self.fail:
            raise TranslationError("scripted failure")
        return Translation(
            utterance_id=UtteranceId("mt"),
            source_text=text,
            translated_text=self.text,
            pair=pair,
        )

    def close(self) -> None: ...


class TestFallbackTranslator:
    """Online primary, local safety net."""

    def test_primary_answer_is_used(self) -> None:
        primary = _ScriptedTranslator(text="online answer")
        fallback = _ScriptedTranslator(text="local answer")
        translator = FallbackTranslator(primary, fallback)  # type: ignore[arg-type]

        result = translator.translate("வணக்கம்", TA_EN)

        assert result.translated_text == "online answer"
        assert fallback.calls == 0
        assert translator.fallbacks == 0

    def test_primary_failure_falls_back_and_is_counted(self) -> None:
        primary = _ScriptedTranslator(fail=True)
        fallback = _ScriptedTranslator(text="local answer")
        translator = FallbackTranslator(primary, fallback)  # type: ignore[arg-type]

        result = translator.translate("வணக்கம்", TA_EN)

        assert result.translated_text == "local answer"
        assert translator.fallbacks == 1

    def test_warmup_warms_the_fallback_first(self) -> None:
        primary = _ScriptedTranslator()
        fallback = _ScriptedTranslator()
        FallbackTranslator(primary, fallback).warmup()  # type: ignore[arg-type]

        assert fallback.warmed == 1
        assert primary.warmed == 1

    def test_model_id_names_both_engines(self) -> None:
        translator = FallbackTranslator(
            _ScriptedTranslator(),  # type: ignore[arg-type]
            _ScriptedTranslator(),  # type: ignore[arg-type]
        )

        assert "fallback" in translator.model_id
