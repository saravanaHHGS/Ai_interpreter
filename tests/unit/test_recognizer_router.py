"""Unit tests for per-language recogniser routing."""

from __future__ import annotations

import numpy as np
import pytest

from ai_interpreter.application.services.recognizer_router import RecognizerRouter
from ai_interpreter.domain.entities import Transcript, Utterance, UtteranceId
from ai_interpreter.domain.errors import TranscriptionError
from ai_interpreter.domain.value_objects import Confidence, LanguageCode, SampleRate

pytestmark = pytest.mark.unit

RATE = SampleRate(16000)
TAMIL = LanguageCode("ta")
ENGLISH = LanguageCode("en")


class FakeRecognizer:
    """Records what it was asked to do, without loading anything."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.warmups = 0
        self.closes = 0
        self.transcribed: list[UtteranceId] = []

    def supports(self, language: LanguageCode) -> bool:
        return bool(language)

    def transcribe(self, utterance: Utterance) -> Transcript:
        self.transcribed.append(utterance.id)
        return Transcript(
            utterance_id=utterance.id,
            text=f"decoded by {self.model_id}",
            language=utterance.language or ENGLISH,
            confidence=Confidence(0.9),
            is_final=True,
            model_id=self.model_id,
        )

    def warmup(self) -> None:
        self.warmups += 1

    def close(self) -> None:
        self.closes += 1


class RecordingFactory:
    """Builds fake recognisers and counts how often it is called."""

    def __init__(self) -> None:
        self.built: list[str] = []
        self.instances: dict[str, FakeRecognizer] = {}

    def __call__(self, language: LanguageCode) -> FakeRecognizer:
        self.built.append(language.code)
        recognizer = FakeRecognizer(f"whisper-{language.code}")
        self.instances[language.code] = recognizer
        return recognizer


def _utterance(language: LanguageCode | None, name: str = "u1") -> Utterance:
    """Build a short utterance tagged with a language.

    Args:
        language: Language tag, or ``None``.
        name: Utterance identifier.

    Returns:
        The utterance.
    """
    return Utterance(
        id=UtteranceId(name),
        pcm=np.zeros(16000, dtype=np.float32),
        sample_rate=RATE,
        started_at_ms=0.0,
        ended_at_ms=1000.0,
        language=language,
    )


def _router(factory: RecordingFactory, **kwargs: object) -> RecognizerRouter:
    """Build a router over Tamil and English.

    Args:
        factory: Recogniser factory.
        **kwargs: Constructor overrides.

    Returns:
        The router.
    """
    options: dict[str, object] = {"languages": (TAMIL, ENGLISH)}
    options.update(kwargs)
    return RecognizerRouter(factory=factory, **options)  # type: ignore[arg-type]


class TestRouting:
    """Choosing a recogniser by language."""

    def test_routes_each_language_to_its_own_model(self) -> None:
        # The whole reason this class exists: `base` is useless on Tamil and
        # the Tamil fine-tune refuses English.
        factory = RecordingFactory()
        router = _router(factory)

        tamil = router.transcribe(_utterance(TAMIL, "u1"))
        english = router.transcribe(_utterance(ENGLISH, "u2"))

        assert tamil.model_id == "whisper-ta"
        assert english.model_id == "whisper-en"

    def test_falls_back_to_the_first_language_when_untagged(self) -> None:
        factory = RecordingFactory()
        _router(factory).transcribe(_utterance(None))

        assert factory.built == ["ta"]

    def test_reports_supported_languages(self) -> None:
        router = _router(RecordingFactory())

        assert router.supports(TAMIL)
        assert router.supports(ENGLISH)
        assert not router.supports(LanguageCode("hi"))

    def test_unconfigured_language_is_refused_clearly(self) -> None:
        router = _router(RecordingFactory())

        with pytest.raises(TranscriptionError, match="No speech recogniser is configured"):
            router.transcribe(_utterance(LanguageCode("hi")))

    def test_the_error_names_the_configured_languages(self) -> None:
        router = _router(RecordingFactory())

        with pytest.raises(TranscriptionError, match="Configured languages: ta, en"):
            router.for_language(LanguageCode("hi"))


class TestLazyLoading:
    """Models load only when a language is actually used."""

    def test_nothing_is_built_up_front(self) -> None:
        factory = RecordingFactory()
        router = _router(factory)

        assert factory.built == []
        assert router.loaded_languages == ()

    def test_only_the_used_language_is_built(self) -> None:
        # A one-directional session must not pay for the other direction's
        # model: these are 145 MB and 253 MB.
        factory = RecordingFactory()
        router = _router(factory)
        router.transcribe(_utterance(TAMIL))

        assert factory.built == ["ta"]
        assert router.loaded_languages == ("ta",)

    def test_each_model_is_built_once(self) -> None:
        factory = RecordingFactory()
        router = _router(factory)
        for index in range(5):
            router.transcribe(_utterance(TAMIL, f"u{index}"))

        assert factory.built == ["ta"]

    def test_repeated_calls_reuse_the_same_instance(self) -> None:
        factory = RecordingFactory()
        router = _router(factory)

        assert router.for_language(TAMIL) is router.for_language(TAMIL)


class TestLifecycle:
    """Warmup and teardown."""

    def test_warmup_loads_only_the_requested_languages(self) -> None:
        factory = RecordingFactory()
        router = _router(factory, warmup_languages=(ENGLISH,))
        router.warmup()

        assert factory.built == ["en"]
        assert factory.instances["en"].warmups == 1

    def test_warmup_does_nothing_by_default(self) -> None:
        factory = RecordingFactory()
        _router(factory).warmup()

        assert factory.built == []

    def test_close_releases_every_loaded_model(self) -> None:
        factory = RecordingFactory()
        router = _router(factory)
        router.for_language(TAMIL)
        router.for_language(ENGLISH)
        router.close()

        assert factory.instances["ta"].closes == 1
        assert factory.instances["en"].closes == 1
        assert router.loaded_languages == ()

    def test_close_is_idempotent(self) -> None:
        router = _router(RecordingFactory())
        router.for_language(TAMIL)
        router.close()
        router.close()

    def test_reloads_after_close(self) -> None:
        factory = RecordingFactory()
        router = _router(factory)
        router.for_language(TAMIL)
        router.close()
        router.for_language(TAMIL)

        assert factory.built == ["ta", "ta"]


class TestModelId:
    """Identifier reported for metrics."""

    def test_reports_loaded_models(self) -> None:
        factory = RecordingFactory()
        router = _router(factory)
        router.for_language(TAMIL)

        assert router.model_id == "router(ta:whisper-ta)"

    def test_reports_empty_before_anything_loads(self) -> None:
        assert _router(RecordingFactory()).model_id == "router(empty)"
