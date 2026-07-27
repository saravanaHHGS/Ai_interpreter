"""Per-language speech recogniser selection.

A bidirectional interpreter cannot use one speech model for both directions.
Measured on the target machine against a written Tamil reference:

* generic Whisper `base` has roughly **80 % word error on Tamil** - fast, and
  useless;
* the Tamil fine-tune reaches roughly **0-8 %**, but has lost English
  entirely and refuses it rather than guessing.

So the choice is per language, not per session. This router holds a recogniser
for each language and dispatches on the utterance's own language tag.

It satisfies the ``SpeechRecognizer`` port itself, so callers cannot tell
whether they hold one model or five - which is the point. The Phase 9 pipeline
will hold this object and never know that Tamil and English take different
code paths.

Recognisers are built **lazily and cached**. A session that only ever speaks
Tamil never loads the English model, so a one-directional session pays for one
model rather than two.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from ai_interpreter.domain.entities import Transcript, Utterance
from ai_interpreter.domain.errors import TranscriptionError
from ai_interpreter.domain.ports import SpeechRecognizer
from ai_interpreter.domain.value_objects import LanguageCode

__all__ = ["RecognizerRouter"]

logger = logging.getLogger(__name__)

RecognizerFactory = Callable[[LanguageCode], SpeechRecognizer]


class RecognizerRouter:
    """Dispatches transcription to a per-language recogniser.

    Args:
        factory: Builds a recogniser for a language. Called at most once per
            language; the result is cached.
        languages: Languages this router serves.
        warmup_languages: Languages to load and prime during
            :meth:`warmup`. Empty means load nothing up front, so the first
            utterance in each language pays the model load.
    """

    def __init__(
        self,
        factory: RecognizerFactory,
        languages: Iterable[LanguageCode],
        warmup_languages: Iterable[LanguageCode] = (),
    ) -> None:
        self._factory = factory
        self._languages = tuple(languages)
        self._warmup_languages = tuple(warmup_languages)
        self._recognizers: dict[str, SpeechRecognizer] = {}

    # -- port interface ----------------------------------------------------
    @property
    def model_id(self) -> str:
        """Identifier describing the routing set, for metrics and logs."""
        loaded = ", ".join(
            f"{code}:{recognizer.model_id}"
            for code, recognizer in sorted(self._recognizers.items())
        )
        return f"router({loaded})" if loaded else "router(empty)"

    @property
    def languages(self) -> tuple[LanguageCode, ...]:
        """Languages this router can serve."""
        return self._languages

    @property
    def loaded_languages(self) -> tuple[str, ...]:
        """Languages whose model is currently in memory."""
        return tuple(sorted(self._recognizers))

    def supports(self, language: LanguageCode) -> bool:
        """Whether a language can be routed.

        Args:
            language: Language to check.

        Returns:
            ``True`` when a recogniser is configured for it.
        """
        return any(language == candidate for candidate in self._languages)

    def for_language(self, language: LanguageCode) -> SpeechRecognizer:
        """Return the recogniser for a language, building it on first use.

        Args:
            language: Language to serve.

        Returns:
            The recogniser.

        Raises:
            TranscriptionError: If no recogniser is configured for it.
        """
        if not self.supports(language):
            available = ", ".join(str(code) for code in self._languages) or "none"
            msg = (
                f"No speech recogniser is configured for {language.english_name} "
                f"({language.code}). Configured languages: {available}. "
                "Add an entry to stt.language_models."
            )
            raise TranscriptionError(msg)

        existing = self._recognizers.get(language.code)
        if existing is not None:
            return existing

        logger.info("Loading the speech recogniser for %s", language.english_name)
        recognizer = self._factory(language)
        self._recognizers[language.code] = recognizer
        return recognizer

    def transcribe(self, utterance: Utterance) -> Transcript:
        """Transcribe an utterance with the model for its language.

        Args:
            utterance: Audio to transcribe. Its ``language`` selects the
                model; without one, the first configured language is used.

        Returns:
            The transcript.

        Raises:
            TranscriptionError: If no recogniser serves the language, or
                transcription fails.
        """
        language = utterance.language or self._default_language()
        return self.for_language(language).transcribe(utterance)

    def warmup(self) -> None:
        """Load and prime the recognisers listed for warmup.

        Loading every configured model up front would cost several seconds and
        several hundred megabytes for languages the session may never use, so
        the caller chooses which are worth pre-loading.
        """
        for language in self._warmup_languages:
            self.for_language(language).warmup()

    def close(self) -> None:
        """Release every loaded recogniser. Safe to call more than once."""
        for recognizer in self._recognizers.values():
            recognizer.close()
        self._recognizers.clear()

    # -- internals ---------------------------------------------------------
    def _default_language(self) -> LanguageCode:
        """Language to use when an utterance carries no tag.

        Returns:
            The first configured language.

        Raises:
            TranscriptionError: If the router has no languages at all.
        """
        if not self._languages:
            msg = "The recogniser router has no languages configured."
            raise TranscriptionError(msg)
        return self._languages[0]
