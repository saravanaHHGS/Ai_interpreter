"""LLM translation through NVIDIA NIM - the opt-in online provider.

Why this exists: the local IndicTrans2 translates *sentences*, but some
meaning lives above the sentence. The motivating live example was

    இன்னும் ரெண்டு நாள் நேட்டு கூட meeting tomorrow இருக்கு

where நேட்டு is a colleague named Nate - something only a model that
*understands the scenario* can know. A large instruction-tuned LLM handles
code-switched colloquial Tamil, person names and technical context in a
way no 200M sentence-to-sentence model can.

The cost is the project's core promise: text sent here LEAVES THE MACHINE.
This provider is therefore strictly opt-in (``translation.online.enabled``
plus an API key in ``.env``), loudly logged when active, and always wrapped
in a fallback to the local engine so a network failure never silences the
interpreter.

The HTTP transport is the standard library (no new dependency); NIM speaks
the OpenAI chat-completions dialect.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Final

from ai_interpreter.domain.entities import Translation, UtteranceId
from ai_interpreter.domain.errors import TranslationError
from ai_interpreter.domain.value_objects import LanguagePair

__all__ = ["NimLlmTranslator"]

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1/chat/completions"

# The languages this adapter is prompted for; the LLM itself is general.
_SUPPORTED: Final[frozenset[str]] = frozenset({"ta", "en"})

_LANGUAGE_NAMES: Final[dict[str, str]] = {"ta": "Tamil", "en": "English"}


def _build_system_prompt(target: str, terms: Sequence[str]) -> str:
    """Build the instruction prompt for one translation direction.

    Args:
        target: Target language name.
        terms: Known names and technical terms from the user's vocabulary.

    Returns:
        The system prompt.
    """
    known = ", ".join(dict.fromkeys(term.strip() for term in terms if term.strip()))
    vocabulary = (
        f" Known names and technical terms in this workplace: {known}."
        " Keep them exactly as written; some may appear transliterated into"
        " Tamil script."
        if known
        else ""
    )
    return (
        "You translate spoken workplace utterances into "
        f"{target}. The input is colloquial and may freely mix Tamil and"
        " English words, including English words and PERSON NAMES written in"
        " Tamil script. Preserve names and technical terms rather than"
        f" translating them.{vocabulary} Reply with ONLY the {target}"
        " translation - no explanations, no quotation marks, no alternatives."
    )


class NimLlmTranslator:
    """Translation via an instruction-tuned LLM on NVIDIA NIM.

    Satisfies the ``Translator`` port. Stateless per call; every request is
    one chat completion.

    Args:
        api_key: NIM API key (from ``.env``, never from YAML).
        model: NIM model identifier, e.g. ``meta/llama-3.3-70b-instruct``.
        timeout_s: Hard deadline per request. A live meeting cannot wait; on
            expiry the fallback translator takes over.
        context_terms: The user's names and technical terms, injected into
            the prompt - this is what lets the model know நேட்டு is Nate.
        base_url: Endpoint override, for tests or self-hosted NIM.
        transport: Test seam: replaces the HTTP call with a callable taking
            the request payload and returning the decoded response body.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 6.0,
        context_terms: Sequence[str] = (),
        base_url: str = _DEFAULT_BASE_URL,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._context_terms = tuple(context_terms)
        self._base_url = base_url
        self._transport = transport
        self._translations = 0
        self._total_ms = 0.0

    # -- port interface ----------------------------------------------------
    @property
    def model_id(self) -> str:
        """Identifier recorded on translations."""
        return f"nim:{self._model}"

    @property
    def translations(self) -> int:
        """Successful online translations so far."""
        return self._translations

    @property
    def mean_ms(self) -> float:
        """Mean round-trip time so far, in milliseconds."""
        if not self._translations:
            return 0.0
        return self._total_ms / self._translations

    def supports(self, pair: LanguagePair) -> bool:
        """Whether this adapter is prompted for a direction.

        Args:
            pair: Direction to check.

        Returns:
            ``True`` for Tamil/English in either direction.
        """
        return pair.source.code in _SUPPORTED and pair.target.code in _SUPPORTED

    def warmup(self) -> None:
        """No-op: a network call costs quota and proves nothing durable."""

    def translate(self, text: str, pair: LanguagePair) -> Translation:
        """Translate one utterance through the LLM.

        Args:
            text: Source text (colloquial, possibly mixed-script).
            pair: Direction to translate.

        Returns:
            The translation.

        Raises:
            TranslationError: On timeout, transport failure, HTTP error or a
                malformed response - the caller's fallback then takes over.
        """
        if not self.supports(pair):
            msg = f"Model {self.model_id} is not prompted for {pair}."
            raise TranslationError(msg)

        cleaned = text.strip()
        if not cleaned:
            return Translation(
                utterance_id=UtteranceId("mt"),
                source_text=text,
                translated_text="",
                pair=pair,
                model_id=self.model_id,
            )

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": _build_system_prompt(
                        _LANGUAGE_NAMES[pair.target.code], self._context_terms
                    ),
                },
                {"role": "user", "content": cleaned},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }

        started = time.perf_counter()
        body = (self._transport or self._request)(payload)
        try:
            output = str(body["choices"][0]["message"]["content"]).strip().strip('"')
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"Unexpected response shape from {self.model_id}: {exc}"
            raise TranslationError(msg) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._translations += 1
        self._total_ms += elapsed_ms
        return Translation(
            utterance_id=UtteranceId("mt"),
            source_text=text,
            translated_text=output,
            pair=pair,
            model_id=self.model_id,
            latency_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Nothing to release; the adapter holds no connection state."""

    # -- transport ---------------------------------------------------------
    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one chat completion to NIM.

        Args:
            payload: OpenAI-dialect request body.

        Returns:
            The decoded response body.

        Raises:
            TranslationError: On any transport or HTTP failure.
        """
        request = urllib.request.Request(
            self._base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            detail = ""
            with contextlib.suppress(Exception):
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            msg = f"NIM returned HTTP {exc.code} for {self._model}: {detail}"
            raise TranslationError(msg) from exc
        except Exception as exc:
            msg = f"NIM request failed for {self._model}: {exc}"
            raise TranslationError(msg) from exc
