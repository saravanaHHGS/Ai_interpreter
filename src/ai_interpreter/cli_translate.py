"""Translation command.

``--translate`` translates one piece of text with the configured engine and
reports timings for a cold call and a repeat call, making the cache's effect
visible: the second call should return in well under a millisecond.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Final

from ai_interpreter.app.container import Container
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.domain.value_objects import LanguagePair
from ai_interpreter.presentation.console import WIDTH, heading, row

__all__ = ["run_translate"]

logger = logging.getLogger(__name__)

_EXIT_OK: Final[int] = 0
_EXIT_ERROR: Final[int] = 1


def run_translate(
    container: Container,
    text: str,
    source: str | None,
    target: str | None,
) -> int:
    """Translate one piece of text and report timings.

    Args:
        container: Built application container.
        text: Text to translate.
        source: Source language code, or ``None`` for the configured pair.
        target: Target language code, or ``None`` for the configured pair.

    Returns:
        Process exit code.
    """
    print("=" * WIDTH)
    print("  Translate")
    print("=" * WIDTH)

    configured = container.settings.app.language_pair
    try:
        pair = LanguagePair.of(source or configured.source, target or configured.target)
    except ValueError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    heading("Loading")
    row("Direction", str(pair))
    try:
        translator = container.create_translator(pair)
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    try:
        started = time.perf_counter()
        translator.warmup()
        row("Model", translator.model_id)
        row("Warmup", f"{(time.perf_counter() - started) * 1000.0:.0f} ms")

        heading("Translation")
        row("Source", text)
        started = time.perf_counter()
        result = translator.translate(text, pair)
        cold_ms = (time.perf_counter() - started) * 1000.0
        row("Result", result.translated_text or "(empty)")
        row("Time", f"{cold_ms:.0f} ms")

        started = time.perf_counter()
        repeat = translator.translate(text, pair)
        warm_ms = (time.perf_counter() - started) * 1000.0
        row(
            "Repeat call",
            f"{warm_ms:.2f} ms ({'cache hit' if repeat.from_cache else 'cache disabled'})",
        )
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR
    finally:
        translator.close()

    return _EXIT_OK
