"""Non-blocking, rotating, privacy-aware logging.

Three design decisions worth understanding, because each one is a bug this
avoids rather than a preference:

**1. Every handler sits behind a queue.**
``RotatingFileHandler`` writes to disk synchronously, and when it rolls a file
over it renames several files while holding a lock. If that happens on the
audio capture thread, samples are dropped and the user hears a click. A
``QueueHandler`` makes ``logger.info(...)`` a near-instant append to an
in-memory queue; a background ``QueueListener`` thread does the disk work.
This is the standard library's own answer to logging from real-time code.

**2. Transcript text is filtered out by default.**
Meeting audio is confidential. Log records tagged as containing transcript
content are dropped unless ``privacy.log_transcripts`` is explicitly enabled,
so the log file you paste into a bug report cannot contain what was said in
your meeting.

**3. The console stream is forced to UTF-8.**
The Windows console defaults to a legacy code page. Logging a Tamil or Hindi
string to it raises ``UnicodeEncodeError`` and, because that happens inside
the logging machinery, it is easy to mistake for a crash in the model. This is
one of the most common failures in Indic-language Python applications on
Windows.

The third-party alternative here is ``loguru``. The standard library is used
instead because ``transformers``, ``torch`` and ``faster-whisper`` all emit
through ``logging``: configuring it directly captures their output too, with
no bridging shim and one less dependency.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import queue
import sys
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self, TextIO

from ai_interpreter.infrastructure.config.settings import LoggingSection

__all__ = ["LoggingService", "TranscriptFilter", "transcript_extra"]

# Attribute name set on records whose message contains conversation content.
_TRANSCRIPT_ATTR: Final[str] = "contains_transcript"

_FILE_FORMAT: Final[str] = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(threadName)-16s | %(name)s | %(message)s"
)
_FILE_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

_CONSOLE_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(message)s"
_CONSOLE_DATE_FORMAT: Final[str] = "%H:%M:%S"

# Libraries that are informative at DEBUG but extremely noisy at INFO.
_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "urllib3",
    "filelock",
    "huggingface_hub",
    "matplotlib",
    "numba",
    "asyncio",
)


def transcript_extra() -> dict[str, Any]:
    """Return the ``extra`` mapping that marks a record as transcript content.

    Use it whenever a log message includes recognised or translated text::

        logger.debug("Transcript: %s", text, extra=transcript_extra())

    Returns:
        A mapping suitable for the ``extra`` argument of a logging call.
    """
    return {_TRANSCRIPT_ATTR: True}


class _ConsoleSuppression:
    """Mutable threshold below which records are hidden from the console.

    Held in one object so the tagging filter and the context manager share
    state without a module-level global.
    """

    __slots__ = ("level",)

    def __init__(self) -> None:
        self.level = logging.NOTSET


class _ConsoleTaggingFilter(logging.Filter):
    """Marks records that the console should skip.

    Attached to the ``QueueHandler``, so it runs on the **calling** thread at
    the moment the record is created. That timing is the whole point: the
    listener thread emits records later, potentially after a live display has
    already finished, so a decision made at emit time would be a race. This
    one is deterministic.

    The record is always kept - only tagged. The file handlers ignore the tag,
    so nothing is ever lost from the log.

    Args:
        state: Shared suppression threshold.
    """

    def __init__(self, state: _ConsoleSuppression) -> None:
        super().__init__(name="console-tagging")
        self._state = state

    def filter(self, record: logging.LogRecord) -> bool:
        """Tag a record and always keep it.

        Args:
            record: Record being created.

        Returns:
            Always ``True``.
        """
        record.console_suppressed = record.levelno < self._state.level
        return True


class _ConsoleGateFilter(logging.Filter):
    """Drops records tagged for console suppression."""

    def __init__(self) -> None:
        super().__init__(name="console-gate")

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a record reaches the console.

        Args:
            record: Record being emitted.

        Returns:
            ``False`` when the record was created while the console was quiet.
        """
        return not getattr(record, "console_suppressed", False)


class TranscriptFilter(logging.Filter):
    """Drops records containing conversation content unless explicitly allowed.

    Args:
        allow: When ``False``, records marked by :func:`transcript_extra` are
            discarded before reaching any handler.
    """

    def __init__(self, *, allow: bool) -> None:
        super().__init__(name="transcript-filter")
        self._allow = allow

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a record may be emitted.

        Args:
            record: Record being considered.

        Returns:
            ``True`` to keep the record, ``False`` to discard it.
        """
        if self._allow:
            return True
        return not getattr(record, _TRANSCRIPT_ATTR, False)


class LoggingService:
    """Owns the application's logging configuration and its listener thread.

    Args:
        listener: Background listener draining the record queue.
        log_file: Path of the main rotating log file.
        error_file: Path of the errors-only log file, or ``None`` if disabled.
        handlers: Handlers attached to the root logger, kept for teardown.
    """

    def __init__(
        self,
        listener: logging.handlers.QueueListener,
        log_file: Path,
        error_file: Path | None,
        handlers: tuple[logging.Handler, ...],
        suppression: _ConsoleSuppression,
    ) -> None:
        self._listener = listener
        self._log_file = log_file
        self._error_file = error_file
        self._handlers = handlers
        self._suppression = suppression
        self._closed = False

    @contextlib.contextmanager
    def quiet_console(self, level: int = logging.WARNING) -> Iterator[None]:
        """Hide low-severity console output for the duration of a block.

        Live console output - a level meter redrawing with carriage returns -
        is destroyed by log lines arriving in the middle of it. Silencing the
        console while such a display is active keeps the terminal readable.

        The **file log is unaffected**: records are tagged, not dropped, so
        everything is still written to disk.

        Args:
            level: Records below this level are hidden from the console.

        Yields:
            Nothing. The previous threshold is restored on exit, including
            when the block raises.
        """
        previous = self._suppression.level
        self._suppression.level = level
        try:
            yield
        finally:
            self._suppression.level = previous

    # -- construction ------------------------------------------------------
    @classmethod
    def configure(
        cls,
        settings: LoggingSection,
        logs_dir: Path,
        *,
        allow_transcripts: bool,
        console_stream: TextIO | None = None,
    ) -> LoggingService:
        """Install the logging configuration on the root logger.

        Any previous configuration is removed first, so calling this twice
        (for example in a test suite) does not produce duplicated output.

        Args:
            settings: Validated logging configuration.
            logs_dir: Directory that will hold the log files.
            allow_transcripts: Whether conversation text may be written to
                logs. Comes from ``privacy.log_transcripts``.
            console_stream: Stream for console output, or ``None`` for
                ``sys.stderr``.

        Returns:
            The configured service. Call :meth:`shutdown` before exit so
            buffered records are flushed.
        """
        logs_dir.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()
        cls._reset(root)
        root.setLevel(logging.getLevelNamesMapping()[settings.level.value])

        transcript_filter = TranscriptFilter(allow=allow_transcripts)
        handlers: list[logging.Handler] = []

        log_file = logs_dir / "interpreter.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(logging.getLevelNamesMapping()[settings.file_level.value])
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, _FILE_DATE_FORMAT))
        file_handler.addFilter(transcript_filter)
        handlers.append(file_handler)

        error_file: Path | None = None
        if settings.error_log:
            error_file = logs_dir / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                filename=error_file,
                maxBytes=settings.max_bytes,
                backupCount=settings.backup_count,
                encoding="utf-8",
                delay=True,
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(logging.Formatter(_FILE_FORMAT, _FILE_DATE_FORMAT))
            error_handler.addFilter(transcript_filter)
            handlers.append(error_handler)

        stream = console_stream if console_stream is not None else sys.stderr
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(logging.getLevelNamesMapping()[settings.console_level.value])
        console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _CONSOLE_DATE_FORMAT))
        console_handler.addFilter(transcript_filter)
        console_handler.addFilter(_ConsoleGateFilter())
        handlers.append(console_handler)

        suppression = _ConsoleSuppression()

        # SimpleQueue is unbounded and lock-free on the producer side, which is
        # exactly what a real-time thread needs: logging can never block it.
        record_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
        queue_handler = logging.handlers.QueueHandler(record_queue)
        # Tagging happens here, on the calling thread, because the listener
        # emits records later - possibly after a live display has finished.
        queue_handler.addFilter(_ConsoleTaggingFilter(suppression))
        root.addHandler(queue_handler)

        listener = logging.handlers.QueueListener(
            record_queue, *handlers, respect_handler_level=True
        )
        listener.start()

        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        # Route warnings.warn() through logging so nothing bypasses the file.
        logging.captureWarnings(capture=True)

        return cls(
            listener=listener,
            log_file=log_file,
            error_file=error_file,
            handlers=tuple(handlers),
            suppression=suppression,
        )

    @staticmethod
    def _reset(root: logging.Logger) -> None:
        """Detach and close every handler currently on a logger.

        Args:
            root: Logger to clear.
        """
        for handler in list(root.handlers):
            root.removeHandler(handler)
            # A handler whose stream is already closed must not prevent
            # reconfiguration; there is nothing useful to do about it.
            with contextlib.suppress(OSError, ValueError):
                handler.close()

    # -- properties --------------------------------------------------------
    @property
    def log_file(self) -> Path:
        """Path of the main rotating log file."""
        return self._log_file

    @property
    def error_file(self) -> Path | None:
        """Path of the errors-only log file, or ``None`` when disabled."""
        return self._error_file

    # -- teardown ----------------------------------------------------------
    def shutdown(self) -> None:
        """Flush buffered records and stop the listener thread.

        Safe to call more than once. Skipping this loses whatever is still in
        the queue when the process exits.
        """
        if self._closed:
            return
        self._closed = True

        self._listener.stop()
        root = logging.getLogger()
        self._reset(root)
        for handler in self._handlers:
            with contextlib.suppress(OSError, ValueError):
                handler.close()
        logging.captureWarnings(capture=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.shutdown()
