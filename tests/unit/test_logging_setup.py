"""Unit tests for logging configuration, rotation and privacy filtering."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from ai_interpreter.infrastructure.config.settings import LoggingSection, LogLevel
from ai_interpreter.infrastructure.logging.setup import LoggingService, transcript_extra

pytestmark = pytest.mark.unit


def _settings(**overrides: object) -> LoggingSection:
    """Build a logging configuration for tests.

    Args:
        **overrides: Fields to override.

    Returns:
        A validated logging section.
    """
    data: dict[str, object] = {
        "level": LogLevel.DEBUG,
        "console_level": LogLevel.WARNING,
        "file_level": LogLevel.DEBUG,
        "directory": "logs",
        "max_bytes": 64_000,
        "backup_count": 2,
        "error_log": True,
    }
    data.update(overrides)
    return LoggingSection.model_validate(data)


class TestLoggingConfiguration:
    """Handler installation and file output."""

    def test_writes_records_to_the_log_file(self, tmp_path: Path) -> None:
        with LoggingService.configure(
            _settings(), tmp_path, allow_transcripts=False, console_stream=io.StringIO()
        ) as service:
            logging.getLogger("test.writer").info("pipeline started")

        content = service.log_file.read_text(encoding="utf-8")
        assert "pipeline started" in content

    def test_creates_a_separate_error_log(self, tmp_path: Path) -> None:
        with LoggingService.configure(
            _settings(), tmp_path, allow_transcripts=False, console_stream=io.StringIO()
        ) as service:
            logging.getLogger("test.errors").info("routine message")
            logging.getLogger("test.errors").error("device disconnected")

        assert service.error_file is not None
        errors = service.error_file.read_text(encoding="utf-8")
        assert "device disconnected" in errors
        assert "routine message" not in errors

    def test_error_log_can_be_disabled(self, tmp_path: Path) -> None:
        with LoggingService.configure(
            _settings(error_log=False),
            tmp_path,
            allow_transcripts=False,
            console_stream=io.StringIO(),
        ) as service:
            assert service.error_file is None

    def test_console_level_is_independent_of_file_level(self, tmp_path: Path) -> None:
        console = io.StringIO()
        with LoggingService.configure(
            _settings(console_level=LogLevel.ERROR),
            tmp_path,
            allow_transcripts=False,
            console_stream=console,
        ) as service:
            logging.getLogger("test.levels").info("only in the file")
            logging.getLogger("test.levels").error("also on the console")

        assert "only in the file" not in console.getvalue()
        assert "also on the console" in console.getvalue()
        assert "only in the file" in service.log_file.read_text(encoding="utf-8")

    def test_reconfiguring_does_not_duplicate_output(self, tmp_path: Path) -> None:
        first = LoggingService.configure(
            _settings(), tmp_path, allow_transcripts=False, console_stream=io.StringIO()
        )
        first.shutdown()

        with LoggingService.configure(
            _settings(), tmp_path, allow_transcripts=False, console_stream=io.StringIO()
        ) as second:
            logging.getLogger("test.dupes").info("written once")

        content = second.log_file.read_text(encoding="utf-8")
        assert content.count("written once") == 1

    def test_shutdown_is_idempotent(self, tmp_path: Path) -> None:
        service = LoggingService.configure(
            _settings(), tmp_path, allow_transcripts=False, console_stream=io.StringIO()
        )
        service.shutdown()
        service.shutdown()

    def test_handles_non_ascii_text(self, tmp_path: Path) -> None:
        with LoggingService.configure(
            _settings(), tmp_path, allow_transcripts=True, console_stream=io.StringIO()
        ) as service:
            logging.getLogger("test.unicode").info("recognised language: தமிழ் / हिन्दी")

        content = service.log_file.read_text(encoding="utf-8")
        assert "தமிழ்" in content
        assert "हिन्दी" in content


class TestTranscriptPrivacy:
    """Conversation content must not reach the log by default."""

    def test_transcript_records_are_dropped_by_default(self, tmp_path: Path) -> None:
        with LoggingService.configure(
            _settings(), tmp_path, allow_transcripts=False, console_stream=io.StringIO()
        ) as service:
            logging.getLogger("test.privacy").info(
                "transcript: the merger closes on Friday", extra=transcript_extra()
            )
            logging.getLogger("test.privacy").info("stt latency: 412 ms")

        content = service.log_file.read_text(encoding="utf-8")
        assert "merger closes on Friday" not in content
        assert "stt latency: 412 ms" in content

    def test_transcript_records_are_kept_when_allowed(self, tmp_path: Path) -> None:
        with LoggingService.configure(
            _settings(), tmp_path, allow_transcripts=True, console_stream=io.StringIO()
        ) as service:
            logging.getLogger("test.privacy").info(
                "transcript: the merger closes on Friday", extra=transcript_extra()
            )

        content = service.log_file.read_text(encoding="utf-8")
        assert "merger closes on Friday" in content

    def test_transcript_records_are_hidden_from_the_console(self, tmp_path: Path) -> None:
        console = io.StringIO()
        with LoggingService.configure(
            _settings(console_level=LogLevel.DEBUG),
            tmp_path,
            allow_transcripts=False,
            console_stream=console,
        ):
            logging.getLogger("test.privacy").info("secret text", extra=transcript_extra())

        assert "secret text" not in console.getvalue()
