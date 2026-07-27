"""Unit tests for console rendering."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from ai_interpreter.domain.entities import DeviceInfo
from ai_interpreter.domain.value_objects import DeviceKind
from ai_interpreter.infrastructure.config.settings import LoggingSection, LogLevel
from ai_interpreter.infrastructure.logging.setup import LoggingService
from ai_interpreter.presentation import console
from ai_interpreter.presentation.console import (
    format_device_table,
    level_bar,
    terminal_width,
)

pytestmark = pytest.mark.unit


class TestLevelBar:
    """Rendering an audio level as a text meter."""

    def test_silence_is_empty(self) -> None:
        bar = level_bar(0.0, width=10)
        assert bar == "░" * 10

    def test_full_scale_fills_the_bar(self) -> None:
        assert level_bar(1.0, width=10) == "█" * 10

    def test_always_returns_the_requested_width(self) -> None:
        for level in (0.0, 0.001, 0.05, 0.5, 1.0):
            assert len(level_bar(level, width=24)) == 24

    def test_scale_is_logarithmic(self) -> None:
        # A linear amplitude meter would spend most of its length on levels
        # the ear cannot tell apart, leaving normal speech pinned near zero.
        quiet = level_bar(0.01, width=30).count("█")
        loud = level_bar(0.1, width=30).count("█")

        assert 0 < quiet < loud < 30
        # 0.01 is -40 dB and 0.1 is -20 dB: a third and two thirds of the
        # -60 dB..0 dB span.
        assert quiet == pytest.approx(10, abs=1)
        assert loud == pytest.approx(20, abs=1)

    def test_survives_a_degenerate_width(self) -> None:
        assert len(level_bar(0.5, width=0)) == 1


class TestTerminalWidth:
    """Adapting output to the console size."""

    def test_holds_back_one_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A line that exactly fills the terminal wraps, and a wrapped line
        # cannot be overwritten with a carriage return.
        monkeypatch.setattr(
            console.shutil,
            "get_terminal_size",
            lambda fallback=None: type("S", (), {"columns": 63}),
        )
        assert terminal_width() == 62

    def test_enforces_a_minimum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            console.shutil,
            "get_terminal_size",
            lambda fallback=None: type("S", (), {"columns": 10}),
        )
        assert terminal_width() == 40

    def test_falls_back_when_the_size_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(fallback: object = None) -> None:
            raise OSError("no tty")

        monkeypatch.setattr(console.shutil, "get_terminal_size", _raise)
        assert terminal_width() == 79


class TestDeviceTable:
    """Formatting the device list."""

    @staticmethod
    def _device(name: str, **overrides: object) -> DeviceInfo:
        values: dict[str, object] = {
            "index": 1,
            "name": name,
            "kind": DeviceKind.INPUT,
            "max_channels": 2,
            "default_sample_rate": 48000.0,
            "host_api": "Windows WASAPI",
            "is_default": False,
        }
        values.update(overrides)
        return DeviceInfo(**values)  # type: ignore[arg-type]

    def test_reports_when_there_are_no_devices(self) -> None:
        assert format_device_table([]) == ["  (none found)"]

    def test_includes_name_host_api_and_rate(self) -> None:
        lines = format_device_table([self._device("Internal Microphone")])
        assert "Internal Microphone" in lines[0]
        assert "Windows WASAPI" in lines[0]
        assert "48000 Hz" in lines[0]

    def test_marks_the_default_device(self) -> None:
        lines = format_device_table([self._device("Internal Microphone", is_default=True)])
        assert "default" in lines[0]

    def test_marks_virtual_cables(self) -> None:
        lines = format_device_table([self._device("CABLE Output (VB-Audio Virtual Cable)")])
        assert "virtual cable" in lines[0]

    def test_aligns_host_api_column(self) -> None:
        lines = format_device_table(
            [
                self._device("A", host_api="MME"),
                self._device("B", host_api="Windows DirectSound"),
            ]
        )
        assert lines[0].index("]") == lines[1].index("]")


class TestQuietConsole:
    """Suppressing console logs while a live display is active."""

    @staticmethod
    def _settings() -> LoggingSection:
        return LoggingSection.model_validate(
            {
                "level": LogLevel.DEBUG,
                "console_level": LogLevel.INFO,
                "file_level": LogLevel.DEBUG,
                "directory": "logs",
                "max_bytes": 64_000,
                "backup_count": 1,
                "error_log": False,
            }
        )

    def test_suppresses_console_output(self, tmp_path: Path) -> None:
        stream = io.StringIO()
        with LoggingService.configure(
            self._settings(), tmp_path, allow_transcripts=False, console_stream=stream
        ) as service:
            with service.quiet_console():
                logging.getLogger("test.meter").info("noisy line")
            logging.getLogger("test.meter").info("visible line")

        output = stream.getvalue()
        assert "noisy line" not in output
        assert "visible line" in output

    def test_records_still_reach_the_file(self, tmp_path: Path) -> None:
        # The console is silenced, never the log. Nothing is lost.
        with (
            LoggingService.configure(
                self._settings(), tmp_path, allow_transcripts=False, console_stream=io.StringIO()
            ) as service,
            service.quiet_console(),
        ):
            logging.getLogger("test.meter").info("suppressed on screen only")

        assert "suppressed on screen only" in service.log_file.read_text(encoding="utf-8")

    def test_level_is_restored_after_an_exception(self, tmp_path: Path) -> None:
        stream = io.StringIO()
        with LoggingService.configure(
            self._settings(), tmp_path, allow_transcripts=False, console_stream=stream
        ) as service:
            with pytest.raises(RuntimeError), service.quiet_console():
                raise RuntimeError("capture failed")
            logging.getLogger("test.meter").info("after the failure")

        assert "after the failure" in stream.getvalue()
