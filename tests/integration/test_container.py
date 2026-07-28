"""Integration tests for the composition root and the CLI.

These wire the real objects together - paths, hardware detection, profile
selection, configuration, logging - with only the project root redirected to a
temporary directory. They are the tests that prove the application actually
starts.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from ai_interpreter.app.container import Container
from ai_interpreter.cli import main
from ai_interpreter.domain.errors import ConfigurationError
from ai_interpreter.infrastructure.config.settings import Profile

pytestmark = pytest.mark.integration


class TestContainerBuild:
    """The full object graph constructs successfully."""

    def test_builds_with_auto_profile(self, project_root: Path) -> None:
        with Container.build(
            root=project_root, environ={}, console_stream=io.StringIO()
        ) as container:
            assert container.settings.app.name == "AI Interpreter"
            assert container.selection.profile is not Profile.AUTO
            assert container.selection.reason

    def test_creates_writable_directories(self, project_root: Path) -> None:
        with Container.build(
            root=project_root, environ={}, console_stream=io.StringIO()
        ) as container:
            assert container.paths.logs_dir.is_dir()
            assert container.paths.models_dir.is_dir()
            assert container.paths.recordings_dir.is_dir()

    def test_writes_a_log_file(self, project_root: Path) -> None:
        with Container.build(
            root=project_root, environ={}, console_stream=io.StringIO()
        ) as container:
            log_file = container.logging_service.log_file
            assert log_file.parent == project_root / "logs"

    def test_profile_override_is_honoured(self, project_root: Path) -> None:
        with Container.build(
            root=project_root,
            profile_override=Profile.CUDA,
            environ={},
            console_stream=io.StringIO(),
        ) as container:
            assert container.selection.profile is Profile.CUDA
            assert container.settings.stt.device == "cuda"
            assert container.selection.was_automatic is False

    def test_resolved_profile_is_recorded_in_settings(self, project_root: Path) -> None:
        with Container.build(
            root=project_root,
            profile_override=Profile.CPU_LOW,
            environ={},
            console_stream=io.StringIO(),
        ) as container:
            # Never "auto": downstream code always sees a concrete tier.
            assert container.settings.app.profile is Profile.CPU_LOW

    def test_missing_configuration_fails_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="Required configuration file not found"):
            Container.build(root=tmp_path, environ={}, console_stream=io.StringIO())

    def test_shutdown_is_idempotent(self, project_root: Path) -> None:
        container = Container.build(root=project_root, environ={}, console_stream=io.StringIO())
        container.shutdown()
        container.shutdown()


class TestCommandLine:
    """The CLI runs against the real project directory."""

    def test_check_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--check"])
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "environment check" in output
        assert "All required checks passed" in output

    def test_check_reports_hardware_and_profile(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--check"])
        output = capsys.readouterr().out

        assert "Selected profile" in output
        assert "Cores" in output
        assert "Speech-to-text" in output

    def test_print_config_emits_valid_yaml(self, capsys: pytest.CaptureFixture[str]) -> None:
        import yaml

        exit_code = main(["--print-config"])
        output = capsys.readouterr().out

        assert exit_code == 0
        body = "\n".join(line for line in output.splitlines() if not line.startswith("#"))
        parsed = yaml.safe_load(body)
        assert parsed["app"]["name"] == "AI Interpreter"
        assert "stt" in parsed

    def test_profile_flag_changes_the_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--print-config", "--profile", "cuda"])
        output = capsys.readouterr().out
        assert "device: cuda" in output

    def test_no_arguments_prints_guidance(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main([])
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "--check" in output

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])

        assert exit_info.value.code == 0
        assert "AI Interpreter" in capsys.readouterr().out
