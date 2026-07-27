"""Unit tests for filesystem layout resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_interpreter.infrastructure.paths import ApplicationPaths

pytestmark = pytest.mark.unit


class TestRootResolution:
    """Project root precedence rules."""

    def test_explicit_root_wins(self, tmp_path: Path) -> None:
        paths = ApplicationPaths.resolve(tmp_path)
        assert paths.root == tmp_path.resolve()

    def test_environment_variable_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_INTERPRETER_HOME", str(tmp_path))
        paths = ApplicationPaths.resolve()
        assert paths.root == tmp_path.resolve()

    def test_detects_root_from_package_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AI_INTERPRETER_HOME", raising=False)
        paths = ApplicationPaths.resolve()
        assert (paths.root / "src" / "ai_interpreter").is_dir()
        assert paths.default_config_file.is_file()


class TestDerivedPaths:
    """Directory layout derived from the root."""

    def test_all_directories_sit_under_root(self, tmp_path: Path) -> None:
        paths = ApplicationPaths.resolve(tmp_path)
        for directory in (
            paths.config_dir,
            paths.logs_dir,
            paths.models_dir,
            paths.recordings_dir,
        ):
            assert directory.is_relative_to(paths.root)

    def test_user_directories_default_to_the_user_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Personal overrides must survive reinstalling the application, so by
        # default they live in the user profile rather than the project folder.
        monkeypatch.delenv("AI_INTERPRETER_USER_HOME", raising=False)
        paths = ApplicationPaths.resolve(tmp_path)

        assert not paths.user_config_dir.is_relative_to(paths.root)
        assert "ai-interpreter" in str(paths.user_config_dir)

    def test_user_home_override_relocates_both_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Used by the portable build and by the test suite, so that neither
        # ever touches the real user profile.
        relocated = tmp_path / "portable"
        monkeypatch.setenv("AI_INTERPRETER_USER_HOME", str(relocated))
        paths = ApplicationPaths.resolve(tmp_path)

        assert paths.user_config_dir == relocated / "config"
        assert paths.user_data_dir == relocated / "data"

    def test_profile_file_path(self, tmp_path: Path) -> None:
        paths = ApplicationPaths.resolve(tmp_path)
        assert paths.profile_file("cpu_low").name == "cpu_low.yaml"


class TestDirectoryCreation:
    """Writable directories are created; read-only ones are not."""

    def test_creates_writable_directories(self, tmp_path: Path) -> None:
        paths = ApplicationPaths.resolve(tmp_path)
        paths.ensure_directories()

        assert paths.logs_dir.is_dir()
        assert paths.models_dir.is_dir()
        assert paths.recordings_dir.is_dir()

    def test_does_not_create_the_config_directory(self, tmp_path: Path) -> None:
        # A missing config directory means a broken installation. Creating an
        # empty one would turn a clear error into a confusing one.
        paths = ApplicationPaths.resolve(tmp_path)
        paths.ensure_directories()
        assert not paths.config_dir.exists()

    def test_is_idempotent(self, tmp_path: Path) -> None:
        paths = ApplicationPaths.resolve(tmp_path)
        paths.ensure_directories()
        paths.ensure_directories()
        assert paths.logs_dir.is_dir()


class TestProfileDiscovery:
    """Enumerating profiles on disk."""

    def test_lists_committed_profiles(self, paths: ApplicationPaths) -> None:
        assert set(paths.available_profiles()) == {"cpu_low", "cpu_high", "cuda"}

    def test_returns_empty_when_directory_missing(self, tmp_path: Path) -> None:
        assert ApplicationPaths.resolve(tmp_path).available_profiles() == ()
