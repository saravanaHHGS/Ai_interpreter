"""Filesystem layout resolution.

Every path the application uses is resolved once, here, and injected onwards.
No module ever calls ``Path("logs")`` on its own: relative paths depend on the
current working directory, which differs between running from a terminal,
running from the VS Code debugger, and running from the Phase 12 installer's
Start Menu shortcut - a classic source of "it writes logs somewhere else in
production" bugs.

Two distinct roots are used:

* the **project root**, holding code, configuration and downloaded models;
* the **user directory** under ``%APPDATA%``, holding personal overrides that
  must survive reinstalling the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from platformdirs import user_config_path, user_data_path

__all__ = ["APP_AUTHOR", "APP_NAME", "ApplicationPaths"]

APP_NAME: Final[str] = "ai-interpreter"
APP_AUTHOR: Final[str] = "HikeHealthGS"

# Escape hatch for the portable build in Phase 12, where everything lives
# beside the executable instead of in the user profile.
_HOME_ENV_VAR: Final[str] = "AI_INTERPRETER_HOME"

# Relocates the per-user directories. Two uses:
#   * the Phase 12 portable build, which must leave no trace in the profile;
#   * the test suite, which must never read or write the real %APPDATA%.
# Without this, a test that saves a user override would silently modify the
# developer's own configuration and make later runs order-dependent.
_USER_HOME_ENV_VAR: Final[str] = "AI_INTERPRETER_USER_HOME"


@dataclass(frozen=True, slots=True)
class ApplicationPaths:
    """Resolved locations of every directory the application uses.

    Args:
        root: Project root containing ``src``, ``config`` and ``docs``.
        config_dir: Directory holding ``default.yaml`` and ``profiles/``.
        logs_dir: Destination for rotating log files.
        models_dir: Cache for downloaded model weights.
        recordings_dir: Destination for test recordings.
        user_config_dir: Per-user directory for personal overrides.
        user_data_dir: Per-user directory for the cache and history databases.
    """

    root: Path
    config_dir: Path
    logs_dir: Path
    models_dir: Path
    recordings_dir: Path
    user_config_dir: Path
    user_data_dir: Path

    @classmethod
    def resolve(cls, root: Path | None = None) -> ApplicationPaths:
        """Determine the layout for this installation.

        Resolution order for the project root:

        1. the explicit ``root`` argument (used by tests);
        2. the ``AI_INTERPRETER_HOME`` environment variable (portable build);
        3. the package location, walking up out of ``src/ai_interpreter``.

        Args:
            root: Explicit project root, or ``None`` to detect it.

        Returns:
            The resolved paths. Directories are not created yet - call
            :meth:`ensure_directories` for that.
        """
        resolved_root = cls._resolve_root(root)
        user_config_dir, user_data_dir = cls._resolve_user_dirs()
        return cls(
            root=resolved_root,
            config_dir=resolved_root / "config",
            logs_dir=resolved_root / "logs",
            models_dir=resolved_root / "models",
            recordings_dir=resolved_root / "recordings",
            user_config_dir=user_config_dir,
            user_data_dir=user_data_dir,
        )

    @staticmethod
    def _resolve_user_dirs() -> tuple[Path, Path]:
        """Determine the per-user configuration and data directories.

        Returns:
            The configuration and data directories. ``AI_INTERPRETER_USER_HOME``
            overrides both; otherwise the platform conventions are used
            (``%APPDATA%`` and ``%LOCALAPPDATA%`` on Windows).
        """
        override = os.environ.get(_USER_HOME_ENV_VAR)
        if override:
            user_root = Path(override).resolve()
            return user_root / "config", user_root / "data"

        return (
            user_config_path(APP_NAME, APP_AUTHOR, roaming=True),
            user_data_path(APP_NAME, APP_AUTHOR, roaming=False),
        )

    @staticmethod
    def _resolve_root(root: Path | None) -> Path:
        """Pick the project root using the documented precedence order.

        Args:
            root: Explicit root, or ``None``.

        Returns:
            An absolute, symlink-free project root.
        """
        if root is not None:
            return root.resolve()

        env_home = os.environ.get(_HOME_ENV_VAR)
        if env_home:
            return Path(env_home).resolve()

        # paths.py lives at <root>/src/ai_interpreter/infrastructure/paths.py,
        # so the root is four levels up from this file.
        return Path(__file__).resolve().parents[3]

    @property
    def default_config_file(self) -> Path:
        """Location of the committed base configuration."""
        return self.config_dir / "default.yaml"

    @property
    def profiles_dir(self) -> Path:
        """Directory holding the hardware profile overrides."""
        return self.config_dir / "profiles"

    @property
    def user_config_file(self) -> Path:
        """Location of the user's personal override file, if they have one."""
        return self.user_config_dir / "config.yaml"

    def profile_file(self, profile: str) -> Path:
        """Path to a named hardware profile.

        Args:
            profile: Profile name such as ``"cpu_low"``.

        Returns:
            The profile's YAML path, which may not exist.
        """
        return self.profiles_dir / f"{profile}.yaml"

    def available_profiles(self) -> tuple[str, ...]:
        """Names of the hardware profiles present on disk.

        Returns:
            Profile names in alphabetical order, empty if the directory is
            missing.
        """
        if not self.profiles_dir.is_dir():
            return ()
        return tuple(sorted(path.stem for path in self.profiles_dir.glob("*.yaml")))

    def ensure_directories(self) -> None:
        """Create every writable directory the application needs.

        Read-only locations such as ``config`` are deliberately not created:
        if they are missing, something is wrong with the installation and the
        application should say so rather than start up misconfigured.
        """
        for directory in (
            self.logs_dir,
            self.models_dir,
            self.recordings_dir,
            self.user_config_dir,
            self.user_data_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
