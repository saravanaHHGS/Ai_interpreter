"""Layered configuration loader.

Configuration is assembled from four layers, each overriding the last:

1. ``config/default.yaml`` - committed defaults, the source of truth.
2. ``config/profiles/<profile>.yaml`` - hardware tier overrides.
3. ``%APPDATA%/ai-interpreter/config.yaml`` - the user's personal overrides.
4. ``AI_INTERPRETER__SECTION__KEY`` environment variables - per-run overrides.

Layer 4 exists so a support session can say "run it once with
``AI_INTERPRETER__STT__MODEL=base``" without editing any file, and so the
Phase 11 test suite can vary one setting without maintaining fixture files.

The loader deliberately performs two passes. Which profile applies depends on
detected hardware, and hardware detection is not the config layer's job - so
pass one reads the *requested* profile, the caller resolves it against the
machine, and pass two merges the resulting profile file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from ai_interpreter.domain.errors import ConfigurationError
from ai_interpreter.infrastructure.config.settings import AppSettings, Profile
from ai_interpreter.infrastructure.paths import ApplicationPaths

__all__ = ["ENV_PREFIX", "ConfigLoadReport", "ConfigLoader", "deep_merge"]

logger = logging.getLogger(__name__)

# Double underscore separates nesting levels, because single underscores
# already appear inside key names (``max_utterance_ms``).
ENV_PREFIX: Final[str] = "AI_INTERPRETER__"
_ENV_SEPARATOR: Final[str] = "__"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new mapping.

    Nested dictionaries are merged key by key rather than replaced wholesale,
    so a profile may override ``stt.model`` without having to restate every
    other key in the ``stt`` section.

    Args:
        base: Lower-priority mapping.
        override: Higher-priority mapping.

    Returns:
        A new merged mapping. Neither input is modified.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _coerce_env_value(raw: str) -> Any:
    """Convert an environment variable string into a typed Python value.

    Environment variables are always strings, but the schema expects integers,
    booleans and nulls. JSON covers all of those with well-defined rules
    (``true``, ``350``, ``null``, ``0.5``), and anything that is not valid JSON
    is kept as a plain string - which is exactly right for values like
    ``small`` or ``CABLE Input``.

    Args:
        raw: Raw environment variable value.

    Returns:
        The parsed value, or the original string when it is not JSON.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    """Assign a value deep inside a nested mapping, creating levels as needed.

    Args:
        target: Mapping to modify in place.
        path: Key path from outermost to innermost.
        value: Value to assign at the innermost key.
    """
    cursor = target
    for key in path[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[path[-1]] = value


def collect_env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a nested override mapping from ``AI_INTERPRETER__*`` variables.

    Args:
        environ: Environment to read, or ``None`` for the real one. Passing an
            explicit mapping keeps the tests free of global state.

    Returns:
        A nested mapping, empty when no matching variables are set.
    """
    source = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}

    for name, raw_value in source.items():
        if not name.startswith(ENV_PREFIX):
            continue
        remainder = name[len(ENV_PREFIX) :]
        if not remainder:
            continue
        path = [part.lower() for part in remainder.split(_ENV_SEPARATOR) if part]
        if not path:
            continue
        _set_nested(overrides, path, _coerce_env_value(raw_value))

    return overrides


@dataclass(frozen=True, slots=True)
class ConfigLoadReport:
    """Provenance of a loaded configuration.

    Printed by ``--check`` and ``--print-config``. When someone asks "why is it
    using the medium model?", this answers it in one line instead of an hour of
    guessing.

    Args:
        sources: Files that contributed, in the order they were applied.
        profile: Hardware profile that was applied.
        env_overrides: Dotted key paths overridden by environment variables.
    """

    sources: tuple[Path, ...]
    profile: Profile
    env_overrides: tuple[str, ...] = field(default=())


class ConfigLoader:
    """Assembles :class:`AppSettings` from the configuration layers.

    Args:
        paths: Resolved application paths.
        environ: Environment to read overrides from, or ``None`` for the real
            one.
    """

    def __init__(self, paths: ApplicationPaths, environ: dict[str, str] | None = None) -> None:
        self._paths = paths
        self._environ = environ

    # -- public API --------------------------------------------------------
    def read_requested_profile(self) -> Profile:
        """Read ``app.profile`` before hardware-specific merging.

        Returns:
            The profile requested by configuration, possibly
            :attr:`Profile.AUTO`.

        Raises:
            ConfigurationError: If the base file is missing or the value is not
                a known profile.
        """
        data = self._read_base_layers()
        raw = data.get("app", {}).get("profile", Profile.AUTO.value)
        try:
            return Profile(raw)
        except ValueError as exc:
            valid = ", ".join(profile.value for profile in Profile)
            msg = f"app.profile must be one of: {valid}. Got {raw!r}."
            raise ConfigurationError(msg) from exc

    def load(self, profile: Profile) -> tuple[AppSettings, ConfigLoadReport]:
        """Load and validate the full configuration for a resolved profile.

        Args:
            profile: Concrete profile to apply. Must not be
                :attr:`Profile.AUTO`; the caller resolves that first using
                detected hardware.

        Returns:
            The validated settings and a report describing where they came
            from.

        Raises:
            ConfigurationError: If a file is missing or malformed, or the
                merged result fails schema validation.
        """
        if profile is Profile.AUTO:
            msg = (
                "ConfigLoader.load requires a concrete profile. Resolve "
                "Profile.AUTO with ProfileSelector before calling load()."
            )
            raise ConfigurationError(msg)

        sources: list[Path] = []

        default_file = self._paths.default_config_file
        data = self._load_yaml_file(default_file, required=True)
        sources.append(default_file)

        profile_file = self._paths.profile_file(profile.value)
        profile_data = self._load_yaml_file(profile_file, required=True)
        data = deep_merge(data, profile_data)
        sources.append(profile_file)

        user_file = self._paths.user_config_file
        if user_file.is_file():
            data = deep_merge(data, self._load_yaml_file(user_file, required=False))
            sources.append(user_file)

        env_overrides = collect_env_overrides(self._environ)
        if env_overrides:
            data = deep_merge(data, env_overrides)

        # The resolved profile is recorded in the settings so that anything
        # reading app.profile later sees the concrete tier, never "auto".
        data = deep_merge(data, {"app": {"profile": profile.value}})

        settings = self._validate(data, sources)
        report = ConfigLoadReport(
            sources=tuple(sources),
            profile=profile,
            env_overrides=tuple(sorted(_flatten_keys(env_overrides))),
        )
        logger.debug(
            "Configuration loaded (profile=%s, sources=%d, env_overrides=%d)",
            profile.value,
            len(sources),
            len(report.env_overrides),
        )
        return settings, report

    # -- internals ---------------------------------------------------------
    def _read_base_layers(self) -> dict[str, Any]:
        """Merge everything except the profile file, for the first pass.

        Returns:
            The merged mapping of defaults, user overrides and environment
            variables.
        """
        data = self._load_yaml_file(self._paths.default_config_file, required=True)
        user_file = self._paths.user_config_file
        if user_file.is_file():
            data = deep_merge(data, self._load_yaml_file(user_file, required=False))
        return deep_merge(data, collect_env_overrides(self._environ))

    def _load_yaml_file(self, path: Path, *, required: bool) -> dict[str, Any]:
        """Read one YAML file into a mapping.

        Args:
            path: File to read.
            required: Whether a missing file is an error.

        Returns:
            The parsed mapping, empty when the file is absent and optional.

        Raises:
            ConfigurationError: If the file is missing while required, cannot
                be read, is not valid YAML, or does not contain a mapping.
        """
        if not path.is_file():
            if required:
                available = ", ".join(self._paths.available_profiles()) or "none found"
                msg = (
                    f"Required configuration file not found: {path}\n"
                    f"Available profiles: {available}"
                )
                raise ConfigurationError(msg)
            return {}

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Could not read configuration file {path}: {exc}"
            raise ConfigurationError(msg) from exc

        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            msg = f"Invalid YAML in {path}: {exc}"
            raise ConfigurationError(msg) from exc

        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            msg = f"Configuration file {path} must contain a mapping, got {type(parsed).__name__}"
            raise ConfigurationError(msg)
        return parsed

    @staticmethod
    def _validate(data: dict[str, Any], sources: list[Path]) -> AppSettings:
        """Validate the merged mapping against the schema.

        Args:
            data: Merged configuration mapping.
            sources: Files that contributed, for the error message.

        Returns:
            The validated settings.

        Raises:
            ConfigurationError: If validation fails, with every problem listed
                at once rather than one per restart.
        """
        try:
            return AppSettings.model_validate(data)
        except ValidationError as exc:
            problems = "\n".join(
                f"  - {'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
                for error in exc.errors()
            )
            source_list = "\n".join(f"  - {path}" for path in sources)
            msg = (
                f"Configuration is invalid.\n\nProblems:\n{problems}\n\n"
                f"Files loaded:\n{source_list}"
            )
            raise ConfigurationError(msg) from exc


def _flatten_keys(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested mapping into dotted key paths.

    Args:
        data: Nested mapping.
        prefix: Accumulated prefix used during recursion.

    Returns:
        Dotted paths to every leaf value.
    """
    keys: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(_flatten_keys(value, path))
        else:
            keys.append(path)
    return keys
