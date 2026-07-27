"""Unit tests for the layered configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ai_interpreter.domain.errors import ConfigurationError
from ai_interpreter.infrastructure.config.loader import (
    ConfigLoader,
    collect_env_overrides,
    deep_merge,
)
from ai_interpreter.infrastructure.config.settings import Profile
from ai_interpreter.infrastructure.paths import ApplicationPaths

pytestmark = pytest.mark.unit


class TestDeepMerge:
    """Recursive mapping merge."""

    def test_overrides_scalar(self) -> None:
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_merges_nested_without_replacing_siblings(self) -> None:
        base = {"stt": {"model": "small", "beam_size": 1}}
        override = {"stt": {"model": "medium"}}
        assert deep_merge(base, override) == {"stt": {"model": "medium", "beam_size": 1}}

    def test_does_not_mutate_inputs(self) -> None:
        base = {"stt": {"model": "small"}}
        override = {"stt": {"model": "medium"}}
        deep_merge(base, override)
        assert base == {"stt": {"model": "small"}}
        assert override == {"stt": {"model": "medium"}}

    def test_replaces_dict_with_scalar(self) -> None:
        assert deep_merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}


class TestEnvOverrides:
    """Environment variable override collection."""

    def test_ignores_unrelated_variables(self) -> None:
        assert collect_env_overrides({"PATH": "C:/", "HOME": "C:/Users"}) == {}

    def test_builds_nested_structure(self) -> None:
        env = {"AI_INTERPRETER__STT__MODEL": "medium"}
        assert collect_env_overrides(env) == {"stt": {"model": "medium"}}

    def test_coerces_integers(self) -> None:
        env = {"AI_INTERPRETER__VAD__MIN_SILENCE_MS": "500"}
        assert collect_env_overrides(env) == {"vad": {"min_silence_ms": 500}}

    def test_coerces_booleans(self) -> None:
        env = {"AI_INTERPRETER__PRIVACY__LOG_TRANSCRIPTS": "true"}
        assert collect_env_overrides(env) == {"privacy": {"log_transcripts": True}}

    def test_coerces_null(self) -> None:
        env = {"AI_INTERPRETER__AUDIO__INPUT__DEVICE": "null"}
        assert collect_env_overrides(env) == {"audio": {"input": {"device": None}}}

    def test_keeps_non_json_strings_verbatim(self) -> None:
        env = {"AI_INTERPRETER__AUDIO__INPUT__DEVICE": "CABLE Input"}
        assert collect_env_overrides(env) == {"audio": {"input": {"device": "CABLE Input"}}}

    def test_merges_multiple_variables(self) -> None:
        env = {
            "AI_INTERPRETER__STT__MODEL": "base",
            "AI_INTERPRETER__STT__BEAM_SIZE": "3",
        }
        assert collect_env_overrides(env) == {"stt": {"model": "base", "beam_size": 3}}


class TestConfigLoader:
    """Full layered load against the committed configuration."""

    def test_loads_cpu_low_profile(self, paths: ApplicationPaths) -> None:
        settings, report = ConfigLoader(paths, environ={}).load(Profile.CPU_LOW)

        assert settings.app.profile is Profile.CPU_LOW
        # `base`, not `small`: measured at 1.70 s versus 5.88 s per utterance
        # on the 2-core target CPU. See docs/phases/phase-04-speech-to-text.md.
        assert settings.stt.model == "base"
        assert settings.stt.cpu_threads == 2
        assert settings.pipeline.inference_lane.value == "serial"
        assert len(report.sources) == 2

    def test_profile_overrides_defaults(self, paths: ApplicationPaths) -> None:
        low, _ = ConfigLoader(paths, environ={}).load(Profile.CPU_LOW)
        cuda, _ = ConfigLoader(paths, environ={}).load(Profile.CUDA)

        assert low.stt.device == "cpu"
        assert cuda.stt.device == "cuda"
        assert cuda.tts.provider == "kokoro"

    def test_environment_overrides_win(self, paths: ApplicationPaths) -> None:
        environ = {"AI_INTERPRETER__STT__MODEL": "tiny"}
        settings, report = ConfigLoader(paths, environ=environ).load(Profile.CPU_LOW)

        assert settings.stt.model == "tiny"
        assert "stt.model" in report.env_overrides

    def test_user_config_overrides_profile(self, paths: ApplicationPaths) -> None:
        paths.user_config_file.parent.mkdir(parents=True, exist_ok=True)
        paths.user_config_file.write_text(
            yaml.safe_dump({"vad": {"min_silence_ms": 500}}), encoding="utf-8"
        )
        settings, report = ConfigLoader(paths, environ={}).load(Profile.CPU_LOW)

        assert settings.vad.min_silence_ms == 500
        assert paths.user_config_file in report.sources

    def test_reads_requested_profile_as_auto(self, paths: ApplicationPaths) -> None:
        assert ConfigLoader(paths, environ={}).read_requested_profile() is Profile.AUTO

    def test_requested_profile_honours_environment(self, paths: ApplicationPaths) -> None:
        environ = {"AI_INTERPRETER__APP__PROFILE": "cuda"}
        assert ConfigLoader(paths, environ=environ).read_requested_profile() is Profile.CUDA

    def test_rejects_auto_profile_at_load(self, paths: ApplicationPaths) -> None:
        with pytest.raises(ConfigurationError, match="requires a concrete profile"):
            ConfigLoader(paths, environ={}).load(Profile.AUTO)

    def test_reports_missing_default_file(self, paths: ApplicationPaths) -> None:
        paths.default_config_file.unlink()
        with pytest.raises(ConfigurationError, match="Required configuration file not found"):
            ConfigLoader(paths, environ={}).load(Profile.CPU_LOW)

    def test_rejects_unknown_key(self, paths: ApplicationPaths) -> None:
        data = yaml.safe_load(paths.default_config_file.read_text(encoding="utf-8"))
        data["stt"]["typo_key"] = 1
        paths.default_config_file.write_text(yaml.safe_dump(data), encoding="utf-8")

        with pytest.raises(ConfigurationError, match="typo_key"):
            ConfigLoader(paths, environ={}).load(Profile.CPU_LOW)

    def test_rejects_invalid_yaml(self, paths: ApplicationPaths) -> None:
        paths.default_config_file.write_text("app: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            ConfigLoader(paths, environ={}).load(Profile.CPU_LOW)

    def test_rejects_out_of_range_value(self, paths: ApplicationPaths) -> None:
        environ = {"AI_INTERPRETER__TTS__SPEED": "9.0"}
        with pytest.raises(ConfigurationError, match=r"tts\.speed"):
            ConfigLoader(paths, environ=environ).load(Profile.CPU_LOW)

    def test_rejects_unsupported_sample_rate(self, paths: ApplicationPaths) -> None:
        environ = {"AI_INTERPRETER__AUDIO__INPUT__SAMPLE_RATE": "12345"}
        with pytest.raises(ConfigurationError, match="sample_rate"):
            ConfigLoader(paths, environ=environ).load(Profile.CPU_LOW)

    def test_rejects_identical_language_pair(self, paths: ApplicationPaths) -> None:
        environ = {"AI_INTERPRETER__APP__LANGUAGE_PAIR__SOURCE": "en"}
        with pytest.raises(ConfigurationError, match="must differ"):
            ConfigLoader(paths, environ=environ).load(Profile.CPU_LOW)

    def test_rejects_enabled_telemetry(self, paths: ApplicationPaths) -> None:
        environ = {"AI_INTERPRETER__PRIVACY__TELEMETRY": "true"}
        with pytest.raises(ConfigurationError, match="sends no telemetry"):
            ConfigLoader(paths, environ=environ).load(Profile.CPU_LOW)

    def test_rejects_cache_without_privacy_permission(self, paths: ApplicationPaths) -> None:
        environ = {"AI_INTERPRETER__PRIVACY__CACHE_TRANSLATIONS": "false"}
        with pytest.raises(ConfigurationError, match="privacy setting must permit caching"):
            ConfigLoader(paths, environ=environ).load(Profile.CPU_LOW)

    def test_settings_are_immutable(self, paths: ApplicationPaths) -> None:
        settings, _ = ConfigLoader(paths, environ={}).load(Profile.CPU_LOW)
        with pytest.raises(ValidationError):
            settings.stt.model = "medium"  # type: ignore[misc]


class TestProfileFilesExist:
    """Every profile referenced by the schema has a file behind it."""

    def test_all_concrete_profiles_have_files(self, paths: ApplicationPaths) -> None:
        available = set(paths.available_profiles())
        expected = {profile.value for profile in Profile if profile is not Profile.AUTO}
        assert expected <= available

    def test_every_profile_loads(self, paths: ApplicationPaths) -> None:
        for profile in Profile:
            if profile is Profile.AUTO:
                continue
            settings, _ = ConfigLoader(paths, environ={}).load(profile)
            assert settings.app.profile is profile


class TestLogDirectoryIsRelativeToRoot:
    """Log paths never depend on the working directory."""

    def test_directory_default_is_relative(self, paths: ApplicationPaths) -> None:
        settings, _ = ConfigLoader(paths, environ={}).load(Profile.CPU_LOW)
        assert not Path(settings.logging.directory).is_absolute()
