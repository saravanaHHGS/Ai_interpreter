"""Unit tests for secret loading and redaction."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_interpreter.infrastructure.config.secrets import Secrets

pytestmark = pytest.mark.unit


class TestSecretLoading:
    """Reading credentials from a .env file."""

    def test_absent_file_yields_empty_secrets(self, tmp_path: Path) -> None:
        secrets = Secrets.load(tmp_path / "missing.env")

        assert secrets.has_hf_token is False
        assert secrets.has_nim_key is False

    def test_reads_values_from_env_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "AI_INTERPRETER_HF_TOKEN=hf_example_token\n"
            "AI_INTERPRETER_NVIDIA_NIM_API_KEY=nvapi_example\n",
            encoding="utf-8",
        )

        secrets = Secrets.load(env_file)

        assert secrets.has_hf_token is True
        assert secrets.has_nim_key is True
        assert secrets.hf_token is not None
        assert secrets.hf_token.get_secret_value() == "hf_example_token"

    def test_empty_value_counts_as_unset(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("AI_INTERPRETER_HF_TOKEN=\n", encoding="utf-8")

        assert Secrets.load(env_file).has_hf_token is False

    def test_ignores_unrelated_keys(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("SOME_OTHER_APP_KEY=value\n", encoding="utf-8")

        assert Secrets.load(env_file).has_hf_token is False


class TestRedaction:
    """A token must never appear in a log line or traceback."""

    def test_repr_is_masked(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("AI_INTERPRETER_HF_TOKEN=hf_super_secret\n", encoding="utf-8")

        secrets = Secrets.load(env_file)

        assert "hf_super_secret" not in repr(secrets)
        assert "hf_super_secret" not in str(secrets)

    def test_value_is_available_when_explicitly_requested(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("AI_INTERPRETER_HF_TOKEN=hf_super_secret\n", encoding="utf-8")

        secrets = Secrets.load(env_file)

        assert secrets.hf_token is not None
        assert secrets.hf_token.get_secret_value() == "hf_super_secret"
