"""Secret loading, kept strictly separate from ordinary configuration.

Two storage mechanisms with two different security properties:

* ``config/*.yaml`` is committed to git, safe to share, and describes
  behaviour;
* ``.env`` is git-ignored, never shared, and holds credentials.

Splitting them structurally - rather than by convention - means a secret
cannot end up in the repository by accident, because the YAML schema has
``extra="forbid"`` and simply has no field for one.

Values are wrapped in :class:`~pydantic.SecretStr`, whose ``repr`` prints
``**********``. A token therefore cannot leak through a stack trace or a
debug log line, which is the most common way credentials escape.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Secrets"]


class Secrets(BaseSettings):
    """Credentials read from ``.env`` and the process environment.

    Every field is optional. The application is local-first: with an empty
    ``.env`` it runs completely offline apart from one-time model downloads.

    Args:
        hf_token: Hugging Face token, needed only for gated repositories.
        nvidia_nim_api_key: NVIDIA NIM key, needed only if the opt-in cloud
            translation provider is explicitly enabled.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AI_INTERPRETER_",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    hf_token: SecretStr | None = Field(default=None)
    nvidia_nim_api_key: SecretStr | None = Field(default=None)

    @classmethod
    def load(cls, env_file: Path | None = None) -> Secrets:
        """Load secrets, optionally from an explicit ``.env`` path.

        Args:
            env_file: Path to an environment file, or ``None`` to use ``.env``
                in the working directory.

        Returns:
            The loaded secrets.
        """
        if env_file is None:
            return cls()
        return cls(_env_file=env_file)

    @property
    def has_hf_token(self) -> bool:
        """Whether a Hugging Face token is configured."""
        return self.hf_token is not None and bool(self.hf_token.get_secret_value())

    @property
    def has_nim_key(self) -> bool:
        """Whether an NVIDIA NIM key is configured."""
        return self.nvidia_nim_api_key is not None and bool(
            self.nvidia_nim_api_key.get_secret_value()
        )
