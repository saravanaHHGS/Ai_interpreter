"""The composition root.

Dependency injection in this project is done by hand, in this one file. No
container library, no decorators, no registration calls - just explicit
construction in a known order.

Why not ``dependency-injector`` or ``punq``? Both resolve dependencies at
runtime through reflection, which means mypy cannot verify the object graph
and a wiring mistake surfaces as an exception minutes into a session. A
hand-written root is fully statically checked, reads top to bottom, and costs
one file. The benefits people actually want from DI - swappable
implementations and testable components - come from the *ports*, not from the
container.

Startup order matters and is deliberate:

1. **Paths** - nothing can be read before we know where things are.
2. **Hardware** - profile selection needs it, and it must not require config.
3. **Profile** - resolves ``auto`` into a concrete tier.
4. **Configuration** - merged and validated for that tier.
5. **Logging** - configured only now, because its levels and privacy
   behaviour come from configuration. Steps 1-4 therefore buffer their
   messages, which is why failures there raise :class:`ConfigurationError`
   with a complete message rather than relying on a log line.
6. **Secrets** - last, and never logged.

Later phases extend this file, and only this file: Phase 3 adds the audio
device adapters, Phase 4 the recogniser, Phase 5 the translator, Phase 6 the
synthesizer. No other module ever chooses an implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO

from ai_interpreter.application.services.profile_selector import (
    ProfileSelection,
    ProfileSelector,
)
from ai_interpreter.domain.entities import HardwareInfo
from ai_interpreter.infrastructure.config.loader import ConfigLoader, ConfigLoadReport
from ai_interpreter.infrastructure.config.secrets import Secrets
from ai_interpreter.infrastructure.config.settings import AppSettings, Profile
from ai_interpreter.infrastructure.logging.setup import LoggingService
from ai_interpreter.infrastructure.paths import ApplicationPaths
from ai_interpreter.infrastructure.system.hardware import HardwareProbe

__all__ = ["Container"]


@dataclass(frozen=True, slots=True)
class Container:
    """The fully constructed application object graph.

    Args:
        paths: Resolved filesystem layout.
        hardware: Machine snapshot taken at startup.
        selection: Profile that was chosen and why.
        settings: Validated configuration.
        config_report: Provenance of the configuration.
        secrets: Credentials loaded from ``.env``.
        logging_service: Owner of the logging configuration.
    """

    paths: ApplicationPaths
    hardware: HardwareInfo
    selection: ProfileSelection
    settings: AppSettings
    config_report: ConfigLoadReport
    secrets: Secrets
    logging_service: LoggingService

    @classmethod
    def build(
        cls,
        *,
        root: Path | None = None,
        profile_override: Profile | None = None,
        environ: dict[str, str] | None = None,
        console_stream: TextIO | None = None,
    ) -> Container:
        """Construct the application graph.

        Args:
            root: Explicit project root, or ``None`` to detect it.
            profile_override: Profile forced by the command line, taking
                precedence over the configured value.
            environ: Environment used for configuration overrides, or ``None``
                for the real one.
            console_stream: Stream for console log output, or ``None`` for
                ``sys.stderr``. Tests redirect it to keep output clean.

        Returns:
            The constructed container.

        Raises:
            ConfigurationError: If configuration is missing, malformed, or
                fails validation.
        """
        # 1. Paths
        paths = ApplicationPaths.resolve(root)
        paths.ensure_directories()

        # 2. Hardware
        hardware = HardwareProbe().detect(reference_path=paths.root)

        # 3. Profile
        loader = ConfigLoader(paths, environ=environ)
        requested = (
            profile_override if profile_override is not None else loader.read_requested_profile()
        )
        selector = ProfileSelector(paths.available_profiles())
        selection = selector.select(requested, hardware)

        # 4. Configuration
        settings, config_report = loader.load(selection.profile)

        # 5. Logging
        logs_dir = cls._resolve_logs_dir(paths, settings)
        logging_service = LoggingService.configure(
            settings.logging,
            logs_dir,
            allow_transcripts=settings.privacy.log_transcripts,
            console_stream=console_stream,
        )

        # 6. Secrets
        secrets = Secrets.load(paths.root / ".env")

        return cls(
            paths=paths,
            hardware=hardware,
            selection=selection,
            settings=settings,
            config_report=config_report,
            secrets=secrets,
            logging_service=logging_service,
        )

    @staticmethod
    def _resolve_logs_dir(paths: ApplicationPaths, settings: AppSettings) -> Path:
        """Resolve the configured log directory against the project root.

        Args:
            paths: Application paths.
            settings: Validated configuration.

        Returns:
            An absolute log directory. A relative value in configuration is
            interpreted relative to the project root, never to the current
            working directory, so logs land in the same place however the
            application was started.
        """
        configured = Path(settings.logging.directory)
        if configured.is_absolute():
            return configured
        return paths.root / configured

    def shutdown(self) -> None:
        """Release everything the container owns, in reverse build order."""
        self.logging_service.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.shutdown()
