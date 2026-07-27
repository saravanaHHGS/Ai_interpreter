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
from ai_interpreter.application.services.utterance_segmenter import UtteranceSegmenter
from ai_interpreter.domain.entities import DeviceInfo, HardwareInfo
from ai_interpreter.domain.errors import ConfigurationError
from ai_interpreter.domain.ports import VoiceActivityDetector
from ai_interpreter.domain.value_objects import DeviceKind, LanguageCode, SampleRate
from ai_interpreter.infrastructure.audio.capture.microphone import MicrophoneSource
from ai_interpreter.infrastructure.audio.devices import SounddeviceDeviceEnumerator
from ai_interpreter.infrastructure.audio.dsp import AudioPreprocessor
from ai_interpreter.infrastructure.audio.vad.energy import EnergyVad
from ai_interpreter.infrastructure.audio.vad.silero import SileroVad
from ai_interpreter.infrastructure.config.loader import ConfigLoader, ConfigLoadReport
from ai_interpreter.infrastructure.config.secrets import Secrets
from ai_interpreter.infrastructure.config.settings import AppSettings, Profile
from ai_interpreter.infrastructure.logging.setup import LoggingService
from ai_interpreter.infrastructure.models.hf_repository import HuggingFaceModelRepository
from ai_interpreter.infrastructure.models.registry import ModelRegistry
from ai_interpreter.infrastructure.paths import ApplicationPaths
from ai_interpreter.infrastructure.stt.faster_whisper import (
    FasterWhisperRecognizer,
    WhisperDecodeOptions,
)
from ai_interpreter.infrastructure.system.hardware import HardwareProbe

__all__ = ["Container"]

# Registry identifier of the neural voice activity detector.
_SILERO_MODEL_ID = "silero-vad"


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
        devices: Audio endpoint enumerator.
        models: Declared model catalogue.
        model_repository: Model download and cache.
    """

    paths: ApplicationPaths
    hardware: HardwareInfo
    selection: ProfileSelection
    settings: AppSettings
    config_report: ConfigLoadReport
    secrets: Secrets
    logging_service: LoggingService
    devices: SounddeviceDeviceEnumerator
    models: ModelRegistry
    model_repository: HuggingFaceModelRepository

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

        # 7. Adapters. Construction is cheap and side-effect free: no device
        #    is opened and no model is downloaded until something is used.
        models = ModelRegistry.load(paths.config_dir / "models.yaml")
        token = secrets.hf_token.get_secret_value() if secrets.hf_token else None

        return cls(
            paths=paths,
            hardware=hardware,
            selection=selection,
            settings=settings,
            config_report=config_report,
            secrets=secrets,
            logging_service=logging_service,
            devices=SounddeviceDeviceEnumerator(preferred_host_api=settings.audio.input.host_api),
            models=models,
            model_repository=HuggingFaceModelRepository(
                cache_dir=paths.models_dir,
                token=token,
                registry=tuple(models),
            ),
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

    # -- audio factories ---------------------------------------------------
    #
    # These are the only place the audio stack is assembled. Everything they
    # build is chosen from configuration, so switching to the energy detector
    # or a different microphone changes one YAML value and no code.
    def resolve_input_device(self, name_override: str | None = None) -> DeviceInfo:
        """Resolve which microphone to open.

        Args:
            name_override: Device name fragment from the command line, taking
                precedence over configuration.

        Returns:
            The endpoint to capture from.

        Raises:
            DeviceNotFoundError: If nothing matches, or the machine has no
                capture device.
        """
        configured = name_override or self.settings.audio.input.device
        return self.devices.resolve(configured, DeviceKind.INPUT)

    def create_microphone_source(self, device: DeviceInfo | None = None) -> MicrophoneSource:
        """Build a microphone capture source.

        Args:
            device: Endpoint to open, or ``None`` to resolve from
                configuration.

        Returns:
            An unopened capture source; call ``start()`` to begin.
        """
        audio_input = self.settings.audio.input
        return MicrophoneSource(
            device=device or self.resolve_input_device(),
            sample_rate=SampleRate(audio_input.sample_rate),
            frame_ms=audio_input.frame_ms,
        )

    def create_preprocessor(self, input_rate: SampleRate) -> AudioPreprocessor:
        """Build the resample, filter and gain chain.

        Args:
            input_rate: Rate audio arrives at, which may differ from the
                configured rate if the device negotiated another.

        Returns:
            The configured preprocessor.
        """
        processing = self.settings.audio.processing
        return AudioPreprocessor(
            input_rate=input_rate,
            output_rate=SampleRate(processing.target_sample_rate),
            high_pass_hz=processing.high_pass_hz,
            gain_db=self.settings.audio.input.gain_db,
        )

    def create_vad(self) -> VoiceActivityDetector:
        """Build the voice activity detector named in configuration.

        Downloads the Silero model on first use. The energy detector needs no
        model and is the documented fallback when downloading is impossible.

        Returns:
            The detector, already warmed up.

        Raises:
            ConfigurationError: If the configured provider is unknown.
            ModelDownloadError: If the Silero model cannot be obtained.
            ModelLoadError: If the model cannot be loaded.
        """
        provider = self.settings.vad.provider.strip().lower()

        if provider == "energy":
            detector: VoiceActivityDetector = EnergyVad()
        elif provider == "silero":
            descriptor = self.models.get(_SILERO_MODEL_ID)
            model_path = self.model_repository.ensure_file(descriptor, descriptor.files[0])
            detector = SileroVad(model_path=model_path, num_threads=1)
        else:
            msg = f"Unknown vad.provider {self.settings.vad.provider!r}. Valid: silero, energy"
            raise ConfigurationError(msg)

        warmup = getattr(detector, "warmup", None)
        if callable(warmup):
            warmup()
        return detector

    def source_language(self) -> LanguageCode:
        """Language the user speaks, from ``app.language_pair.source``.

        Returns:
            The configured source language.
        """
        return LanguageCode(self.settings.app.language_pair.source)

    def create_recognizer(self, language: LanguageCode | None = None) -> FasterWhisperRecognizer:
        """Build the speech recogniser named in configuration.

        Downloads the model on first use.

        Args:
            language: Language to decode, overriding configuration. ``None``
                resolves ``stt.language``, then the configured source
                language.

        Returns:
            The recogniser, not yet warmed up - call ``warmup()`` before
            timing anything, or the first decode pays the initialisation cost.

        Raises:
            ConfigurationError: If the provider or model name is unknown.
            ModelDownloadError: If the weights cannot be obtained.
        """
        stt = self.settings.stt
        provider = stt.provider.strip().lower()
        if provider != "faster_whisper":
            msg = f"Unknown stt.provider {stt.provider!r}. Valid: faster_whisper"
            raise ConfigurationError(msg)

        descriptor = self.models.get(f"whisper-{stt.model}")
        model_dir = self.model_repository.ensure(descriptor)

        chosen = language
        if chosen is None:
            chosen = LanguageCode(stt.language) if stt.language else self.source_language()

        return FasterWhisperRecognizer(
            model_dir=model_dir,
            model_id=descriptor.id,
            device=stt.device,
            compute_type=stt.compute_type,
            cpu_threads=stt.cpu_threads,
            language=chosen,
            options=WhisperDecodeOptions(
                beam_size=stt.beam_size,
                word_timestamps=stt.word_timestamps,
                min_confidence=stt.min_confidence,
            ),
        )

    def create_segmenter(
        self,
        sample_rate: SampleRate | None = None,
        language: LanguageCode | None = None,
    ) -> UtteranceSegmenter:
        """Build the utterance state machine from configuration.

        Args:
            sample_rate: Rate of the frames it will receive, or ``None`` to
                use the configured model rate.
            language: Language to tag utterances with, or ``None`` for the
                configured source language. An utterance's own tag takes
                precedence over the recogniser's default, so a caller
                overriding the language must override it here too - otherwise
                the tag silently wins and the override appears to do nothing.

        Returns:
            The configured segmenter.
        """
        vad = self.settings.vad
        rate = sample_rate or SampleRate(self.settings.audio.processing.target_sample_rate)
        return UtteranceSegmenter(
            sample_rate=rate,
            threshold=vad.threshold,
            min_speech_ms=vad.min_speech_ms,
            min_silence_ms=vad.min_silence_ms,
            pre_roll_ms=vad.pre_roll_ms,
            max_utterance_ms=vad.max_utterance_ms,
            language=language or self.source_language(),
        )

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
