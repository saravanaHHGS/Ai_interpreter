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

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO, TypeVar

from ai_interpreter.application.services.cached_translator import CachedTranslator
from ai_interpreter.application.services.profile_selector import (
    ProfileSelection,
    ProfileSelector,
)
from ai_interpreter.application.services.utterance_segmenter import UtteranceSegmenter
from ai_interpreter.domain.entities import DeviceInfo, HardwareInfo, ModelDescriptor
from ai_interpreter.domain.errors import ConfigurationError
from ai_interpreter.domain.ports import SpeechRecognizer, Translator, VoiceActivityDetector
from ai_interpreter.domain.value_objects import (
    DeviceKind,
    LanguageCode,
    LanguagePair,
    SampleRate,
)
from ai_interpreter.infrastructure.audio.capture.microphone import MicrophoneSource
from ai_interpreter.infrastructure.audio.devices import SounddeviceDeviceEnumerator
from ai_interpreter.infrastructure.audio.dsp import AudioPreprocessor
from ai_interpreter.infrastructure.audio.playback.virtual_cable import VirtualCableSink
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
    build_initial_prompt,
)
from ai_interpreter.infrastructure.stt.onnx_metadata import ensure_onnx_metadata
from ai_interpreter.infrastructure.stt.sherpa_nemo import (
    SherpaNemoCtcRecognizer,
    SherpaNemoStreamingRecognizer,
)
from ai_interpreter.infrastructure.system.hardware import HardwareProbe
from ai_interpreter.infrastructure.translation.cache import LruTranslationCache
from ai_interpreter.infrastructure.translation.indictrans2 import IndicTrans2Translator
from ai_interpreter.infrastructure.tts.sherpa_vits import SherpaVitsSynthesizer

__all__ = ["Container"]

logger = logging.getLogger(__name__)

# Registry identifier of the neural voice activity detector.
_SILERO_MODEL_ID = "silero-vad"

_T = TypeVar("_T")


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

    # Built model components, reused across sessions. Loading a model costs
    # seconds; a UI Start after Stop must not pay it again. Keyed by what
    # makes a component unique (model id, language, options); released in
    # shutdown(). The dict itself is mutable inside this frozen dataclass -
    # the *binding* is what frozen protects.
    component_cache: dict[tuple[object, ...], object] = field(default_factory=dict)

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

    def _cached(self, key: tuple[object, ...], factory: Callable[[], _T]) -> _T:
        """Return the cached component for a key, building it once.

        Args:
            key: What makes the component unique.
            factory: Builds the component on a cache miss.

        Returns:
            The shared component instance.
        """
        component = self.component_cache.get(key)
        if component is None:
            component = factory()
            self.component_cache[key] = component
        return component  # type: ignore[return-value]

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

        def build() -> VoiceActivityDetector:
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

        # Reuse is safe: CaptureSession calls reset() before every run, so a
        # recurrent detector never carries one session's state into the next.
        return self._cached(("vad", provider), build)

    def source_language(self) -> LanguageCode:
        """Language the user speaks, from ``app.language_pair.source``.

        Returns:
            The configured source language.
        """
        return LanguageCode(self.settings.app.language_pair.source)

    def create_recognizer(
        self,
        language: LanguageCode | None = None,
        *,
        word_timestamps: bool | None = None,
    ) -> SpeechRecognizer:
        """Build the speech recogniser for a language.

        Downloads the model on first use, and dispatches on the model's
        declared ``runtime`` - CTranslate2 Whisper and sherpa-onnx NeMo
        conformers are constructed from the same registry entry format, so
        adding a runtime touches this method and nothing else.

        Args:
            language: Language to decode, overriding configuration. ``None``
                resolves ``stt.language``, then the configured source
                language.
            word_timestamps: Override ``stt.word_timestamps`` for this
                recogniser. The code-switch fallback needs per-word timings
                for transcript fusion whatever the global setting says.
                Sherpa CTC models timestamp every token regardless, so the
                override only affects Whisper.

        Returns:
            The recogniser, not yet warmed up - call ``warmup()`` before
            timing anything, or the first decode pays the initialisation cost.

        Raises:
            ConfigurationError: If the model name or runtime is unknown.
            ModelDownloadError: If the weights cannot be obtained.
            ModelLoadError: If an ONNX model needs metadata patching and the
                patch fails.
        """
        stt = self.settings.stt
        chosen = language
        if chosen is None:
            chosen = LanguageCode(stt.language) if stt.language else self.source_language()

        descriptor = self._resolve_stt_descriptor(self._model_name_for(chosen))
        effective_timestamps = stt.word_timestamps if word_timestamps is None else word_timestamps

        def build() -> SpeechRecognizer:
            if descriptor.runtime == "ctranslate2":
                return self._create_whisper(
                    descriptor, chosen, word_timestamps=effective_timestamps
                )
            if descriptor.runtime in ("sherpa-nemo-ctc", "sherpa-nemo-ctc-streaming"):
                return self._create_sherpa(descriptor, chosen)

            msg = (
                f"Model {descriptor.id!r} declares unknown runtime {descriptor.runtime!r}. "
                "Known: ctranslate2, sherpa-nemo-ctc, sherpa-nemo-ctc-streaming"
            )
            raise ConfigurationError(msg)

        return self._cached(("stt", descriptor.id, chosen.code, effective_timestamps), build)

    def _create_whisper(
        self,
        descriptor: ModelDescriptor,
        language: LanguageCode,
        word_timestamps: bool,
    ) -> FasterWhisperRecognizer:
        """Build a CTranslate2 Whisper recogniser.

        Args:
            descriptor: Registry entry with runtime ``ctranslate2``.
            language: Language to decode.
            word_timestamps: Whether to emit per-word segments.

        Returns:
            The recogniser.
        """
        stt = self.settings.stt
        model_dir = self.model_repository.ensure(descriptor)

        # "*" means the full multilingual set; anything else is a fine-tune
        # restricted to the languages it was trained on.
        supported = None if "*" in descriptor.languages else frozenset(descriptor.languages)

        # Bias the decoder toward the user's own vocabulary: explicit
        # hotwords plus the glossary's canonical terms, which are by
        # definition words the user speaks.
        prompt = build_initial_prompt([*stt.hotwords, *self.settings.translation.glossary.keys()])

        return FasterWhisperRecognizer(
            model_dir=model_dir,
            model_id=descriptor.id,
            device=stt.device,
            compute_type=stt.compute_type,
            cpu_threads=stt.cpu_threads,
            language=language,
            options=WhisperDecodeOptions(
                beam_size=stt.beam_size,
                word_timestamps=word_timestamps,
                min_confidence=stt.min_confidence,
                initial_prompt=prompt,
            ),
            supported_languages=supported,
        )

    def _create_sherpa(
        self, descriptor: ModelDescriptor, language: LanguageCode
    ) -> SpeechRecognizer:
        """Build a sherpa-onnx NeMo conformer recogniser.

        Args:
            descriptor: Registry entry with a ``sherpa-nemo-ctc*`` runtime.
            language: Language to decode; must be one the model serves.

        Returns:
            The offline chunked recogniser or the online streaming one,
            depending on the declared runtime.

        Raises:
            ConfigurationError: If the language is not served, or the entry
                does not name its model and token files.
        """
        if not descriptor.supports(language):
            msg = (
                f"Model {descriptor.id!r} does not support {language.english_name}; "
                f"it serves: {', '.join(descriptor.languages)}"
            )
            raise ConfigurationError(msg)
        if len(descriptor.files) < 2:
            msg = (
                f"Model {descriptor.id!r} must list its model file and token table "
                "in 'files' (in that order)"
            )
            raise ConfigurationError(msg)

        model_path = self.model_repository.ensure_file(descriptor, descriptor.files[0])
        tokens_path = self.model_repository.ensure_file(descriptor, descriptor.files[1])

        if descriptor.onnx_metadata:
            model_path = ensure_onnx_metadata(
                source=model_path,
                patched_dir=self.paths.models_dir / "patched" / descriptor.id,
                required=dict(descriptor.onnx_metadata),
            )

        threads = self.settings.stt.cpu_threads
        if descriptor.runtime == "sherpa-nemo-ctc-streaming":
            return SherpaNemoStreamingRecognizer(
                model_path=model_path,
                tokens_path=tokens_path,
                model_id=descriptor.id,
                language=language,
                cpu_threads=threads,
            )
        return SherpaNemoCtcRecognizer(
            model_path=model_path,
            tokens_path=tokens_path,
            model_id=descriptor.id,
            language=language,
            cpu_threads=threads,
            chunk_seconds=self.settings.stt.chunk_ms / 1000.0,
        )

    def create_translator(self, pair: LanguagePair | None = None) -> Translator:
        """Build the translator for a direction, cache included.

        Downloads the model on first use (~850 MB per direction, quantised to
        roughly 220 MB of RAM at load).

        Args:
            pair: Direction to translate, or ``None`` for the configured
                ``app.language_pair``.

        Returns:
            A ``Translator``; when the cache is enabled it is a
            :class:`CachedTranslator` wrapping the engine, and callers cannot
            tell the difference - which is the point.

        Raises:
            ConfigurationError: If the direction has no configured model.
            ModelDownloadError: If the weights cannot be obtained.
        """
        chosen = pair or LanguagePair.of(
            self.settings.app.language_pair.source,
            self.settings.app.language_pair.target,
        )
        direction = "en-indic" if chosen.source.code == "en" else "indic-en"

        translation = self.settings.translation
        model_name = translation.models.get(direction)
        if not model_name:
            configured = ", ".join(sorted(translation.models)) or "none"
            msg = (
                f"No translation model configured for direction {direction!r} "
                f"(needed for {chosen}). Configured directions: {configured}."
            )
            raise ConfigurationError(msg)

        def build() -> Translator:
            descriptor = self.models.get(model_name)
            model_dir = self.model_repository.ensure(descriptor)

            engine = IndicTrans2Translator(
                model_dir=model_dir,
                model_id=descriptor.id,
                direction=direction,
                cpu_threads=self.settings.stt.cpu_threads,
                beam_size=translation.beam_size,
                max_input_chars=translation.max_input_chars,
            )
            if not translation.cache.enabled:
                return engine
            return CachedTranslator(
                inner=engine,
                cache=LruTranslationCache(max_entries=translation.cache.max_entries),
            )

        # Reusing the translator across sessions also carries its LRU cache
        # forward - repeated sentences stay at 0 ms in the next session too.
        return self._cached(("mt", direction), build)

    def resolve_output_device(self, name_override: str | None = None) -> DeviceInfo:
        """Resolve which playback endpoint to write into.

        Args:
            name_override: Device name fragment from the command line, taking
                precedence over ``audio.output.device``. For the virtual
                microphone this is ``"CABLE Input"``.

        Returns:
            The endpoint to open.

        Raises:
            DeviceNotFoundError: If nothing matches, or the machine has no
                playback device.
        """
        configured = name_override or self.settings.audio.output.device
        return self.devices.resolve(configured, DeviceKind.OUTPUT)

    def create_audio_sink(
        self,
        device: DeviceInfo | None = None,
        device_name: str | None = None,
    ) -> VirtualCableSink:
        """Build the playback sink for synthesised speech.

        Args:
            device: Endpoint to open, or ``None`` to resolve one.
            device_name: Name fragment used when ``device`` is ``None``.

        Returns:
            An unopened sink; call ``open()`` to start the stream.
        """
        output = self.settings.audio.output
        return VirtualCableSink(
            device=device or self.resolve_output_device(device_name),
            output_rate=SampleRate(output.sample_rate),
            jitter_buffer_ms=output.jitter_buffer_ms,
        )

    def create_synthesizer(self, language: LanguageCode | None = None) -> SherpaVitsSynthesizer:
        """Build the voice for a language.

        Downloads the voice model on first use.

        Args:
            language: Language to speak, or ``None`` for the configured
                target language.

        Returns:
            The synthesizer, not yet warmed up.

        Raises:
            ConfigurationError: If no voice is configured for the language -
                the caller decides whether that means captions instead of
                audio (the designed Phase 1 fallback) or a hard failure.
            ModelDownloadError: If the voice cannot be obtained.
        """
        tts = self.settings.tts
        chosen = language or LanguageCode(self.settings.app.language_pair.target)

        model_name = tts.voices.get(chosen.code)
        if not model_name:
            configured = ", ".join(sorted(tts.voices)) or "none"
            msg = (
                f"No voice is configured for {chosen.english_name} ({chosen.code}). "
                f"Configured languages: {configured}. Add an entry to tts.voices, "
                "or fall back to on-screen captions."
            )
            raise ConfigurationError(msg)

        def build() -> SherpaVitsSynthesizer:
            descriptor = self.models.get(model_name)
            if descriptor.is_non_commercial:
                logger.warning(
                    "Voice %s is licensed %s - NON-COMMERCIAL use only. See "
                    "docs/deployment.md before distributing anything built on it.",
                    descriptor.id,
                    descriptor.license,
                )

            model_dir = self.model_repository.ensure(descriptor)
            if descriptor.files:
                model_path = model_dir / Path(descriptor.files[0]).name
                tokens_path = model_dir / Path(descriptor.files[1]).name
            else:
                # Snapshot layout (Piper): one .onnx at the root plus tokens.txt.
                onnx_files = sorted(model_dir.glob("*.onnx"))
                if not onnx_files:
                    msg = f"No .onnx file found in the snapshot for {descriptor.id} at {model_dir}"
                    raise ConfigurationError(msg)
                model_path = onnx_files[0]
                tokens_path = model_dir / "tokens.txt"

            espeak_dir = model_dir / "espeak-ng-data"
            return SherpaVitsSynthesizer(
                model_path=model_path,
                tokens_path=tokens_path,
                model_id=descriptor.id,
                language=chosen,
                data_dir=espeak_dir if espeak_dir.is_dir() else None,
                cpu_threads=self.settings.stt.cpu_threads,
                speed=tts.speed,
                sentence_split=tts.sentence_split,
            )

        return self._cached(("tts", model_name, chosen.code), build)

    def _resolve_stt_descriptor(self, name: str) -> ModelDescriptor:
        """Look up a speech model, accepting bare Whisper size names.

        ``stt.model: base`` predates the multi-runtime registry and remains
        supported: a name with no direct entry is retried as ``whisper-<name>``.

        Args:
            name: Registry identifier or bare Whisper size.

        Returns:
            The descriptor.

        Raises:
            ConfigurationError: If neither form is declared.
        """
        try:
            return self.models.get(name)
        except ConfigurationError:
            return self.models.get(f"whisper-{name}")

    def describe_model_for(self, language: LanguageCode | None) -> str:
        """Name the checkpoint that would serve a language, for display.

        Args:
            language: Language to describe, or ``None`` for the configured
                source language.

        Returns:
            The registry identifier, e.g. ``"whisper-tamil-small"``.
        """
        chosen = language or (
            LanguageCode(self.settings.stt.language)
            if self.settings.stt.language
            else self.source_language()
        )
        return self._resolve_stt_descriptor(self._model_name_for(chosen)).id

    def _model_name_for(self, language: LanguageCode | None) -> str:
        """Resolve which Whisper checkpoint serves a language.

        Args:
            language: Language to serve, or ``None`` for the default model.

        Returns:
            The model name, from ``stt.language_models`` when the language has
            an entry, otherwise ``stt.model``.
        """
        stt = self.settings.stt
        if language is None:
            return stt.model
        return stt.language_models.get(language.code, stt.model)

    def create_recognizer_router(self, languages: Sequence[LanguageCode] | None = None):  # type: ignore[no-untyped-def]
        """Build a router that picks a recogniser per language.

        No single checkpoint serves both directions of a Tamil/English
        interpreter, so the pipeline holds this rather than one recogniser.

        Args:
            languages: Languages to serve, or ``None`` for both sides of the
                configured language pair.

        Returns:
            A :class:`RecognizerRouter`. Models load lazily, so a session that
            only ever speaks one language pays for one model.
        """
        from ai_interpreter.application.services.recognizer_router import RecognizerRouter

        pair = self.settings.app.language_pair
        chosen = (
            tuple(languages)
            if languages is not None
            else (LanguageCode(pair.source), LanguageCode(pair.target))
        )
        return RecognizerRouter(factory=self.create_recognizer, languages=chosen)

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
        for component in reversed(list(self.component_cache.values())):
            close = getattr(component, "close", None)
            if callable(close):
                close()
        self.component_cache.clear()
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
