"""Command line entry point.

Until the PySide6 interface arrives in Phase 8, this is how the application is
driven. Three commands exist and all three are useful for the whole life of the
project:

``--check``
    An environment doctor. Reports hardware, selected profile, configuration
    provenance, dependency availability and audio endpoints, then exits
    non-zero if anything required is missing. This is the first thing to run
    when something behaves unexpectedly, and the output belongs in every bug
    report.
``--print-config``
    Dumps the fully merged, validated configuration. Answers "what settings is
    it actually using?" definitively, rather than by reading four YAML files
    and guessing how they combined.
``--version``
    Prints the version.

Console output is forced to UTF-8 before anything else runs: the Windows
console defaults to a legacy code page, and printing Tamil or Hindi text to it
raises ``UnicodeEncodeError``.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from ai_interpreter import __version__
from ai_interpreter.app.container import Container
from ai_interpreter.cli_audio import run_list_devices, run_record
from ai_interpreter.cli_interpret import run_interpret
from ai_interpreter.cli_stt import run_benchmark, run_listen, run_transcribe
from ai_interpreter.cli_translate import run_translate
from ai_interpreter.cli_tts import run_speak
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.infrastructure.config.settings import Profile
from ai_interpreter.infrastructure.system.audio_endpoints import list_windows_audio_endpoints

__all__ = ["main"]

_EXIT_OK: Final[int] = 0
_EXIT_FAILED_CHECK: Final[int] = 1
_EXIT_CONFIG_ERROR: Final[int] = 2

_WIDTH: Final[int] = 78

# Packages required right now, with the phase that introduced each. Later
# phases append to this list as they add dependencies, so ``--check`` always
# reflects what the project currently needs.
_REQUIRED_PACKAGES: Final[tuple[tuple[str, str], ...]] = (
    ("numpy", "Phase 2 - audio buffer representation"),
    ("pydantic", "Phase 2 - configuration validation"),
    ("pydantic_settings", "Phase 2 - secret loading"),
    ("yaml", "Phase 2 - configuration files"),
    ("platformdirs", "Phase 2 - user directory resolution"),
    ("psutil", "Phase 2 - hardware detection"),
    ("sounddevice", "Phase 3 - microphone capture"),
    ("soundfile", "Phase 3 - WAV recording"),
    ("soxr", "Phase 3 - sample rate conversion"),
    ("onnxruntime", "Phase 3 - voice activity detection"),
    ("huggingface_hub", "Phase 3 - model download"),
    ("faster_whisper", "Phase 4 - speech to text"),
    ("ctranslate2", "Phase 4 - speech to text runtime"),
    ("sherpa_onnx", "Phase 4b - streaming speech to text"),
    ("onnx", "Phase 4b - model metadata patching"),
    ("sentencepiece", "Phase 5 - translation tokenisation"),
    ("PySide6", "Phase 8 - desktop interface"),
)

# External programs. Missing entries are warnings, not failures: none of them
# is needed until a later phase.
_EXTERNAL_TOOLS: Final[tuple[tuple[str, str], ...]] = (
    ("git", "version control (recommended)"),
    ("ffmpeg", "audio conversion - needed from Phase 3"),
)


@dataclass(slots=True)
class _CheckResult:
    """Accumulates the outcome of the environment check.

    Args:
        failures: Problems that prevent the application from working.
        warnings: Problems that will matter in a later phase.
    """

    failures: list[str]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        """Whether no blocking problem was found."""
        return not self.failures


def _force_utf8_console() -> None:
    """Reconfigure standard streams to UTF-8.

    Without this, printing a Tamil or Hindi string on a default Windows
    console raises ``UnicodeEncodeError``. ``errors="replace"`` guarantees
    output is never lost even on a console font that cannot draw the glyphs.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _heading(title: str) -> None:
    """Print a section heading.

    Args:
        title: Heading text.
    """
    print(f"\n{title}")
    print("-" * _WIDTH)


def _row(label: str, value: str) -> None:
    """Print an aligned label/value row.

    Args:
        label: Left-hand label.
        value: Right-hand value.
    """
    print(f"  {label:<26} {value}")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="ai-interpreter",
        description="Real-time speech-to-speech interpreter for Windows meeting applications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m ai_interpreter --check\n"
            "  python -m ai_interpreter --list-devices\n"
            "  python -m ai_interpreter --record 10\n"
            '  python -m ai_interpreter --record 10 --device "Internal Microphone"\n'
            "  python -m ai_interpreter --print-config\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"AI Interpreter {__version__}",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="launch the desktop interface",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the environment doctor and exit non-zero if anything is missing",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the fully merged, validated configuration as YAML",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list every audio input and output device",
    )
    parser.add_argument(
        "--record",
        type=float,
        metavar="SECONDS",
        default=None,
        help="record from the microphone, detect speech, and save test WAV files",
    )
    parser.add_argument(
        "--transcribe",
        type=Path,
        metavar="WAV",
        default=None,
        help="transcribe a WAV file, reporting timings and confidence",
    )
    parser.add_argument(
        "--listen",
        type=float,
        metavar="SECONDS",
        default=None,
        help="capture from the microphone and transcribe each utterance live",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="measure decode time across thread counts",
    )
    parser.add_argument(
        "--translate",
        type=str,
        metavar="TEXT",
        default=None,
        help="translate text with the configured engine and report timings",
    )
    parser.add_argument(
        "--speak",
        type=str,
        metavar="TEXT",
        default=None,
        help="synthesise text with the configured voice, save and play it",
    )
    parser.add_argument(
        "--interpret",
        type=float,
        metavar="SECONDS",
        default=None,
        help="run live interpretation: mic in, translated speech out",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        metavar="FILE",
        default=None,
        help="for --interpret: replay a recording instead of the microphone",
    )
    parser.add_argument(
        "--out",
        type=str,
        metavar="NAME",
        default=None,
        help='output device for --speak, e.g. "CABLE Input" for the virtual microphone',
    )
    parser.add_argument(
        "--source",
        type=str,
        metavar="CODE",
        default=None,
        help="source language for --translate, e.g. ta (default: configured pair)",
    )
    parser.add_argument(
        "--target",
        type=str,
        metavar="CODE",
        default=None,
        help="target language for --translate, e.g. en (default: configured pair)",
    )
    parser.add_argument(
        "--device",
        type=str,
        metavar="NAME",
        default=None,
        help="input device name fragment, overriding audio.input.device",
    )
    parser.add_argument(
        "--language",
        type=str,
        metavar="CODE",
        default=None,
        help="language to decode, e.g. ta or en, overriding stt.language",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        metavar="N",
        default=3,
        help="timed runs per benchmark configuration (default: 3)",
    )
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in Profile],
        default=None,
        help="override the hardware profile instead of auto-detecting it",
    )
    return parser


def _check_packages(result: _CheckResult) -> None:
    """Verify that every required Python package can be imported.

    ``importlib.util.find_spec`` is used rather than a real import: it answers
    the question without paying the cost of loading numpy and pydantic.

    Args:
        result: Accumulator to record failures into.
    """
    _heading("Python packages")
    for module_name, purpose in _REQUIRED_PACKAGES:
        found = importlib.util.find_spec(module_name) is not None
        status = "OK" if found else "MISSING"
        _row(f"[{status}] {module_name}", purpose)
        if not found:
            result.failures.append(
                f"Python package {module_name!r} is not installed "
                f"(run: pip install -r requirements-dev.txt)"
            )


def _check_tools(result: _CheckResult) -> None:
    """Verify external command line programs.

    Args:
        result: Accumulator to record warnings into.
    """
    _heading("External tools")
    for tool, purpose in _EXTERNAL_TOOLS:
        path = shutil.which(tool)
        status = "OK" if path else "MISSING"
        _row(f"[{status}] {tool}", path or purpose)
        if path is None:
            result.warnings.append(f"{tool} is not on PATH - {purpose}")


def _check_audio_endpoints(result: _CheckResult) -> None:
    """Report audio endpoints and whether a virtual cable is installed.

    Args:
        result: Accumulator to record warnings into.
    """
    _heading("Audio endpoints (Windows registry)")
    endpoints = list_windows_audio_endpoints()

    if not endpoints:
        _row("[UNKNOWN]", "could not read audio endpoints from the registry")
        result.warnings.append(
            "Audio endpoints could not be enumerated; Phase 3 will report devices directly."
        )
        return

    for endpoint in endpoints:
        # Direction is always shown. Replacing it with a "CABLE" tag would hide
        # exactly the information needed to route audio: the playback endpoint
        # (CABLE Input) and the capture endpoint (CABLE Output) are different
        # halves of one cable and are not interchangeable.
        suffix = "   <- virtual cable" if endpoint.is_virtual_cable else ""
        _row(f"[{endpoint.direction:<7}]", f"{endpoint.name}{suffix}")

    cables = [endpoint for endpoint in endpoints if endpoint.is_virtual_cable]
    if not cables:
        result.warnings.append(
            "No virtual audio cable detected. Install VB-CABLE and reboot before Phase 7 "
            "(see docs/setup.md). It is not needed until then."
        )
        return

    has_render = any(cable.direction == "render" for cable in cables)
    has_capture = any(cable.direction == "capture" for cable in cables)
    if has_render and has_capture:
        print()
        _row("Virtual cable ready", "playback -> CABLE Input, meeting app mic -> CABLE Output")
    else:
        missing = "capture (CABLE Output)" if has_render else "playback (CABLE Input)"
        result.warnings.append(
            f"A virtual cable was found but its {missing} half is missing or disabled. "
            "Check Windows Sound settings, or reinstall VB-CABLE and reboot."
        )


def _report_hardware(container: Container) -> None:
    """Print the detected hardware snapshot.

    Args:
        container: Built application container.
    """
    hardware = container.hardware
    _heading("Hardware")
    _row("Operating system", f"{hardware.os_name} {hardware.os_version}")
    _row("Python", hardware.python_version)
    _row("CPU", hardware.cpu_name)
    _row("Cores", f"{hardware.physical_cores} physical / {hardware.logical_cores} logical")
    _row("RAM", f"{hardware.available_ram_gb:.1f} GB free of {hardware.total_ram_gb:.1f} GB")
    _row("Free disk", f"{hardware.free_disk_gb:.1f} GB")
    if hardware.gpus:
        for gpu in hardware.gpus:
            _row("GPU", f"{gpu.name} ({gpu.total_memory_gb:.1f} GB, driver {gpu.driver_version})")
    else:
        _row("GPU", "no NVIDIA GPU detected - CPU execution")


def _report_configuration(container: Container) -> None:
    """Print the selected profile and configuration provenance.

    Args:
        container: Built application container.
    """
    _heading("Profile and configuration")
    _row("Selected profile", container.selection.profile.value)
    _row("Reason", container.selection.reason)
    _row("Speech-to-text", f"{container.settings.stt.provider} / {container.settings.stt.model}")
    _row("Translation", container.settings.translation.provider)
    _row("Text-to-speech", container.settings.tts.provider)
    _row("Inference lane", container.settings.pipeline.inference_lane.value)

    print()
    for source in container.config_report.sources:
        _row("Config source", str(source))
    if container.config_report.env_overrides:
        for key in container.config_report.env_overrides:
            _row("Env override", key)


def _report_paths(container: Container) -> None:
    """Print resolved filesystem locations.

    Args:
        container: Built application container.
    """
    paths = container.paths
    _heading("Paths")
    _row("Project root", str(paths.root))
    _row("Configuration", str(paths.config_dir))
    _row("Logs", str(container.logging_service.log_file))
    _row("Models", str(paths.models_dir))
    _row("Recordings", str(paths.recordings_dir))
    _row("User overrides", str(paths.user_config_file))


def _report_privacy(container: Container) -> None:
    """Print the privacy-relevant settings currently in force.

    Args:
        container: Built application container.
    """
    privacy = container.settings.privacy
    _heading("Privacy")
    _row("Transcripts in logs", "yes" if privacy.log_transcripts else "no (default)")
    _row("Session history saved", "yes" if privacy.persist_history else "no (default)")
    _row("Translation cache", "enabled" if privacy.cache_translations else "disabled")
    _row("Telemetry", "none - the application sends no data anywhere")
    _row("Hugging Face token", "configured" if container.secrets.has_hf_token else "not set")
    _row("NVIDIA NIM key", "configured" if container.secrets.has_nim_key else "not set")


def _run_check(container: Container) -> int:
    """Run the full environment check.

    Args:
        container: Built application container.

    Returns:
        Process exit code.
    """
    result = _CheckResult(failures=[], warnings=[])

    print("=" * _WIDTH)
    print(f"  AI Interpreter {__version__} - environment check")
    print("=" * _WIDTH)

    _report_hardware(container)
    _report_configuration(container)
    _report_paths(container)
    _check_packages(result)
    _check_tools(result)
    _check_audio_endpoints(result)
    _report_privacy(container)

    _heading("Result")
    if result.warnings:
        for warning in result.warnings:
            print(f"  WARNING  {warning}")
    if result.failures:
        for failure in result.failures:
            print(f"  FAILED   {failure}")
        print(f"\n  {len(result.failures)} blocking problem(s) found.")
        return _EXIT_FAILED_CHECK

    print("  All required checks passed.")
    if result.warnings:
        print(f"  {len(result.warnings)} warning(s) above relate to later phases.")
    return _EXIT_OK


def _run_print_config(container: Container) -> int:
    """Print the effective configuration as YAML.

    Args:
        container: Built application container.

    Returns:
        Process exit code.
    """
    data = container.settings.model_dump(mode="json")
    print(f"# Effective configuration - profile: {container.selection.profile.value}")
    for source in container.config_report.sources:
        print(f"# source: {source}")
    for key in container.config_report.env_overrides:
        print(f"# env override: {key}")
    print()
    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False))
    return _EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Application entry point.

    Args:
        argv: Command line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success, ``1`` when a check failed, ``2``
        when configuration could not be loaded.
    """
    _force_utf8_console()
    parser = _build_parser()
    args = parser.parse_args(argv)

    profile_override = Profile(args.profile) if args.profile else None

    try:
        container = Container.build(profile_override=profile_override)
    except InterpreterError as exc:
        print(f"\nStartup failed:\n{exc}\n", file=sys.stderr)
        return _EXIT_CONFIG_ERROR

    try:
        if args.ui:
            # Imported lazily: a broken Qt installation must not take the
            # console commands down with it.
            from ai_interpreter.presentation.ui.app import run_ui

            return run_ui(container)
        if args.check:
            return _run_check(container)
        if args.print_config:
            return _run_print_config(container)
        if args.list_devices:
            return run_list_devices(container)
        if args.record is not None:
            if args.record <= 0:
                print("--record needs a positive number of seconds.", file=sys.stderr)
                return _EXIT_FAILED_CHECK
            return run_record(container, args.record, args.device)
        if args.transcribe is not None:
            return run_transcribe(container, args.transcribe, args.language)
        if args.listen is not None:
            if args.listen <= 0:
                print("--listen needs a positive number of seconds.", file=sys.stderr)
                return _EXIT_FAILED_CHECK
            return run_listen(container, args.listen, args.device)
        if args.benchmark:
            return run_benchmark(container, args.transcribe, args.repeats, args.language)
        if args.translate is not None:
            return run_translate(container, args.translate, args.source, args.target)
        if args.speak is not None:
            return run_speak(container, args.speak, args.language or args.target, args.out)
        if args.interpret is not None or args.wav is not None:
            if args.interpret is not None and args.interpret <= 0 and args.wav is None:
                print("--interpret needs a positive number of seconds.", file=sys.stderr)
                return _EXIT_FAILED_CHECK
            return run_interpret(
                container,
                args.interpret or 0.0,
                args.device,
                args.out,
                args.wav,
                args.source,
                args.target,
            )

        print(f"AI Interpreter {__version__}")
        print(f"Profile: {container.selection.profile.value} ({container.selection.reason})")
        print()
        print("Available commands:")
        print("  python -m ai_interpreter --ui              DESKTOP INTERFACE")
        print("  python -m ai_interpreter --check           verify the environment")
        print("  python -m ai_interpreter --list-devices    list audio devices")
        print("  python -m ai_interpreter --record 10       test the microphone")
        print("  python -m ai_interpreter --listen 20       live speech to text")
        print("  python -m ai_interpreter --transcribe f.wav  transcribe a file")
        print('  python -m ai_interpreter --translate "..."  translate text')
        print('  python -m ai_interpreter --speak "..."      text to speech')
        print("  python -m ai_interpreter --interpret 60     LIVE INTERPRETATION")
        print("  python -m ai_interpreter --benchmark       measure decode speed")
        print("  python -m ai_interpreter --print-config    show effective settings")
        print("  python -m ai_interpreter --help            full option list")
        return _EXIT_OK
    finally:
        container.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
