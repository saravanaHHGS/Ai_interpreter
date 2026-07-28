"""Speech synthesis command.

``--speak`` synthesises text with the configured voice, reports the
first-chunk latency - the number that matters for perceived responsiveness -
saves a WAV for later listening, and plays the audio so quality can be judged
by ear immediately.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Final

import numpy as np

from ai_interpreter.app.container import Container
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.domain.value_objects import LanguageCode
from ai_interpreter.infrastructure.audio.recording import WavRecorder
from ai_interpreter.presentation.console import WIDTH, heading, row

__all__ = ["run_speak"]

logger = logging.getLogger(__name__)

_EXIT_OK: Final[int] = 0
_EXIT_ERROR: Final[int] = 1


def run_speak(
    container: Container,
    text: str,
    language: str | None,
    out_device: str | None = None,
) -> int:
    """Synthesise text, report timings, save and play the audio.

    Args:
        container: Built application container.
        text: Text to speak.
        language: Language code, or ``None`` for the configured target.
        out_device: Output device name fragment. ``"CABLE Input"`` routes the
            speech into the virtual microphone - written **chunk by chunk as
            synthesis produces them**, exactly as the live pipeline will, so
            this is the Phase 7 end-to-end verification path.

    Returns:
        Process exit code.
    """
    print("=" * WIDTH)
    print("  Speak")
    print("=" * WIDTH)

    try:
        chosen = (
            LanguageCode(language)
            if language
            else LanguageCode(container.settings.app.language_pair.target)
        )
    except ValueError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    heading("Loading")
    row("Language", f"{chosen.english_name} ({chosen.code})")
    sink = None
    try:
        synthesizer = container.create_synthesizer(chosen)
        if out_device:
            sink = container.create_audio_sink(device_name=out_device)
            row("Output device", f"{sink.device.name}  [{sink.device.host_api}]")
            if sink.device.is_virtual_cable:
                row("", "-> select 'CABLE Output' as the microphone in Teams/Zoom/Meet")
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    try:
        started = time.perf_counter()
        synthesizer.warmup()
        row("Voice", synthesizer.provider_id)
        row("Warmup", f"{(time.perf_counter() - started) * 1000.0:.0f} ms")
        if sink is not None:
            sink.open()

        heading("Synthesis")
        row("Text", text if len(text) <= 60 else text[:57] + "...")

        chunks = []
        first_chunk_ms: float | None = None
        started = time.perf_counter()
        for chunk in synthesizer.synthesize_stream(text, chosen):
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - started) * 1000.0
            if sink is not None:
                sink.write(chunk)
            chunks.append(chunk)
        total_ms = (time.perf_counter() - started) * 1000.0

        if not chunks or all(not chunk.pcm.size for chunk in chunks):
            print("  Nothing was synthesised.")
            return _EXIT_ERROR

        rate = chunks[0].sample_rate
        audio = np.concatenate([chunk.pcm for chunk in chunks if chunk.pcm.size])
        duration_s = rate.ms_for_samples(audio.size) / 1000.0

        row("First chunk ready", f"{first_chunk_ms:.0f} ms  <- perceived latency")
        row("All chunks done", f"{total_ms:.0f} ms for {len(chunks)} chunk(s)")
        row("Audio produced", f"{duration_s:.2f} s at {rate}")
        row(
            "Real-time factor",
            f"{total_ms / 1000.0 / duration_s:.2f}"
            + ("  (above 1.0: slower than playback)" if total_ms / 1000.0 > duration_s else ""),
        )

        stamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = container.paths.recordings_dir / f"speak-{chosen.code}-{stamp}.wav"
        with WavRecorder(wav_path, rate) as recorder:
            recorder.write(audio)
        row("Saved", wav_path.name)

        if sink is not None:
            print("\n  Draining into the output device...")
            sink.flush(timeout=duration_s + 5.0)
            row("Underruns", str(sink.underruns))
        else:
            _play(audio, rate.hz)
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR
    finally:
        synthesizer.close()
        if sink is not None:
            sink.close()

    return _EXIT_OK


def _play(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio on the default output device, best effort.

    Args:
        audio: Mono float32 samples.
        sample_rate: Their sample rate.
    """
    try:
        import sounddevice

        print("\n  Playing...")
        sounddevice.play(audio, samplerate=sample_rate, blocking=True)
    except Exception as exc:
        logger.warning("Playback failed (%s); the saved WAV can be played manually", exc)
