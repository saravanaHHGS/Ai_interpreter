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


def run_speak(container: Container, text: str, language: str | None) -> int:
    """Synthesise text, report timings, save and play the audio.

    Args:
        container: Built application container.
        text: Text to speak.
        language: Language code, or ``None`` for the configured target.

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
    try:
        synthesizer = container.create_synthesizer(chosen)
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR

    try:
        started = time.perf_counter()
        synthesizer.warmup()
        row("Voice", synthesizer.provider_id)
        row("Warmup", f"{(time.perf_counter() - started) * 1000.0:.0f} ms")

        heading("Synthesis")
        row("Text", text if len(text) <= 60 else text[:57] + "...")

        chunks = []
        first_chunk_ms: float | None = None
        started = time.perf_counter()
        for chunk in synthesizer.synthesize_stream(text, chosen):
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - started) * 1000.0
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

        _play(audio, rate.hz)
    except InterpreterError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return _EXIT_ERROR
    finally:
        synthesizer.close()

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
