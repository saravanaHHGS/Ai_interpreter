"""WAV recording of captured audio.

Two uses, both practical:

* **Diagnosis.** When recognition is poor the first question is always
  "what did the microphone actually hear?". A recording answers it in seconds;
  guessing does not.
* **Test fixtures.** A saved utterance becomes a deterministic input for the
  Phase 4 accuracy tests and Phase 10 latency benchmarks.

Recordings are written incrementally rather than accumulated in memory, so a
long session cannot exhaust RAM, and a crash still leaves a usable file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from ai_interpreter.domain.errors import AudioCaptureError
from ai_interpreter.domain.value_objects import SampleRate

__all__ = ["WavRecorder"]

logger = logging.getLogger(__name__)


class WavRecorder:
    """Writes mono float32 audio to a 16-bit WAV file as it arrives.

    16-bit PCM is chosen over 32-bit float because every audio tool on Windows
    opens it without complaint, and speech captured from a microphone has
    nothing near the dynamic range that would justify the larger format.

    Args:
        path: Destination file. Parent directories are created.
        sample_rate: Rate of the audio being written.
    """

    def __init__(self, path: Path, sample_rate: SampleRate) -> None:
        self._path = path
        self._sample_rate = sample_rate
        self._handle: Any = None
        self._samples_written = 0

    @property
    def path(self) -> Path:
        """Destination file."""
        return self._path

    @property
    def duration_ms(self) -> float:
        """Length of audio written so far."""
        return self._sample_rate.ms_for_samples(self._samples_written)

    @property
    def is_open(self) -> bool:
        """Whether the file is open for writing."""
        return self._handle is not None

    def open(self) -> None:
        """Create the file and prepare for writing. Idempotent.

        Raises:
            AudioCaptureError: If the file cannot be created.
        """
        if self._handle is not None:
            return

        try:
            import soundfile as sf

            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = sf.SoundFile(
                str(self._path),
                mode="w",
                samplerate=self._sample_rate.hz,
                channels=1,
                subtype="PCM_16",
            )
        except Exception as exc:
            self._handle = None
            msg = f"Could not create recording {self._path}: {exc}"
            raise AudioCaptureError(msg) from exc

    def write(self, samples: NDArray[np.float32]) -> None:
        """Append samples to the file.

        Args:
            samples: Mono float32 samples in ``[-1.0, 1.0]``.

        Raises:
            AudioCaptureError: If the recorder is not open or writing fails.
        """
        if self._handle is None:
            msg = "write() called on a closed recorder; call open() first"
            raise AudioCaptureError(msg)
        if not samples.size:
            return

        try:
            self._handle.write(samples)
        except Exception as exc:
            msg = f"Could not write to recording {self._path}: {exc}"
            raise AudioCaptureError(msg) from exc

        self._samples_written += int(samples.size)

    def close(self) -> None:
        """Finalise the file. Idempotent."""
        handle, self._handle = self._handle, None
        if handle is None:
            return

        try:
            handle.close()
        except Exception as exc:
            logger.warning("Error while closing recording %s: %s", self._path, exc)
            return

        logger.info(
            "Recording saved: %s (%.1f s, %s)",
            self._path,
            self.duration_ms / 1000.0,
            self._sample_rate,
        )

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
