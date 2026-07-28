"""WAV file capture source.

Replays a recording through the same interface as a real microphone. This is
what makes the whole pipeline testable: every stage from voice activity
detection to the virtual microphone can be exercised on a build machine with
no audio hardware, deterministically, with a known-correct expected result.

It is also the honest way to compare configurations. Tuning the endpoint delay
against a live microphone means comparing two different utterances; replaying
one recording compares the settings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ai_interpreter.domain.entities import AudioFrame, DeviceInfo
from ai_interpreter.domain.errors import AudioCaptureError
from ai_interpreter.domain.value_objects import DeviceKind, SampleRate

__all__ = ["WavFileSource"]

logger = logging.getLogger(__name__)


class WavFileSource:
    """Replays a WAV file, satisfying the ``AudioSource`` port.

    Args:
        path: WAV file to replay.
        frame_ms: Block size in milliseconds, matching a microphone's.
        loop: Restart at the end instead of stopping.
        realtime: Pace frames at wall-clock speed, as a microphone would.
            Without pacing a 15-second file is consumed in well under a
            second, which makes every downstream timing measurement a lie:
            queue waits masquerade as decode time, and voice-activity state
            flaps at replay speed. Tests keep the fast default; anything
            *simulating* live input must pace.

    Raises:
        AudioCaptureError: If the file cannot be read, or its sample rate is
            not one the application supports.
    """

    def __init__(
        self,
        path: Path,
        frame_ms: int = 20,
        loop: bool = False,
        realtime: bool = False,
    ) -> None:
        self._path = path
        self._frame_ms = frame_ms
        self._loop = loop
        self._realtime = realtime
        self._next_deadline: float | None = None

        samples, rate = self._load(path)
        self._samples = samples
        self._sample_rate = rate
        self._block_samples = rate.samples_for_ms(frame_ms)
        self._position = 0
        self._frames_read = 0
        self._running = False

        self._device = DeviceInfo(
            index=-1,
            name=f"WAV file: {path.name}",
            kind=DeviceKind.INPUT,
            max_channels=1,
            default_sample_rate=float(rate.hz),
            host_api="file",
        )

    @staticmethod
    def _load(path: Path) -> tuple[NDArray[np.float32], SampleRate]:
        """Read a WAV file as mono float32.

        Args:
            path: File to read.

        Returns:
            The samples and their sample rate.

        Raises:
            AudioCaptureError: If reading fails or the rate is unsupported.
        """
        try:
            import soundfile as sf

            data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            msg = f"Could not read audio file {path}: {exc}"
            raise AudioCaptureError(msg) from exc

        # mean() promotes to float64, so convert back explicitly.
        mono: NDArray[np.float32] = (
            np.asarray(data, dtype=np.float32).mean(axis=1).astype(np.float32)
        )

        try:
            sample_rate = SampleRate(int(rate))
        except ValueError as exc:
            msg = (
                f"Audio file {path} has an unsupported sample rate of {rate} Hz. "
                "Convert it first, e.g.: ffmpeg -i input.wav -ar 16000 -ac 1 output.wav"
            )
            raise AudioCaptureError(msg) from exc

        return mono, sample_rate

    # -- port interface ----------------------------------------------------
    @property
    def device(self) -> DeviceInfo:
        """Pseudo-device describing the file."""
        return self._device

    @property
    def sample_rate(self) -> SampleRate:
        """Rate of the file's audio."""
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        """Whether replay is active."""
        return self._running

    @property
    def duration_ms(self) -> float:
        """Total length of the file in milliseconds."""
        return self._sample_rate.ms_for_samples(self._samples.size)

    @property
    def is_exhausted(self) -> bool:
        """Whether replay has reached the end of a non-looping file."""
        return not self._loop and self._position >= self._samples.size

    def start(self) -> None:
        """Begin replay from the start of the file. Idempotent."""
        self._running = True
        self._next_deadline = None

    def _pace(self) -> None:
        """Sleep so frames are delivered at wall-clock speed."""
        import time

        now = time.monotonic()
        if self._next_deadline is None:
            self._next_deadline = now
        else:
            delay = self._next_deadline - now
            if delay > 0:
                time.sleep(delay)
        self._next_deadline += self._frame_ms / 1000.0

    def stop(self) -> None:
        """Stop replay. Idempotent."""
        self._running = False

    def close(self) -> None:
        """Release resources. Alias of :meth:`stop`."""
        self.stop()

    def read(self, timeout: float | None = None) -> AudioFrame | None:
        """Return the next block of the file.

        Args:
            timeout: Ignored - a file is always ready. Present so this is
                interchangeable with a microphone.

        Returns:
            The next frame, or ``None`` at end of file.

        Raises:
            AudioCaptureError: If replay has not been started.
        """
        if not self._running:
            msg = "read() called while replay is not running; call start() first"
            raise AudioCaptureError(msg)

        if self._realtime:
            self._pace()

        if self._position >= self._samples.size:
            if not self._loop:
                return None
            self._position = 0

        end = min(self._position + self._block_samples, self._samples.size)
        block = self._samples[self._position : end]
        self._position = end

        if block.size < self._block_samples:
            padded = np.zeros(self._block_samples, dtype=np.float32)
            padded[: block.size] = block
            block = padded

        timestamp_ms = self._frames_read * self._frame_ms
        self._frames_read += 1
        return AudioFrame(
            pcm=block.copy(),
            sample_rate=self._sample_rate,
            timestamp_ms=timestamp_ms,
        )
