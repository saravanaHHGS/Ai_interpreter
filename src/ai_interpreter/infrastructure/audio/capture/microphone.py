"""Microphone capture through PortAudio.

The callback runs on a high-priority thread owned by the audio driver and is
expected back within a couple of milliseconds. It therefore does exactly two
things: downmix to mono and hand the block to :class:`AudioBlockBuffer`. No
logging, no filtering, no allocation beyond one array copy.

The copy is mandatory, not defensive: PortAudio reuses its input buffer as
soon as the callback returns, so keeping a reference would produce audio that
silently changes underneath the consumer.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ai_interpreter.domain.entities import AudioFrame, DeviceInfo
from ai_interpreter.domain.errors import AudioCaptureError, DeviceError
from ai_interpreter.domain.value_objects import SampleRate
from ai_interpreter.infrastructure.audio.buffers import AudioBlockBuffer

__all__ = ["MicrophoneSource"]

logger = logging.getLogger(__name__)

# Roughly two seconds of audio at 20 ms blocks. Large enough to absorb a
# scheduling hiccup on a busy 2-core machine, small enough that a genuinely
# stalled consumer is detected rather than hidden behind growing latency.
_DEFAULT_BUFFER_BLOCKS: Final[int] = 100


class MicrophoneSource:
    """Captures audio from an input device, satisfying the ``AudioSource`` port.

    Args:
        device: Endpoint to capture from.
        sample_rate: Requested capture rate.
        frame_ms: Driver block size in milliseconds.
        buffer_blocks: Blocks buffered before the oldest is dropped.
    """

    def __init__(
        self,
        device: DeviceInfo,
        sample_rate: SampleRate,
        frame_ms: int,
        buffer_blocks: int = _DEFAULT_BUFFER_BLOCKS,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._block_samples = sample_rate.samples_for_ms(frame_ms)
        self._buffer = AudioBlockBuffer(max_blocks=buffer_blocks)
        self._stream: Any = None
        self._frames_read = 0
        self._callback_errors = 0

    # -- port interface ----------------------------------------------------
    @property
    def device(self) -> DeviceInfo:
        """Endpoint this source reads from."""
        return self._device

    @property
    def sample_rate(self) -> SampleRate:
        """Rate frames are delivered at."""
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        """Whether capture is currently active."""
        return self._stream is not None

    @property
    def dropped_blocks(self) -> int:
        """Blocks lost because the consumer fell behind."""
        return self._buffer.dropped_blocks

    @property
    def callback_errors(self) -> int:
        """Overflow or underflow conditions reported by the driver."""
        return self._callback_errors

    @property
    def block_samples(self) -> int:
        """Samples in each block requested from the driver."""
        return self._block_samples

    def start(self) -> None:
        """Open the device and begin capturing. Idempotent.

        Raises:
            DeviceError: If the device cannot be opened at the requested
                format.
        """
        if self._stream is not None:
            return

        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                device=self._device.index,
                channels=1,
                samplerate=self._sample_rate.hz,
                blocksize=self._block_samples,
                dtype="float32",
                callback=self._on_audio,
                latency="low",
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            msg = (
                f"Could not open input device {self._device.name!r} "
                f"[{self._device.host_api}] at {self._sample_rate}: {exc}\n"
                "The device may be in use by another application, disabled in "
                "Windows Sound settings, or unable to provide this sample rate."
            )
            raise DeviceError(msg) from exc

        logger.info(
            "Capturing from %r [%s] at %s, %d ms blocks (%d samples)",
            self._device.name,
            self._device.host_api,
            self._sample_rate,
            self._frame_ms,
            self._block_samples,
        )

    def stop(self) -> None:
        """Stop capturing and release the device. Idempotent."""
        stream, self._stream = self._stream, None
        if stream is None:
            return

        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            logger.warning("Error while closing the input stream: %s", exc)

        logger.info(
            "Capture stopped: %d frames read, %d blocks dropped, %d driver warnings",
            self._frames_read,
            self._buffer.dropped_blocks,
            self._callback_errors,
        )

    def read(self, timeout: float | None = None) -> AudioFrame | None:
        """Take the next captured block.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            The next frame, or ``None`` if the timeout expired.

        Raises:
            AudioCaptureError: If capture is not running.
        """
        if self._stream is None:
            msg = "read() called while capture is not running; call start() first"
            raise AudioCaptureError(msg)

        block = self._buffer.read(timeout)
        if block is None:
            return None

        timestamp_ms = self._frames_read * self._frame_ms
        self._frames_read += 1
        return AudioFrame(pcm=block, sample_rate=self._sample_rate, timestamp_ms=timestamp_ms)

    def close(self) -> None:
        """Release the device. Alias of :meth:`stop` for lifecycle symmetry."""
        self.stop()

    # -- real-time callback ------------------------------------------------
    def _on_audio(
        self,
        indata: NDArray[np.float32],
        _frames: int,
        _time: Any,
        status: Any,
    ) -> None:
        """Receive a block from the driver.

        Runs on the audio thread. Everything here is bounded and allocation
        light; anything slower belongs on the consumer side of the buffer.

        Args:
            indata: Driver buffer, reused after this call returns.
            _frames: Sample count, implied by the array shape.
            _time: Driver timestamps, unused.
            status: Driver status flags.
        """
        if status:
            self._callback_errors += 1

        # copy() because the driver reuses indata; ravel() flattens the
        # (frames, 1) mono shape without another allocation.
        self._buffer.write(np.asarray(indata[:, 0], dtype=np.float32).copy())
