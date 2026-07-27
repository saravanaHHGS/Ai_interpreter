"""Buffering between the real-time audio callback and the processing thread.

The audio driver calls back on a high-priority thread every few milliseconds
and expects to be released almost immediately. Anything slow there - a lock
held by another thread, a file write, a model call - causes the driver to skip
a buffer, which the user hears as a click.

:class:`AudioBlockBuffer` is the boundary. The callback does exactly one
thing: append a copied array to a ``collections.deque``. Under CPython that
append is a single bytecode operating on a C-level structure, so it is atomic
with respect to other threads without any explicit lock, and ``maxlen`` gives
drop-oldest overflow behaviour for free.

Dropping the *oldest* audio is deliberate. If the consumer stalls, keeping
stale audio and falling further behind is worse than losing it: in live
interpretation, audio from four seconds ago has no value.

Honesty about "lock-free": Python holds the GIL, so no Python code is
lock-free in the hard real-time sense. The goal here is bounded, microsecond
work in the callback rather than a theoretical guarantee.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np
from numpy.typing import NDArray

__all__ = ["AudioBlockBuffer", "FrameAssembler"]

logger = logging.getLogger(__name__)


class AudioBlockBuffer:
    """Hand-off queue of audio blocks between the driver callback and a worker.

    Args:
        max_blocks: Blocks retained before the oldest is discarded. Sets the
            maximum buffering delay: ``max_blocks * block duration``.
    """

    def __init__(self, max_blocks: int) -> None:
        if max_blocks < 1:
            msg = f"max_blocks must be at least 1, got {max_blocks}"
            raise ValueError(msg)
        self._blocks: deque[NDArray[np.float32]] = deque(maxlen=max_blocks)
        self._ready = threading.Event()
        self._dropped_blocks = 0
        self._written_blocks = 0

    @property
    def dropped_blocks(self) -> int:
        """Blocks discarded because the consumer could not keep up.

        Any non-zero value means audio was lost and the user may have heard a
        click. Surfaced on the Performance page rather than hidden.
        """
        return self._dropped_blocks

    @property
    def written_blocks(self) -> int:
        """Blocks accepted from the driver since creation."""
        return self._written_blocks

    @property
    def pending_blocks(self) -> int:
        """Blocks waiting to be consumed."""
        return len(self._blocks)

    def write(self, block: NDArray[np.float32]) -> None:
        """Accept a block from the audio callback.

        Called on the real-time thread. Must stay allocation-light and must
        never block.

        Args:
            block: Mono float32 samples. The caller must pass a copy - the
                driver reuses its own buffer after the callback returns.
        """
        if len(self._blocks) == self._blocks.maxlen:
            self._dropped_blocks += 1
        self._blocks.append(block)
        self._written_blocks += 1
        self._ready.set()

    def read(self, timeout: float | None = None) -> NDArray[np.float32] | None:
        """Take the oldest pending block, waiting if necessary.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            The next block, or ``None`` if the timeout expired.
        """
        while True:
            try:
                return self._blocks.popleft()
            except IndexError:
                pass

            self._ready.clear()
            # Re-check after clearing: a block may have arrived in between,
            # and without this the consumer could sleep on already-ready data.
            if self._blocks:
                continue
            if not self._ready.wait(timeout):
                return None

    def clear(self) -> None:
        """Discard everything pending, e.g. when capture restarts."""
        self._blocks.clear()
        self._ready.clear()


class FrameAssembler:
    """Re-chunks a variable-length sample stream into fixed-size frames.

    Neural voice activity detectors accept one exact frame size - Silero v5
    requires precisely 512 samples at 16 kHz. Driver blocks and resampler
    output do not respect that, so this sits between them and emits only
    correctly sized frames, carrying the remainder to the next call.

    Args:
        frame_samples: Samples per emitted frame.
    """

    def __init__(self, frame_samples: int) -> None:
        if frame_samples < 1:
            msg = f"frame_samples must be at least 1, got {frame_samples}"
            raise ValueError(msg)
        self._frame_samples = frame_samples
        self._pending: NDArray[np.float32] = np.empty(0, dtype=np.float32)

    @property
    def frame_samples(self) -> int:
        """Samples in each emitted frame."""
        return self._frame_samples

    @property
    def pending_samples(self) -> int:
        """Samples held back, waiting to complete the next frame."""
        return int(self._pending.size)

    def push(self, samples: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        """Add samples and return every complete frame now available.

        Args:
            samples: Mono float32 samples of any length.

        Returns:
            Complete frames in order, possibly empty.
        """
        if samples.size:
            self._pending = np.concatenate((self._pending, samples))

        frames: list[NDArray[np.float32]] = []
        while self._pending.size >= self._frame_samples:
            frames.append(self._pending[: self._frame_samples].copy())
            self._pending = self._pending[self._frame_samples :]
        return frames

    def flush(self) -> NDArray[np.float32] | None:
        """Return any partial frame, zero-padded, and reset.

        Used at end of capture so the final fraction of a frame is not lost.

        Returns:
            A full-length zero-padded frame, or ``None`` if nothing is pending.
        """
        if not self._pending.size:
            return None
        padded = np.zeros(self._frame_samples, dtype=np.float32)
        padded[: self._pending.size] = self._pending
        self._pending = np.empty(0, dtype=np.float32)
        return padded

    def reset(self) -> None:
        """Discard any partial frame."""
        self._pending = np.empty(0, dtype=np.float32)
