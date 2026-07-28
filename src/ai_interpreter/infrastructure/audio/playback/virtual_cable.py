"""Playback into a virtual audio cable (or any output device).

This is the adapter that makes the whole project an *interpreter*: synthesised
speech written here, with the device set to ``CABLE Input``, appears at
``CABLE Output`` - which Teams, Zoom and Meet select as their microphone.
Nothing about the class is cable-specific; pointed at real speakers it is an
ordinary monitor output, which is exactly how the ``monitor`` option will use
it later.

Design mirrors the capture side in reverse, for the same real-time reason:
the PortAudio callback must never block or allocate, so it only copies from a
pending-block queue, and everything else happens on the writer's side.

Three behaviours matter for a virtual microphone specifically:

**The stream stays open and emits silence between utterances.** Meeting
applications dislike microphones that appear and disappear; a continuously
running stream that plays zeros when idle looks like a normal, quiet mic.

**A jitter buffer gates the start of each utterance.** Synthesis chunks
arrive in bursts; starting playback on the first sample and immediately
starving would stutter the first word. Playback holds until
``jitter_buffer_ms`` of audio is queued - or the final chunk arrives,
whichever is first, so an utterance shorter than the buffer still plays.

**``clear()`` implements barge-in.** When the speaker interrupts, the
half-played previous translation must stop *now*; dropping the queue is the
mechanism.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ai_interpreter.domain.entities import DeviceInfo, SpeechAudio
from ai_interpreter.domain.errors import AudioOutputError
from ai_interpreter.domain.value_objects import SampleRate
from ai_interpreter.infrastructure.audio.dsp import StreamingResampler

__all__ = ["VirtualCableSink"]

logger = logging.getLogger(__name__)

# Driver block size. Matches the capture side's reasoning: small enough for
# low latency, large enough not to wake a 2-core CPU excessively.
_BLOCK_MS: Final[int] = 20

# Extra wait after the queue drains, covering audio still inside the driver.
_DRAIN_PAD_SECONDS: Final[float] = 0.1


class VirtualCableSink:
    """Queued playback to one output device, satisfying the ``AudioSink`` port.

    Args:
        device: Endpoint to play into - ``CABLE Input`` for the virtual
            microphone, or any speaker for monitoring.
        output_rate: Rate the device is opened at. Chunks arriving at other
            rates are resampled on write.
        jitter_buffer_ms: Audio queued before playback of an utterance
            starts.
    """

    def __init__(
        self,
        device: DeviceInfo,
        output_rate: SampleRate,
        jitter_buffer_ms: int = 60,
    ) -> None:
        self._device = device
        self._rate = output_rate
        self._prebuffer_samples = output_rate.samples_for_ms(jitter_buffer_ms)

        self._stream: Any = None
        self._lock = threading.Lock()
        self._pending: deque[NDArray[np.float32]] = deque()
        self._head_offset = 0
        self._pending_samples = 0
        self._playing = False
        self._drained = threading.Event()
        self._drained.set()

        self._resampler: StreamingResampler | None = None
        self._resampler_rate: int | None = None

        self._chunks_written = 0
        self._samples_played = 0
        self._underruns = 0

    # -- port interface ----------------------------------------------------
    @property
    def device(self) -> DeviceInfo:
        """Endpoint this sink writes to."""
        return self._device

    @property
    def is_open(self) -> bool:
        """Whether the sink is currently accepting audio."""
        return self._stream is not None

    @property
    def pending_ms(self) -> float:
        """Milliseconds of audio queued but not yet played."""
        with self._lock:
            return self._rate.ms_for_samples(self._pending_samples)

    @property
    def underruns(self) -> int:
        """Times playback stopped because the queue ran dry mid-stream."""
        return self._underruns

    @property
    def chunks_written(self) -> int:
        """Chunks accepted since the sink was created."""
        return self._chunks_written

    def open(self) -> None:
        """Open the device and start the (initially silent) stream. Idempotent.

        Raises:
            AudioOutputError: If the device cannot be opened.
        """
        if self._stream is not None:
            return

        try:
            import sounddevice

            self._stream = sounddevice.OutputStream(
                device=self._device.index,
                channels=1,
                samplerate=self._rate.hz,
                blocksize=self._rate.samples_for_ms(_BLOCK_MS),
                dtype="float32",
                callback=self._on_audio,
                latency="low",
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            msg = (
                f"Could not open output device {self._device.name!r} "
                f"[{self._device.host_api}] at {self._rate}: {exc}\n"
                "The device may be disabled in Windows Sound settings or in "
                "exclusive use by another application."
            )
            raise AudioOutputError(msg) from exc

        logger.info(
            "Audio sink open on %r [%s] at %s",
            self._device.name,
            self._device.host_api,
            self._rate,
        )

    def write(self, audio: SpeechAudio) -> None:
        """Queue a chunk for playback, resampling if its rate differs.

        Args:
            audio: Chunk to play. Empty chunks are ignored, except that a
                final empty chunk still releases the jitter gate so a pending
                short utterance plays out.

        Raises:
            AudioOutputError: If the sink is not open.
        """
        if self._stream is None:
            msg = "write() called on a closed sink; call open() first"
            raise AudioOutputError(msg)

        samples = audio.pcm
        if samples.size:
            if audio.sample_rate.hz != self._rate.hz:
                samples = self._resample(samples, audio.sample_rate.hz)
            if samples.size:
                with self._lock:
                    self._pending.append(samples)
                    self._pending_samples += samples.size
                    self._drained.clear()
                    if not self._playing and self._pending_samples >= self._prebuffer_samples:
                        self._playing = True
            self._chunks_written += 1

        if audio.is_last:
            # An utterance shorter than the jitter buffer must still play.
            with self._lock:
                if self._pending_samples:
                    self._playing = True

    def flush(self, timeout: float | None = None) -> None:
        """Block until queued audio has been handed to the driver.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely.
        """
        if self._stream is None:
            return
        with self._lock:
            if self._pending_samples:
                self._playing = True
        drained = self._drained.wait(timeout)
        if drained:
            # The queue is empty but the last block may still be inside the
            # driver's own buffer.
            time.sleep(_DRAIN_PAD_SECONDS)

    def clear(self) -> None:
        """Discard queued audio immediately - the barge-in path."""
        with self._lock:
            self._pending.clear()
            self._head_offset = 0
            self._pending_samples = 0
            self._playing = False
            self._drained.set()

    def close(self) -> None:
        """Stop the stream and release the device. Idempotent."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            logger.warning("Error while closing the output stream: %s", exc)

        logger.info(
            "Audio sink closed: %d chunks written, %.1f s played, %d underruns",
            self._chunks_written,
            self._samples_played / self._rate.hz,
            self._underruns,
        )

    # -- internals ---------------------------------------------------------
    def _resample(self, samples: NDArray[np.float32], input_rate: int) -> NDArray[np.float32]:
        """Convert a chunk to the device rate.

        The resampler is stateful across chunks of one utterance (matching
        the capture side's correctness argument) and is rebuilt when the
        incoming rate changes - which happens when the target language
        switches between voices with different native rates.

        Args:
            samples: Chunk samples.
            input_rate: Their rate.

        Returns:
            Samples at the device rate.
        """
        if self._resampler is None or self._resampler_rate != input_rate:
            logger.debug("Sink resampler: %d Hz -> %d Hz", input_rate, self._rate.hz)
            self._resampler = StreamingResampler(input_rate, self._rate.hz)
            self._resampler_rate = input_rate
        return self._resampler.process(samples)

    def _on_audio(
        self,
        outdata: NDArray[np.float32],
        _frames: int,
        _time: Any,
        status: Any,
    ) -> None:
        """Driver callback: copy queued audio out, or silence.

        Args:
            outdata: Driver buffer to fill, shape ``(frames, 1)``.
            _frames: Frame count, implied by the shape.
            _time: Driver timestamps, unused.
            status: Driver status flags.
        """
        if status:
            self._underruns += 1
        self._fill(outdata[:, 0])

    def _fill(self, out: NDArray[np.float32]) -> None:
        """Fill one output block from the queue.

        Separated from the driver callback so the drain logic is testable
        without a device.

        Args:
            out: Mono block to fill in place.
        """
        with self._lock:
            if not self._playing:
                out[:] = 0.0
                return

            filled = 0
            while filled < out.size and self._pending:
                head = self._pending[0]
                available = head.size - self._head_offset
                take = min(available, out.size - filled)
                out[filled : filled + take] = head[self._head_offset : self._head_offset + take]
                filled += take
                self._head_offset += take
                if self._head_offset >= head.size:
                    self._pending.popleft()
                    self._head_offset = 0

            self._pending_samples -= filled
            self._samples_played += filled

            if filled < out.size:
                out[filled:] = 0.0

            if not self._pending:
                # Queue exhausted: return to the gated state so the next
                # utterance prebuffers, and wake anyone in flush().
                self._playing = False
                self._drained.set()
