"""Background capture: microphone to utterances.

Owns the worker thread that drives the capture chain::

    AudioSource -> AudioPreprocessor -> FrameAssembler -> VAD -> Segmenter

and hands finished utterances to a callback. In Phase 9 that callback becomes
the entry point of the streaming pipeline; today it feeds the recorder and the
command line report.

Threading contract:

* The **driver callback thread** only fills a buffer (see
  :mod:`ai_interpreter.infrastructure.audio.buffers`).
* **This worker thread** does resampling, filtering, voice activity detection
  and segmentation. Together those cost about 2 % of one core, which is why
  they can share a thread on a 2-core machine.
* The **caller's thread** starts and stops the session and reads statistics.

Callbacks are invoked on the worker thread. Anything expensive done inside one
delays capture and eventually drops audio, so the user interface will marshal
them onto the Qt thread in Phase 8 rather than working directly.

Exceptions in the worker are captured rather than allowed to kill the thread
silently: a capture session that has quietly stopped, with the UI still
showing "listening", is the worst possible failure mode.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from ai_interpreter.application.services.utterance_segmenter import (
    SegmenterState,
    UtteranceSegmenter,
)
from ai_interpreter.domain.entities import AudioFrame, Utterance
from ai_interpreter.domain.ports import AudioSource, VoiceActivityDetector
from ai_interpreter.infrastructure.audio.buffers import FrameAssembler
from ai_interpreter.infrastructure.audio.dsp import AudioPreprocessor

__all__ = ["CaptureSession", "CaptureStats"]

logger = logging.getLogger(__name__)

# How long the worker waits for audio before looping to re-check the stop
# flag. Short enough that stopping feels instant, long enough not to spin.
_READ_TIMEOUT_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class CaptureStats:
    """Snapshot of a capture session.

    Args:
        frames_captured: Blocks taken from the audio source.
        vad_frames: Fixed-size frames scored by the detector.
        utterances: Utterances emitted.
        dropped_blocks: Blocks lost because the worker fell behind. Any
            non-zero value means audio was lost.
        peak_level: Loudest sample seen, for level metering.
        speech_ratio: Fraction of frames the detector considered speech.
    """

    frames_captured: int
    vad_frames: int
    utterances: int
    dropped_blocks: int
    peak_level: float
    speech_ratio: float


class CaptureSession:
    """Runs the capture chain on a background thread.

    Args:
        source: Where audio comes from.
        preprocessor: Resampling, filtering and gain.
        vad: Voice activity detector.
        segmenter: Utterance state machine.
        on_utterance: Called with each completed utterance.
        on_frame: Called with every processed frame and its speech
            probability, for level meters and live waveform display.
        on_state_change: Called when speech starts or stops.
        on_error: Called if the worker fails. The session stops afterwards.
    """

    def __init__(
        self,
        source: AudioSource,
        preprocessor: AudioPreprocessor,
        vad: VoiceActivityDetector,
        segmenter: UtteranceSegmenter,
        on_utterance: Callable[[Utterance], None] | None = None,
        on_frame: Callable[[AudioFrame, float], None] | None = None,
        on_state_change: Callable[[SegmenterState], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._source = source
        self._preprocessor = preprocessor
        self._vad = vad
        self._segmenter = segmenter
        self._assembler = FrameAssembler(vad.required_frame_samples)

        self._on_utterance = on_utterance
        self._on_frame = on_frame
        self._on_state_change = on_state_change
        self._on_error = on_error

        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._error: Exception | None = None

        self._frames_captured = 0
        self._vad_frames = 0
        self._speech_frames = 0
        self._utterances = 0
        self._peak_level = 0.0
        self._frame_index = 0

    # -- lifecycle ---------------------------------------------------------
    @property
    def is_running(self) -> bool:
        """Whether the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Exception | None:
        """The exception that stopped the worker, if any."""
        return self._error

    def start(self) -> None:
        """Open the source and start the worker thread. Idempotent.

        Raises:
            DeviceError: If the audio source cannot be opened.
        """
        if self.is_running:
            return

        self._stop_requested.clear()
        self._error = None
        self._source.start()

        self._thread = threading.Thread(
            target=self._run,
            name="audio-capture",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Capture session started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker and release the source. Idempotent.

        Args:
            timeout: Seconds to wait for the worker to finish.
        """
        self._stop_requested.set()

        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                # Daemon thread, so it cannot block interpreter shutdown, but
                # this indicates a bug worth knowing about.
                logger.warning("Capture worker did not stop within %.1f s", timeout)

        self._source.stop()
        logger.debug("Capture session stopped")

    def stats(self) -> CaptureStats:
        """Return a snapshot of activity.

        Returns:
            Current capture statistics.
        """
        dropped = getattr(self._source, "dropped_blocks", 0)
        ratio = self._speech_frames / self._vad_frames if self._vad_frames else 0.0
        return CaptureStats(
            frames_captured=self._frames_captured,
            vad_frames=self._vad_frames,
            utterances=self._utterances,
            dropped_blocks=int(dropped),
            peak_level=self._peak_level,
            speech_ratio=ratio,
        )

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        """Worker thread body: pull, process, segment, until stopped."""
        try:
            self._vad.reset()
            while not self._stop_requested.is_set():
                block = self._source.read(timeout=_READ_TIMEOUT_SECONDS)
                if block is None:
                    if self._source_is_exhausted():
                        break
                    continue
                self._frames_captured += 1
                self._process_block(block)

            self._drain()
        except Exception as exc:
            self._error = exc
            logger.exception("Capture worker failed")
            if self._on_error is not None:
                self._on_error(exc)

    def _source_is_exhausted(self) -> bool:
        """Whether a finite source (a WAV file) has reached its end.

        Returns:
            ``True`` when the worker should stop of its own accord.
        """
        return bool(getattr(self._source, "is_exhausted", False))

    def _process_block(self, block: AudioFrame) -> None:
        """Condition one driver block and segment the frames it yields.

        Args:
            block: Raw block from the audio source.
        """
        conditioned = self._preprocessor.process(block.pcm)
        for samples in self._assembler.push(conditioned):
            self._handle_frame(samples)

    def _handle_frame(self, samples) -> None:  # type: ignore[no-untyped-def]
        """Score and segment one fixed-size frame.

        Args:
            samples: Exactly the number of samples the detector requires.
        """
        rate = self._preprocessor.output_rate
        frame_ms = rate.ms_for_samples(samples.size)
        frame = AudioFrame(
            pcm=samples,
            sample_rate=rate,
            timestamp_ms=self._frame_index * frame_ms,
        )
        self._frame_index += 1
        self._vad_frames += 1
        self._peak_level = max(self._peak_level, frame.peak_amplitude)

        probability = self._vad.speech_probability(frame)
        if probability >= 0.5:
            self._speech_frames += 1

        if self._on_frame is not None:
            self._on_frame(frame, probability)

        previous_state = self._segmenter.state
        utterance = self._segmenter.push(frame, probability)
        self._notify_state_change(previous_state)

        if utterance is not None:
            self._utterances += 1
            # Silero is recurrent: carrying one speaker's hidden state into
            # the next utterance degrades its first frames.
            self._vad.reset()
            if self._on_utterance is not None:
                self._on_utterance(utterance)

    def _notify_state_change(self, previous_state: SegmenterState) -> None:
        """Report a speaking/not-speaking transition.

        Args:
            previous_state: State before the last frame was pushed.
        """
        if self._on_state_change is None:
            return
        current = self._segmenter.state
        if current is not previous_state:
            self._on_state_change(current)

    def _drain(self) -> None:
        """Emit whatever remains when capture stops.

        Without this, the last utterance is lost whenever a recording ends
        while someone is still speaking - which is exactly what happens at the
        end of a fixed-length test recording.
        """
        tail = self._assembler.flush()
        if tail is not None:
            self._handle_frame(tail)

        utterance = self._segmenter.flush()
        if utterance is not None:
            self._utterances += 1
            if self._on_utterance is not None:
                self._on_utterance(utterance)
