"""Turning a stream of scored frames into discrete utterances.

This is the state machine that decides when someone started speaking and,
more importantly, when they stopped. It is pure logic - frames and
probabilities in, utterances out - with no audio device, no model and no
threads, so every timing rule is testable deterministically.

Four parameters control it, and each one is a real trade-off:

``threshold``
    Probability above which a frame counts as speech.
``min_speech_ms``
    Speech must persist this long before an utterance starts. Rejects
    keyboard clicks and door slams without delaying anything, because the
    pre-roll buffer supplies the audio retroactively.
``min_silence_ms``
    Silence needed to declare the utterance finished. **This is the single
    largest fixed cost in the end-to-end latency budget.** Too short and the
    speaker is cut off mid-sentence; too long and every translation is late.
``pre_roll_ms``
    Audio retained from *before* speech was detected. Without it the first
    syllable is always clipped, because detection inevitably lags onset - and
    a clipped first syllable is a recognition error, not a cosmetic issue.

The trailing silence is deliberately kept in the emitted utterance. Speech
recognisers use it to decide that the final word has ended, and trimming it
costs accuracy on the last word of every sentence.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from ai_interpreter.domain.entities import AudioFrame, Utterance, UtteranceId
from ai_interpreter.domain.value_objects import LanguageCode, SampleRate

__all__ = ["SegmenterState", "SegmenterStats", "UtteranceSegmenter"]

logger = logging.getLogger(__name__)


class SegmenterState(StrEnum):
    """What the segmenter currently believes is happening."""

    SILENCE = "silence"
    """No speech; frames are kept only as pre-roll."""

    MAYBE_SPEECH = "maybe_speech"
    """Speech detected but not yet long enough to be believed."""

    SPEECH = "speech"
    """Inside an utterance."""

    TRAILING_SILENCE = "trailing_silence"
    """Speech has stopped; waiting to see whether it resumes."""


@dataclass(frozen=True, slots=True)
class SegmenterStats:
    """Counters describing what the segmenter has done.

    Args:
        frames_processed: Frames pushed in total.
        utterances_emitted: Completed utterances produced.
        rejected_short_bursts: Sounds discarded for being too brief.
        forced_cuts: Utterances ended by the maximum-length guard rather than
            by silence.
    """

    frames_processed: int
    utterances_emitted: int
    rejected_short_bursts: int
    forced_cuts: int


class UtteranceSegmenter:
    """Groups scored audio frames into utterances.

    Args:
        sample_rate: Rate of the frames pushed in.
        threshold: Speech probability above which a frame counts as speech.
        min_speech_ms: Speech required before an utterance starts.
        min_silence_ms: Silence required to end an utterance.
        pre_roll_ms: Audio retained from before speech onset.
        max_utterance_ms: Hard limit, for a speaker who never pauses.
        language: Language tag attached to emitted utterances.

    Raises:
        ValueError: If the parameters cannot produce a usable utterance.
    """

    def __init__(
        self,
        sample_rate: SampleRate,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 350,
        pre_roll_ms: int = 300,
        max_utterance_ms: int = 12000,
        language: LanguageCode | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            msg = f"threshold must be within [0.0, 1.0], got {threshold}"
            raise ValueError(msg)
        if min_silence_ms >= max_utterance_ms:
            msg = (
                f"min_silence_ms ({min_silence_ms}) must be smaller than "
                f"max_utterance_ms ({max_utterance_ms})"
            )
            raise ValueError(msg)

        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._pre_roll_ms = pre_roll_ms
        self._max_utterance_ms = max_utterance_ms
        self._language = language

        self._state = SegmenterState.SILENCE
        self._pre_roll: deque[NDArray[np.float32]] = deque()
        self._pre_roll_samples = 0
        self._pre_roll_capacity = sample_rate.samples_for_ms(pre_roll_ms)
        # Frames seen since speech was first suspected. Held separately from
        # the pre-roll ring and never capped: these are candidate *speech*,
        # and the ring would discard them whenever min_speech_ms exceeds
        # pre_roll_ms - clipping the start of every utterance.
        self._pending_speech: list[NDArray[np.float32]] = []
        self._active: list[NDArray[np.float32]] = []
        self._active_samples = 0
        self._speech_run_ms = 0.0
        self._silence_run_ms = 0.0
        self._utterance_start_ms = 0.0

        self._frames_processed = 0
        self._utterances_emitted = 0
        self._rejected_short_bursts = 0
        self._forced_cuts = 0

    # -- observation -------------------------------------------------------
    @property
    def state(self) -> SegmenterState:
        """Current state of the machine."""
        return self._state

    @property
    def is_speaking(self) -> bool:
        """Whether an utterance is currently being collected."""
        return self._state in (SegmenterState.SPEECH, SegmenterState.TRAILING_SILENCE)

    @property
    def active_duration_ms(self) -> float:
        """Length of the utterance being collected."""
        return self._sample_rate.ms_for_samples(self._active_samples)

    def stats(self) -> SegmenterStats:
        """Return the counters gathered so far.

        Returns:
            A snapshot of segmentation activity.
        """
        return SegmenterStats(
            frames_processed=self._frames_processed,
            utterances_emitted=self._utterances_emitted,
            rejected_short_bursts=self._rejected_short_bursts,
            forced_cuts=self._forced_cuts,
        )

    # -- main entry point --------------------------------------------------
    def push(self, frame: AudioFrame, speech_probability: float) -> Utterance | None:
        """Feed one scored frame into the state machine.

        Args:
            frame: The audio frame.
            speech_probability: Its speech probability.

        Returns:
            A completed utterance, or ``None`` if none finished on this frame.
        """
        self._frames_processed += 1
        is_speech = speech_probability >= self._threshold
        frame_ms = frame.duration_ms

        if self._state is SegmenterState.SILENCE:
            if is_speech:
                self._state = SegmenterState.MAYBE_SPEECH
                self._speech_run_ms = frame_ms
                self._pending_speech = [frame.pcm]
            else:
                self._remember_pre_roll(frame.pcm)
            return None

        if self._state is SegmenterState.MAYBE_SPEECH:
            if not is_speech:
                # A false alarm. The frames were not speech after all, so they
                # become ordinary pre-roll for whatever comes next.
                self._rejected_short_bursts += 1
                for pcm in self._pending_speech:
                    self._remember_pre_roll(pcm)
                self._remember_pre_roll(frame.pcm)
                self._pending_speech = []
                self._state = SegmenterState.SILENCE
                self._speech_run_ms = 0.0
                return None

            self._pending_speech.append(frame.pcm)
            self._speech_run_ms += frame_ms
            if self._speech_run_ms >= self._min_speech_ms:
                self._begin_utterance(frame)
            return None

        # SPEECH or TRAILING_SILENCE: the frame belongs to the utterance.
        self._active.append(frame.pcm)
        self._active_samples += frame.pcm.size

        if is_speech:
            self._state = SegmenterState.SPEECH
            self._silence_run_ms = 0.0
        else:
            self._state = SegmenterState.TRAILING_SILENCE
            self._silence_run_ms += frame_ms
            if self._silence_run_ms >= self._min_silence_ms:
                return self._finish_utterance(frame, forced=False)

        if self.active_duration_ms >= self._max_utterance_ms:
            self._forced_cuts += 1
            logger.debug("Utterance reached the %d ms limit and was cut", self._max_utterance_ms)
            return self._finish_utterance(frame, forced=True)

        return None

    def flush(self, timestamp_ms: float | None = None) -> Utterance | None:
        """Emit whatever is in progress, e.g. when capture stops.

        Args:
            timestamp_ms: End time to record, or ``None`` to derive it from
                the collected audio.

        Returns:
            The final utterance, or ``None`` if nothing was in progress.
        """
        if not self._active:
            self.reset()
            return None

        end_ms = (
            timestamp_ms
            if timestamp_ms is not None
            else self._utterance_start_ms + self.active_duration_ms
        )
        return self._emit(end_ms)

    def reset(self) -> None:
        """Return to silence, discarding any partial utterance."""
        self._state = SegmenterState.SILENCE
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._pending_speech = []
        self._active.clear()
        self._active_samples = 0
        self._speech_run_ms = 0.0
        self._silence_run_ms = 0.0

    # -- internals ---------------------------------------------------------
    def _remember_pre_roll(self, pcm: NDArray[np.float32]) -> None:
        """Keep a frame in the pre-roll ring, discarding the oldest.

        Args:
            pcm: Frame samples.
        """
        self._pre_roll.append(pcm)
        self._pre_roll_samples += pcm.size
        while self._pre_roll_samples > self._pre_roll_capacity and len(self._pre_roll) > 1:
            self._pre_roll_samples -= self._pre_roll.popleft().size

    def _begin_utterance(self, frame: AudioFrame) -> None:
        """Promote the buffered pre-roll into a new utterance.

        Args:
            frame: Frame that confirmed speech.
        """
        self._state = SegmenterState.SPEECH
        self._silence_run_ms = 0.0

        # Pre-roll first (audio from before onset), then every frame collected
        # while onset was being confirmed. Both are needed: dropping either
        # clips the beginning of the utterance.
        self._active = [*self._pre_roll, *self._pending_speech]
        self._active_samples = sum(pcm.size for pcm in self._active)

        # The utterance began where the retained audio began, not where
        # detection completed - that is the whole point of keeping it.
        retained_ms = self._sample_rate.ms_for_samples(self._active_samples)
        self._utterance_start_ms = max(0.0, frame.timestamp_ms + frame.duration_ms - retained_ms)

        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._pending_speech = []

    def _finish_utterance(self, frame: AudioFrame, *, forced: bool) -> Utterance:
        """Complete the active utterance.

        Args:
            frame: Frame on which it ended.
            forced: Whether the maximum-length guard ended it.

        Returns:
            The finished utterance.
        """
        utterance = self._emit(frame.timestamp_ms + frame.duration_ms)
        if forced:
            # A forced cut means the speaker is still talking, so the next
            # utterance starts immediately rather than waiting for onset.
            self._state = SegmenterState.SPEECH
            self._utterance_start_ms = frame.timestamp_ms + frame.duration_ms
            self._silence_run_ms = 0.0
        return utterance

    def _emit(self, end_ms: float) -> Utterance:
        """Build an utterance from the collected frames and reset.

        Args:
            end_ms: End timestamp.

        Returns:
            The utterance.
        """
        pcm = np.concatenate(self._active) if self._active else np.empty(0, dtype=np.float32)
        utterance = Utterance(
            id=UtteranceId(uuid.uuid4().hex[:16]),
            pcm=pcm.astype(np.float32, copy=False),
            sample_rate=self._sample_rate,
            started_at_ms=self._utterance_start_ms,
            ended_at_ms=end_ms,
            language=self._language,
        )

        self._utterances_emitted += 1
        logger.debug(
            "Utterance %s: %.0f ms of audio (%.0f ms to %.0f ms)",
            utterance.id,
            utterance.duration_ms,
            utterance.started_at_ms,
            utterance.ended_at_ms,
        )

        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._pending_speech = []
        self._active = []
        self._active_samples = 0
        self._speech_run_ms = 0.0
        self._silence_run_ms = 0.0
        self._state = SegmenterState.SILENCE
        return utterance
