"""Energy-based voice activity detection.

A fallback for when the Silero model cannot be downloaded or loaded. It needs
no model, no ONNX Runtime and no network - only numpy - so the application
always has a working detector.

It is genuinely worse than the neural detector and the code says so rather
than pretending otherwise: energy cannot distinguish speech from a fan, a
keyboard or music. What it *can* do is track the background noise level and
trigger on sound significantly above it, which is adequate in a quiet room.

The noise floor is estimated with an asymmetric exponential moving average:
it rises slowly and falls quickly. That asymmetry matters. A symmetric average
drifts upward during a long sentence until the speaker's own voice becomes the
"noise floor" and the detector stops hearing them.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

from ai_interpreter.domain.entities import AudioFrame
from ai_interpreter.domain.value_objects import SampleRate

__all__ = ["EnergyVad"]

logger = logging.getLogger(__name__)

# Matches Silero's framing so the two are interchangeable in the pipeline.
_DEFAULT_FRAME_SAMPLES: Final[int] = 512
_DEFAULT_SAMPLE_RATE: Final[int] = 16000

# Adaptation rates per frame. Rising slowly and falling quickly keeps the
# estimate anchored to true silence rather than creeping up during speech.
_NOISE_RISE_RATE: Final[float] = 0.002
_NOISE_FALL_RATE: Final[float] = 0.05

# Absolute floor, so a perfectly silent digital input does not make any
# non-zero sample look like speech.
_MIN_NOISE_FLOOR: Final[float] = 1e-4

# Smallest fraction of the adaptation rate that still applies while a frame
# looks like speech. Without it, a detector started in a permanently noisy
# room scores every frame at 1.0, which zeroes the adaptation rate, which
# keeps every frame at 1.0 - it never recalibrates and reports speech forever.
# A small residual rate lets the estimate creep up over a couple of minutes,
# which is far too slow to be dragged along by an ordinary sentence.
_MIN_ADAPTATION_FACTOR: Final[float] = 0.02


class EnergyVad:
    """Adaptive energy detector, satisfying the ``VoiceActivityDetector`` port.

    Args:
        frame_samples: Samples per frame.
        sample_rate: Rate of incoming audio.
        snr_db: Level above the noise floor, in decibels, that counts as
            certain speech.
    """

    def __init__(
        self,
        frame_samples: int = _DEFAULT_FRAME_SAMPLES,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        snr_db: float = 12.0,
    ) -> None:
        self._frame_samples = frame_samples
        self._sample_rate = SampleRate(sample_rate)
        self._snr_ratio = float(10.0 ** (snr_db / 20.0))
        self._noise_floor = _MIN_NOISE_FLOOR
        self._frames_scored = 0

    # -- port interface ----------------------------------------------------
    @property
    def required_frame_samples(self) -> int:
        """Samples expected per call."""
        return self._frame_samples

    @property
    def sample_rate(self) -> SampleRate:
        """Rate expected."""
        return self._sample_rate

    @property
    def noise_floor(self) -> float:
        """Current background level estimate, as RMS amplitude."""
        return self._noise_floor

    @property
    def frames_scored(self) -> int:
        """Frames processed since creation."""
        return self._frames_scored

    def warmup(self) -> None:
        """No-op: there is no model to load."""

    def speech_probability(self, frame: AudioFrame) -> float:
        """Score a frame for speech.

        Args:
            frame: Frame of audio.

        Returns:
            Probability in ``[0.0, 1.0]``, scaled between the noise floor and
            the configured signal-to-noise threshold.

        Raises:
            ValueError: If the frame size is wrong.
        """
        if frame.pcm.size != self._frame_samples:
            msg = (
                f"EnergyVad is configured for {self._frame_samples} samples per frame, "
                f"got {frame.pcm.size}"
            )
            raise ValueError(msg)

        self._frames_scored += 1
        rms = float(np.sqrt(np.mean(np.square(frame.pcm, dtype=np.float64))))

        threshold = self._noise_floor * self._snr_ratio
        if rms <= self._noise_floor:
            probability = 0.0
        elif rms >= threshold:
            probability = 1.0
        else:
            # Linear in decibels rather than amplitude: loudness is
            # logarithmic, so a linear amplitude ramp would spend most of its
            # range on levels the ear treats as nearly identical.
            span_db = 20.0 * np.log10(threshold / self._noise_floor)
            level_db = 20.0 * np.log10(rms / self._noise_floor)
            probability = float(np.clip(level_db / span_db, 0.0, 1.0))

        self._update_noise_floor(rms, probability)
        return probability

    def reset(self) -> None:
        """Forget the noise floor estimate, e.g. after a device change."""
        self._noise_floor = _MIN_NOISE_FLOOR

    def close(self) -> None:
        """No-op: there is nothing to release."""

    # -- internals ---------------------------------------------------------
    def _update_noise_floor(self, rms: float, probability: float) -> None:
        """Adapt the background level estimate.

        Args:
            rms: Level of the frame just scored.
            probability: Its speech probability. Frames that look like speech
                barely move the estimate, so a long sentence cannot drag the
                floor up to the speaker's own level - but they still move it
                slightly, so a noisy room is eventually recalibrated to.
        """
        rate = _NOISE_FALL_RATE if rms < self._noise_floor else _NOISE_RISE_RATE
        rate *= max(1.0 - probability, _MIN_ADAPTATION_FACTOR)
        updated = (1.0 - rate) * self._noise_floor + rate * rms
        self._noise_floor = max(updated, _MIN_NOISE_FLOOR)
