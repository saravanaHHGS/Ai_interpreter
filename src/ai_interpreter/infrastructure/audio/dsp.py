"""Signal conditioning between the microphone and the speech models.

Three operations, in this order:

1. **Resample** the device rate (48 kHz) to the model rate (16 kHz).
2. **High-pass filter** to remove rumble, desk thumps and mains hum.
3. **Gain** to bring a quiet microphone up to a usable level.

Resampling comes first because filtering costs three times less at 16 kHz
than at 48 kHz, and nothing below 8 kHz is harmed by the order.

Both the resampler and the filter are *stateful*. Audio arrives in chunks, and
a filter restarted at every chunk boundary produces a discontinuity at each
one - an audible buzz at the chunk rate, and a measurable accuracy loss for
speech recognition. Each class therefore carries the tail of the previous
chunk, so the result is bit-identical to filtering the whole stream at once.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ai_interpreter.domain.errors import AudioCaptureError
from ai_interpreter.domain.value_objects import SampleRate

__all__ = [
    "AudioPreprocessor",
    "HighPassFilter",
    "StreamingResampler",
    "apply_gain",
    "design_high_pass_taps",
]

logger = logging.getLogger(__name__)

# Odd tap count so the filter has an exact integer group delay of
# (taps - 1) / 2 samples - 50 samples, or 3.1 ms at 16 kHz. Recorded here
# because it is part of the latency budget.
_DEFAULT_TAPS: Final[int] = 101


def apply_gain(samples: NDArray[np.float32], gain_db: float) -> NDArray[np.float32]:
    """Scale samples by a decibel gain and clip to the valid range.

    Args:
        samples: Mono float32 samples.
        gain_db: Gain in decibels. ``0.0`` returns the input unchanged.

    Returns:
        Scaled samples clipped to ``[-1.0, 1.0]``.
    """
    if gain_db == 0.0:
        return samples
    factor = float(10.0 ** (gain_db / 20.0))
    scaled: NDArray[np.float32] = np.clip(samples * factor, -1.0, 1.0, dtype=np.float32)
    return scaled


def design_high_pass_taps(
    cutoff_hz: float, sample_rate: int, num_taps: int = _DEFAULT_TAPS
) -> NDArray[np.float32]:
    """Design a linear-phase FIR high-pass filter.

    Built as a windowed-sinc low-pass followed by spectral inversion. This is
    the textbook construction and needs nothing beyond numpy - adding SciPy
    for one filter would cost roughly 60 MB of install size for no benefit.

    Args:
        cutoff_hz: -6 dB cutoff frequency.
        sample_rate: Sample rate the filter will run at.
        num_taps: Filter length. Must be odd, so the group delay is a whole
            number of samples.

    Returns:
        Filter coefficients.

    Raises:
        ValueError: If the tap count is even or the cutoff is out of range.
    """
    if num_taps % 2 == 0:
        msg = f"num_taps must be odd for linear phase, got {num_taps}"
        raise ValueError(msg)
    if not 0.0 < cutoff_hz < sample_rate / 2:
        msg = f"cutoff_hz must be between 0 and Nyquist ({sample_rate / 2}), got {cutoff_hz}"
        raise ValueError(msg)

    normalised_cutoff = cutoff_hz / sample_rate
    offsets = np.arange(num_taps) - (num_taps - 1) / 2

    low_pass = np.sinc(2 * normalised_cutoff * offsets) * np.hamming(num_taps)
    low_pass /= low_pass.sum()

    # Spectral inversion: an all-pass minus the low-pass leaves the high-pass.
    high_pass = -low_pass
    high_pass[(num_taps - 1) // 2] += 1.0

    taps: NDArray[np.float32] = high_pass.astype(np.float32)
    return taps


class HighPassFilter:
    """Stateful linear-phase FIR high-pass filter.

    Uses overlap-save: the last ``taps - 1`` input samples are retained so the
    next chunk is filtered with the correct history and output is continuous
    across chunk boundaries.

    Args:
        cutoff_hz: -6 dB cutoff frequency.
        sample_rate: Sample rate of the audio.
        num_taps: Filter length, must be odd.
    """

    def __init__(self, cutoff_hz: float, sample_rate: int, num_taps: int = _DEFAULT_TAPS) -> None:
        self._taps = design_high_pass_taps(cutoff_hz, sample_rate, num_taps)
        self._tail = np.zeros(num_taps - 1, dtype=np.float32)
        self._cutoff_hz = cutoff_hz
        self._sample_rate = sample_rate

    @property
    def group_delay_samples(self) -> int:
        """Constant delay the filter introduces, in samples."""
        return (self._taps.size - 1) // 2

    @property
    def group_delay_ms(self) -> float:
        """Constant delay the filter introduces, in milliseconds."""
        return self.group_delay_samples * 1000.0 / self._sample_rate

    def process(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter a chunk, continuing from the previous one.

        Args:
            samples: Mono float32 samples.

        Returns:
            Filtered samples, the same length as the input.
        """
        if not samples.size:
            return samples

        padded = np.concatenate((self._tail, samples))
        filtered = np.convolve(padded, self._taps, mode="valid")
        self._tail = padded[-self._taps.size + 1 :].astype(np.float32)
        return filtered.astype(np.float32)

    def reset(self) -> None:
        """Clear filter history, e.g. when the input device changes."""
        self._tail = np.zeros(self._taps.size - 1, dtype=np.float32)


class StreamingResampler:
    """Sample rate conversion that is continuous across chunks.

    Wraps ``soxr``'s streaming interface. A naive per-chunk resample would
    restart the polyphase filter each time, producing a discontinuity at every
    boundary, and would not handle ratios where the chunk length does not
    divide evenly - 44.1 kHz to 16 kHz being the obvious case.

    Args:
        input_rate: Incoming sample rate.
        output_rate: Desired sample rate.
        quality: soxr quality setting; ``"HQ"`` is transparent for speech and
            costs a fraction of a millisecond per frame.

    Raises:
        AudioCaptureError: If the resampler cannot be created.
    """

    def __init__(self, input_rate: int, output_rate: int, quality: str = "HQ") -> None:
        self._input_rate = input_rate
        self._output_rate = output_rate
        self._passthrough = input_rate == output_rate
        self._stream = None

        if self._passthrough:
            return

        try:
            import soxr

            self._stream = soxr.ResampleStream(
                in_rate=float(input_rate),
                out_rate=float(output_rate),
                num_channels=1,
                dtype="float32",
                quality=quality,
            )
        except Exception as exc:
            msg = f"Could not create a resampler for {input_rate} Hz -> {output_rate} Hz: {exc}"
            raise AudioCaptureError(msg) from exc

    @property
    def is_passthrough(self) -> bool:
        """Whether input and output rates match, making this a no-op."""
        return self._passthrough

    @property
    def ratio(self) -> float:
        """Output samples produced per input sample."""
        return self._output_rate / self._input_rate

    def process(self, samples: NDArray[np.float32], *, last: bool = False) -> NDArray[np.float32]:
        """Resample a chunk, continuing from the previous one.

        Args:
            samples: Mono float32 samples.
            last: ``True`` for the final chunk, flushing internal state.

        Returns:
            Resampled samples. The length varies between calls, which is
            expected - use :class:`FrameAssembler` to regain fixed frames.
        """
        if self._passthrough:
            return samples
        assert self._stream is not None
        resampled = self._stream.resample_chunk(samples, last=last)
        return np.asarray(resampled, dtype=np.float32).reshape(-1)


class AudioPreprocessor:
    """Resample, filter and apply gain, turning device audio into model audio.

    Args:
        input_rate: Rate audio is captured at.
        output_rate: Rate the speech models require.
        high_pass_hz: High-pass cutoff, or ``None`` to skip filtering.
        gain_db: Gain applied after filtering.
    """

    def __init__(
        self,
        input_rate: SampleRate,
        output_rate: SampleRate,
        high_pass_hz: int | None = None,
        gain_db: float = 0.0,
    ) -> None:
        self._input_rate = input_rate
        self._output_rate = output_rate
        self._gain_db = gain_db
        self._resampler = StreamingResampler(input_rate.hz, output_rate.hz)
        self._high_pass = (
            HighPassFilter(float(high_pass_hz), output_rate.hz) if high_pass_hz else None
        )

    @property
    def output_rate(self) -> SampleRate:
        """Rate of the audio this preprocessor emits."""
        return self._output_rate

    @property
    def added_latency_ms(self) -> float:
        """Latency this stage contributes, for the latency budget."""
        return self._high_pass.group_delay_ms if self._high_pass else 0.0

    def process(self, samples: NDArray[np.float32], *, last: bool = False) -> NDArray[np.float32]:
        """Run the full conditioning chain over a chunk.

        Args:
            samples: Mono float32 samples at the input rate.
            last: ``True`` for the final chunk of a stream.

        Returns:
            Conditioned samples at the output rate.
        """
        resampled = self._resampler.process(samples, last=last)
        if self._high_pass is not None:
            resampled = self._high_pass.process(resampled)
        return apply_gain(resampled, self._gain_db)

    def reset(self) -> None:
        """Clear filter history. The resampler keeps its own phase."""
        if self._high_pass is not None:
            self._high_pass.reset()
