"""Unit tests for resampling, filtering and gain."""

from __future__ import annotations

import numpy as np
import pytest

from ai_interpreter.domain.value_objects import SampleRate
from ai_interpreter.infrastructure.audio.dsp import (
    AudioPreprocessor,
    HighPassFilter,
    StreamingResampler,
    apply_gain,
    design_high_pass_taps,
)

pytestmark = pytest.mark.unit


def _tone(frequency: float, sample_rate: int, duration_s: float = 0.25) -> np.ndarray:
    """Generate a sine wave.

    Args:
        frequency: Tone frequency in hertz.
        sample_rate: Sample rate.
        duration_s: Length in seconds.

    Returns:
        Float32 samples.
    """
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


def _rms(samples: np.ndarray) -> float:
    """Root-mean-square level of a signal.

    Args:
        samples: Input samples.

    Returns:
        RMS level.
    """
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class TestApplyGain:
    """Decibel gain and clipping."""

    def test_zero_db_is_a_no_op(self) -> None:
        samples = _tone(440, 16000)
        assert apply_gain(samples, 0.0) is samples

    def test_six_db_roughly_doubles_amplitude(self) -> None:
        samples = np.array([0.25], dtype=np.float32)
        assert float(apply_gain(samples, 6.0)[0]) == pytest.approx(0.4988, abs=1e-3)

    def test_negative_gain_attenuates(self) -> None:
        samples = np.array([0.5], dtype=np.float32)
        assert float(apply_gain(samples, -6.0)[0]) == pytest.approx(0.2506, abs=1e-3)

    def test_clips_rather_than_wrapping(self) -> None:
        samples = np.array([0.9, -0.9], dtype=np.float32)
        boosted = apply_gain(samples, 20.0)

        assert float(boosted[0]) == pytest.approx(1.0)
        assert float(boosted[1]) == pytest.approx(-1.0)


class TestHighPassDesign:
    """Filter coefficient design."""

    def test_rejects_even_tap_counts(self) -> None:
        with pytest.raises(ValueError, match="must be odd"):
            design_high_pass_taps(80.0, 16000, num_taps=100)

    def test_rejects_cutoff_above_nyquist(self) -> None:
        with pytest.raises(ValueError, match="between 0 and Nyquist"):
            design_high_pass_taps(9000.0, 16000)

    def test_coefficients_sum_to_about_zero(self) -> None:
        # A high-pass must reject DC, which means its taps sum to zero.
        taps = design_high_pass_taps(80.0, 16000)
        assert float(np.sum(taps)) == pytest.approx(0.0, abs=1e-6)


class TestHighPassFilter:
    """Stateful filtering behaviour."""

    def test_removes_a_dc_offset(self) -> None:
        filt = HighPassFilter(80.0, 16000)
        constant = np.ones(4000, dtype=np.float32)
        filtered = filt.process(constant)

        # Ignore the settling region at the start.
        assert _rms(filtered[500:]) < 0.01

    def test_attenuates_low_frequencies(self) -> None:
        filt = HighPassFilter(80.0, 16000)
        rumble = _tone(20, 16000, duration_s=0.5)
        filtered = filt.process(rumble)

        assert _rms(filtered[500:]) < _rms(rumble) * 0.2

    def test_passes_speech_frequencies(self) -> None:
        filt = HighPassFilter(80.0, 16000)
        speech_band = _tone(500, 16000, duration_s=0.5)
        filtered = filt.process(speech_band)

        assert _rms(filtered[500:]) == pytest.approx(_rms(speech_band), rel=0.1)

    def test_output_length_always_matches_input(self) -> None:
        filt = HighPassFilter(80.0, 16000)
        for size in (1, 100, 512, 4096):
            assert filt.process(np.zeros(size, dtype=np.float32)).size == size

    def test_chunked_filtering_matches_whole_stream(self) -> None:
        # This is the point of keeping filter state: without it, every chunk
        # boundary produces a discontinuity that buzzes at the chunk rate.
        signal = _tone(300, 16000, duration_s=0.5)

        whole = HighPassFilter(80.0, 16000).process(signal)

        chunked_filter = HighPassFilter(80.0, 16000)
        chunks = [chunked_filter.process(signal[i : i + 512]) for i in range(0, signal.size, 512)]
        chunked = np.concatenate(chunks)

        np.testing.assert_allclose(whole, chunked, atol=1e-6)

    def test_reset_clears_history(self) -> None:
        filt = HighPassFilter(80.0, 16000)
        filt.process(np.ones(1000, dtype=np.float32))
        filt.reset()
        first = filt.process(np.zeros(200, dtype=np.float32))

        np.testing.assert_allclose(first, np.zeros(200), atol=1e-7)

    def test_reports_its_group_delay(self) -> None:
        filt = HighPassFilter(80.0, 16000, num_taps=101)
        assert filt.group_delay_samples == 50
        assert filt.group_delay_ms == pytest.approx(3.125)

    def test_handles_empty_input(self) -> None:
        assert HighPassFilter(80.0, 16000).process(np.empty(0, dtype=np.float32)).size == 0


class TestStreamingResampler:
    """Sample rate conversion across chunk boundaries."""

    def test_matching_rates_are_passthrough(self) -> None:
        resampler = StreamingResampler(16000, 16000)
        samples = _tone(440, 16000)

        assert resampler.is_passthrough
        assert resampler.process(samples) is samples

    def test_downsampling_produces_the_expected_length(self) -> None:
        resampler = StreamingResampler(48000, 16000)
        assert resampler.ratio == pytest.approx(1 / 3)

        out = resampler.process(_tone(440, 48000, duration_s=1.0), last=True)
        assert out.size == pytest.approx(16000, abs=200)

    def test_preserves_level_of_an_in_band_tone(self) -> None:
        resampler = StreamingResampler(48000, 16000)
        tone = _tone(440, 48000, duration_s=1.0)
        out = resampler.process(tone, last=True)

        assert _rms(out[200:-200]) == pytest.approx(_rms(tone), rel=0.05)

    def test_handles_non_integer_ratios(self) -> None:
        # 44.1 kHz to 16 kHz is the case a naive per-chunk resampler breaks on.
        resampler = StreamingResampler(44100, 16000)
        out = resampler.process(_tone(440, 44100, duration_s=1.0), last=True)

        assert out.size == pytest.approx(16000, abs=300)

    def test_chunked_output_totals_the_same_length(self) -> None:
        signal = _tone(440, 48000, duration_s=1.0)

        chunked_resampler = StreamingResampler(48000, 16000)
        pieces = [
            chunked_resampler.process(signal[i : i + 960]) for i in range(0, signal.size, 960)
        ]
        pieces.append(chunked_resampler.process(np.empty(0, dtype=np.float32), last=True))
        total = sum(piece.size for piece in pieces)

        assert total == pytest.approx(16000, abs=200)

    def test_returns_float32(self) -> None:
        resampler = StreamingResampler(48000, 16000)
        assert resampler.process(_tone(440, 48000, duration_s=0.1)).dtype == np.float32


class TestAudioPreprocessor:
    """The complete conditioning chain."""

    def test_converts_rate_and_reports_it(self) -> None:
        pre = AudioPreprocessor(SampleRate(48000), SampleRate(16000), high_pass_hz=80)
        out = pre.process(_tone(440, 48000, duration_s=1.0), last=True)

        assert pre.output_rate == SampleRate(16000)
        assert out.size == pytest.approx(16000, abs=200)

    def test_removes_rumble_but_keeps_speech(self) -> None:
        pre = AudioPreprocessor(SampleRate(48000), SampleRate(16000), high_pass_hz=80)
        mixed = _tone(30, 48000, duration_s=1.0) + _tone(600, 48000, duration_s=1.0)
        out = pre.process(mixed, last=True)

        speech_only = AudioPreprocessor(
            SampleRate(48000), SampleRate(16000), high_pass_hz=80
        ).process(_tone(600, 48000, duration_s=1.0), last=True)

        assert _rms(out[400:-400]) == pytest.approx(_rms(speech_only[400:-400]), rel=0.15)

    def test_applies_gain(self) -> None:
        quiet = (_tone(440, 16000, duration_s=0.5) * 0.1).astype(np.float32)

        flat = AudioPreprocessor(SampleRate(16000), SampleRate(16000)).process(quiet)
        boosted = AudioPreprocessor(SampleRate(16000), SampleRate(16000), gain_db=6.0).process(
            quiet
        )

        assert _rms(boosted) == pytest.approx(_rms(flat) * 2.0, rel=0.05)

    def test_reports_added_latency(self) -> None:
        with_filter = AudioPreprocessor(SampleRate(16000), SampleRate(16000), high_pass_hz=80)
        without = AudioPreprocessor(SampleRate(16000), SampleRate(16000))

        assert with_filter.added_latency_ms == pytest.approx(3.125)
        assert without.added_latency_ms == 0.0

    def test_reset_clears_filter_state(self) -> None:
        pre = AudioPreprocessor(SampleRate(16000), SampleRate(16000), high_pass_hz=80)
        pre.process(np.ones(1000, dtype=np.float32))
        pre.reset()
        out = pre.process(np.zeros(200, dtype=np.float32))

        np.testing.assert_allclose(out, np.zeros(200), atol=1e-7)
