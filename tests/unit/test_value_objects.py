"""Unit tests for domain value objects."""

from __future__ import annotations

import pytest

from ai_interpreter.domain.value_objects import (
    Confidence,
    LanguageCode,
    LanguagePair,
    SampleRate,
    StageTiming,
)

pytestmark = pytest.mark.unit


class TestLanguageCode:
    """Language code validation and metadata."""

    def test_accepts_supported_code(self) -> None:
        assert LanguageCode("ta").code == "ta"

    def test_normalises_case_and_whitespace(self) -> None:
        assert LanguageCode("  TA ").code == "ta"

    def test_rejects_unsupported_code(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language code"):
            LanguageCode("zz")

    def test_exposes_english_and_native_names(self) -> None:
        tamil = LanguageCode("ta")
        assert tamil.english_name == "Tamil"
        assert tamil.native_name == "தமிழ்"

    def test_is_hashable_and_comparable(self) -> None:
        assert LanguageCode("en") == LanguageCode("EN")
        assert len({LanguageCode("en"), LanguageCode("en")}) == 1


class TestLanguagePair:
    """Translation direction behaviour."""

    def test_builds_from_raw_codes(self) -> None:
        pair = LanguagePair.of("ta", "en")
        assert pair.source.code == "ta"
        assert pair.target.code == "en"

    def test_rejects_identical_languages(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            LanguagePair.of("en", "en")

    def test_reversed_swaps_direction(self) -> None:
        assert LanguagePair.of("ta", "en").reversed() == LanguagePair.of("en", "ta")

    def test_key_is_stable_for_caching(self) -> None:
        assert LanguagePair.of("hi", "en").key == "hi-en"


class TestSampleRate:
    """Sample rate validation and conversions."""

    @pytest.mark.parametrize("hz", [8000, 16000, 22050, 24000, 32000, 44100, 48000])
    def test_accepts_supported_rates(self, hz: int) -> None:
        assert int(SampleRate(hz)) == hz

    def test_rejects_unsupported_rate(self) -> None:
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            SampleRate(12345)

    def test_converts_milliseconds_to_samples(self) -> None:
        assert SampleRate(16000).samples_for_ms(20) == 320

    def test_converts_samples_to_milliseconds(self) -> None:
        assert SampleRate(48000).ms_for_samples(960) == pytest.approx(20.0)

    def test_conversions_round_trip(self) -> None:
        rate = SampleRate(16000)
        assert rate.ms_for_samples(rate.samples_for_ms(350)) == pytest.approx(350.0)


class TestConfidence:
    """Confidence score validation."""

    def test_accepts_boundary_values(self) -> None:
        assert float(Confidence(0.0)) == 0.0
        assert float(Confidence(1.0)) == 1.0

    @pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
    def test_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            Confidence(value)

    def test_is_below_threshold(self) -> None:
        assert Confidence(0.3).is_below(0.5)
        assert not Confidence(0.7).is_below(0.5)


class TestStageTiming:
    """Latency sample validation."""

    def test_records_a_measurement(self) -> None:
        timing = StageTiming(stage="stt", duration_ms=120.5, utterance_id="u1")
        assert timing.stage == "stt"
        assert timing.duration_ms == pytest.approx(120.5)

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            StageTiming(stage="stt", duration_ms=-1.0, utterance_id="u1")
