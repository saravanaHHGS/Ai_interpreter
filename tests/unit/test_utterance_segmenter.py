"""Unit tests for the utterance state machine.

The segmenter is pure logic, so its timing rules can be verified exactly:
frames and probabilities go in, utterances come out, with no microphone, no
model and no threads involved. Every rule that affects latency or clipping is
pinned down here rather than discovered later with a stopwatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_interpreter.application.services.utterance_segmenter import (
    SegmenterState,
    UtteranceSegmenter,
)
from ai_interpreter.domain.entities import AudioFrame, Utterance
from ai_interpreter.domain.value_objects import LanguageCode, SampleRate

pytestmark = pytest.mark.unit

RATE = SampleRate(16000)
FRAME_SAMPLES = 512  # 32 ms, matching Silero
FRAME_MS = 32.0


def _make_segmenter(**overrides: object) -> UtteranceSegmenter:
    """Build a segmenter with test-friendly defaults.

    Args:
        **overrides: Parameters to override.

    Returns:
        The configured segmenter.
    """
    kwargs: dict[str, object] = {
        "sample_rate": RATE,
        "threshold": 0.5,
        "min_speech_ms": 64,  # 2 frames
        "min_silence_ms": 96,  # 3 frames
        "pre_roll_ms": 64,  # 2 frames
        "max_utterance_ms": 3200,  # 100 frames
    }
    kwargs.update(overrides)
    return UtteranceSegmenter(**kwargs)  # type: ignore[arg-type]


def _feed(
    segmenter: UtteranceSegmenter,
    probabilities: list[float],
    value: float = 0.5,
) -> list[Utterance]:
    """Push a sequence of frames with given speech probabilities.

    Args:
        segmenter: Segmenter under test.
        probabilities: One probability per frame.
        value: Constant sample value, so frames are distinguishable.

    Returns:
        Utterances emitted during the sequence.
    """
    emitted: list[Utterance] = []
    for index, probability in enumerate(probabilities):
        frame = AudioFrame(
            pcm=np.full(FRAME_SAMPLES, value, dtype=np.float32),
            sample_rate=RATE,
            timestamp_ms=index * FRAME_MS,
        )
        utterance = segmenter.push(frame, probability)
        if utterance is not None:
            emitted.append(utterance)
    return emitted


class TestSilence:
    """Behaviour when nobody is speaking."""

    def test_starts_in_silence(self) -> None:
        assert _make_segmenter().state is SegmenterState.SILENCE

    def test_silence_produces_no_utterances(self) -> None:
        segmenter = _make_segmenter()
        assert _feed(segmenter, [0.0] * 50) == []
        assert segmenter.state is SegmenterState.SILENCE


class TestOnsetDetection:
    """Deciding that speech has started."""

    def test_single_loud_frame_is_not_speech(self) -> None:
        # A keyboard click or door slam must not start an utterance.
        segmenter = _make_segmenter()
        _feed(segmenter, [0.0, 0.9, 0.0, 0.0])

        assert segmenter.stats().rejected_short_bursts == 1
        assert segmenter.state is SegmenterState.SILENCE

    def test_sustained_speech_starts_an_utterance(self) -> None:
        segmenter = _make_segmenter()
        _feed(segmenter, [0.0, 0.9, 0.9, 0.9])

        assert segmenter.is_speaking
        assert segmenter.state is SegmenterState.SPEECH

    def test_onset_requires_min_speech_duration(self) -> None:
        segmenter = _make_segmenter(min_speech_ms=160)  # 5 frames
        _feed(segmenter, [0.9] * 4)
        assert not segmenter.is_speaking

        _feed(segmenter, [0.9])
        assert segmenter.is_speaking


class TestPreRoll:
    """Audio kept from before speech was detected."""

    def test_utterance_includes_pre_roll_audio(self) -> None:
        # Without pre-roll the first syllable is clipped, which is a
        # recognition error rather than a cosmetic issue.
        segmenter = _make_segmenter(pre_roll_ms=64, min_speech_ms=64)
        emitted = _feed(segmenter, [0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0])

        assert len(emitted) == 1
        # 2 pre-roll frames + 3 speech + 3 trailing silence = 8 frames.
        assert emitted[0].pcm.size == 8 * FRAME_SAMPLES

    def test_pre_roll_is_bounded(self) -> None:
        segmenter = _make_segmenter(pre_roll_ms=64, min_speech_ms=64)
        emitted = _feed(segmenter, [0.0] * 40 + [0.9, 0.9, 0.9] + [0.0] * 3)

        assert len(emitted) == 1
        # Long silence must not accumulate: still 2 pre-roll frames.
        assert emitted[0].pcm.size == 8 * FRAME_SAMPLES

    def test_onset_frames_survive_a_small_pre_roll(self) -> None:
        # Regression: frames collected while onset was being *confirmed* were
        # stored in the capped pre-roll ring, so whenever min_speech_ms
        # exceeded pre_roll_ms the ring discarded real speech and the start of
        # every utterance was clipped.
        segmenter = _make_segmenter(min_speech_ms=256, pre_roll_ms=32)  # 8 frames vs 1
        emitted = _feed(segmenter, [0.9] * 10 + [0.0] * 3)

        assert len(emitted) == 1
        # All 10 speech frames plus 3 trailing silence frames must be present.
        assert emitted[0].pcm.size == 13 * FRAME_SAMPLES

    def test_zero_pre_roll_still_keeps_all_speech(self) -> None:
        segmenter = _make_segmenter(min_speech_ms=96, pre_roll_ms=0)  # 3 frames
        emitted = _feed(segmenter, [0.9] * 4 + [0.0] * 3)

        assert emitted[0].pcm.size == 7 * FRAME_SAMPLES

    def test_rejected_burst_audio_becomes_pre_roll(self) -> None:
        # A false alarm's frames are not speech, but they are still recent
        # audio and must remain available as pre-roll for what follows.
        segmenter = _make_segmenter(min_speech_ms=64, pre_roll_ms=320)
        emitted = _feed(segmenter, [0.9, 0.0, 0.0, 0.9, 0.9, 0.9] + [0.0] * 3)

        assert segmenter.stats().rejected_short_bursts == 1
        assert len(emitted) == 1
        assert emitted[0].pcm.size == 9 * FRAME_SAMPLES

    def test_start_time_precedes_detection(self) -> None:
        segmenter = _make_segmenter(pre_roll_ms=64, min_speech_ms=64)
        emitted = _feed(segmenter, [0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0])

        # Detection completes on frame index 3 (t=128 ms); the utterance must
        # be timestamped earlier because pre-roll audio is included.
        assert emitted[0].started_at_ms < 128.0


class TestEndpointDetection:
    """Deciding that speech has stopped."""

    def test_utterance_ends_after_min_silence(self) -> None:
        segmenter = _make_segmenter(min_silence_ms=96)  # 3 frames
        emitted = _feed(segmenter, [0.9] * 5 + [0.0, 0.0])

        assert emitted == []

        emitted = _feed(segmenter, [0.0])
        assert len(emitted) == 1

    def test_brief_pause_does_not_split_an_utterance(self) -> None:
        # People pause between words. Splitting there would produce fragments
        # that translate badly.
        segmenter = _make_segmenter(min_silence_ms=96)
        emitted = _feed(segmenter, [0.9] * 4 + [0.0, 0.0] + [0.9] * 4 + [0.0] * 3)

        assert len(emitted) == 1

    def test_trailing_silence_is_kept(self) -> None:
        # Speech recognisers use trailing silence to decide the final word has
        # ended; trimming it costs accuracy on the last word.
        segmenter = _make_segmenter(min_silence_ms=96, pre_roll_ms=0, min_speech_ms=32)
        emitted = _feed(segmenter, [0.9] * 3 + [0.0] * 3)

        assert emitted[0].pcm.size == 6 * FRAME_SAMPLES

    def test_returns_to_silence_after_emitting(self) -> None:
        segmenter = _make_segmenter()
        _feed(segmenter, [0.9] * 4 + [0.0] * 4)

        assert segmenter.state is SegmenterState.SILENCE
        assert not segmenter.is_speaking


class TestMultipleUtterances:
    """Several sentences in one session."""

    def test_detects_each_sentence_separately(self) -> None:
        segmenter = _make_segmenter()
        pattern = [0.9] * 5 + [0.0] * 5
        emitted = _feed(segmenter, pattern * 3)

        assert len(emitted) == 3
        assert segmenter.stats().utterances_emitted == 3

    def test_utterance_ids_are_unique(self) -> None:
        segmenter = _make_segmenter()
        emitted = _feed(segmenter, ([0.9] * 5 + [0.0] * 5) * 3)

        assert len({utterance.id for utterance in emitted}) == 3


class TestMaximumLength:
    """The guard for a speaker who never pauses."""

    def test_cuts_at_the_configured_limit(self) -> None:
        segmenter = _make_segmenter(max_utterance_ms=320)  # 10 frames
        emitted = _feed(segmenter, [0.9] * 40)

        assert len(emitted) >= 3
        assert segmenter.stats().forced_cuts >= 3

    def test_continues_collecting_after_a_forced_cut(self) -> None:
        # The speaker is still talking, so the next utterance starts at once.
        segmenter = _make_segmenter(max_utterance_ms=320)
        _feed(segmenter, [0.9] * 15)

        assert segmenter.is_speaking


class TestFlush:
    """Ending capture mid-sentence."""

    def test_flush_emits_the_partial_utterance(self) -> None:
        segmenter = _make_segmenter()
        _feed(segmenter, [0.9] * 5)

        utterance = segmenter.flush()
        assert utterance is not None
        assert utterance.pcm.size > 0

    def test_flush_returns_none_when_idle(self) -> None:
        assert _make_segmenter().flush() is None

    def test_flush_resets_state(self) -> None:
        segmenter = _make_segmenter()
        _feed(segmenter, [0.9] * 5)
        segmenter.flush()

        assert segmenter.state is SegmenterState.SILENCE


class TestConfiguration:
    """Parameter validation and metadata."""

    def test_rejects_out_of_range_threshold(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            _make_segmenter(threshold=1.5)

    def test_rejects_silence_longer_than_the_maximum(self) -> None:
        with pytest.raises(ValueError, match="must be smaller than"):
            _make_segmenter(min_silence_ms=5000, max_utterance_ms=1000)

    def test_language_is_attached_to_utterances(self) -> None:
        segmenter = _make_segmenter(language=LanguageCode("ta"))
        emitted = _feed(segmenter, [0.9] * 5 + [0.0] * 4)

        assert emitted[0].language == LanguageCode("ta")

    def test_threshold_boundary_counts_as_speech(self) -> None:
        segmenter = _make_segmenter(threshold=0.5, min_speech_ms=32)
        _feed(segmenter, [0.5, 0.5])

        assert segmenter.is_speaking


class TestStats:
    """Counters exposed for the Performance page."""

    def test_counts_frames_and_utterances(self) -> None:
        segmenter = _make_segmenter()
        _feed(segmenter, [0.9] * 5 + [0.0] * 5)
        stats = segmenter.stats()

        assert stats.frames_processed == 10
        assert stats.utterances_emitted == 1
