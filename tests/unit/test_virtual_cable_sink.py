"""Unit tests for the virtual cable sink.

The PortAudio stream is replaced by a marker object and the driver callback's
core (``_fill``) is driven by hand, so the queueing, jitter gating, drain and
barge-in behaviour are verified deterministically. Opening a real device is
covered by the ``--speak --out`` verification path on the target machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_interpreter.domain.entities import DeviceInfo, SpeechAudio, UtteranceId
from ai_interpreter.domain.errors import AudioOutputError
from ai_interpreter.domain.value_objects import DeviceKind, LanguageCode, SampleRate
from ai_interpreter.infrastructure.audio.playback.virtual_cable import VirtualCableSink

pytestmark = pytest.mark.unit

RATE_48K = SampleRate(48000)
ENGLISH = LanguageCode("en")

_DEVICE = DeviceInfo(
    index=5,
    name="CABLE Input (VB-Audio Virtual Cable)",
    kind=DeviceKind.OUTPUT,
    max_channels=2,
    default_sample_rate=48000.0,
    host_api="Windows WASAPI",
)


def _sink(jitter_ms: int = 60) -> VirtualCableSink:
    """Build a sink marked open without touching a device.

    Args:
        jitter_ms: Jitter buffer size.

    Returns:
        The sink, ready for ``write``/``_fill``.
    """
    sink = VirtualCableSink(_DEVICE, RATE_48K, jitter_buffer_ms=jitter_ms)
    sink._stream = object()  # marks the sink open; _fill is driven manually
    return sink


def _chunk(
    samples: int,
    value: float = 0.5,
    rate: SampleRate = RATE_48K,
    *,
    is_last: bool = False,
) -> SpeechAudio:
    """Build an audio chunk.

    Args:
        samples: Sample count.
        value: Constant sample value, so output provenance is checkable.
        rate: Chunk sample rate.
        is_last: Final-chunk flag.

    Returns:
        The chunk.
    """
    return SpeechAudio(
        utterance_id=UtteranceId("tts"),
        pcm=np.full(samples, value, dtype=np.float32),
        sample_rate=rate,
        language=ENGLISH,
        is_last=is_last,
    )


def _drain(sink: VirtualCableSink, block: int = 960) -> np.ndarray:
    """Pull blocks until the sink reports drained, returning played audio.

    Args:
        sink: Sink under test.
        block: Samples per pull.

    Returns:
        Concatenated non-silent prefix of everything played.
    """
    played: list[np.ndarray] = []
    for _ in range(1000):
        out = np.empty(block, dtype=np.float32)
        sink._fill(out)
        played.append(out.copy())
        if sink.pending_ms == 0.0:
            break
    return np.concatenate(played)


class TestWriteAndGate:
    """Queueing and the jitter gate."""

    def test_write_before_open_is_an_error(self) -> None:
        sink = VirtualCableSink(_DEVICE, RATE_48K)
        with pytest.raises(AudioOutputError, match="closed sink"):
            sink.write(_chunk(100))

    def test_playback_is_gated_until_the_buffer_fills(self) -> None:
        # 60 ms at 48 kHz = 2880 samples. A 1000-sample chunk must not start
        # playback: the first block out is silence.
        sink = _sink(jitter_ms=60)
        sink.write(_chunk(1000))

        out = np.empty(960, dtype=np.float32)
        sink._fill(out)
        assert np.all(out == 0.0)
        assert sink.pending_ms > 0.0

    def test_playback_starts_once_the_buffer_is_full(self) -> None:
        sink = _sink(jitter_ms=60)
        sink.write(_chunk(3000))  # above the 2880-sample threshold

        out = np.empty(960, dtype=np.float32)
        sink._fill(out)
        assert np.all(out == 0.5)

    def test_the_final_chunk_releases_the_gate(self) -> None:
        # An utterance shorter than the jitter buffer must still play.
        sink = _sink(jitter_ms=60)
        sink.write(_chunk(1000, is_last=True))

        out = np.empty(960, dtype=np.float32)
        sink._fill(out)
        assert np.all(out == 0.5)

    def test_empty_final_chunk_releases_a_pending_gate(self) -> None:
        sink = _sink(jitter_ms=60)
        sink.write(_chunk(1000))
        sink.write(
            SpeechAudio(
                utterance_id=UtteranceId("tts"),
                pcm=np.empty(0, dtype=np.float32),
                sample_rate=RATE_48K,
                language=ENGLISH,
                is_last=True,
            )
        )

        out = np.empty(960, dtype=np.float32)
        sink._fill(out)
        assert np.all(out == 0.5)


class TestFillAndDrain:
    """The driver-side consumption path."""

    def test_all_written_audio_is_played_in_order(self) -> None:
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(1000, value=0.25))
        sink.write(_chunk(500, value=0.75, is_last=True))

        played = _drain(sink)
        assert np.all(played[:1000] == 0.25)
        assert np.all(played[1000:1500] == 0.75)
        assert np.all(played[1500:] == 0.0)

    def test_silence_after_the_queue_empties(self) -> None:
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(100, is_last=True))
        _drain(sink)

        out = np.empty(960, dtype=np.float32)
        sink._fill(out)
        assert np.all(out == 0.0)

    def test_next_utterance_prebuffers_again(self) -> None:
        # After a drain the gate must re-arm: a small next chunk waits.
        sink = _sink(jitter_ms=60)
        sink.write(_chunk(3000))
        _drain(sink)

        sink.write(_chunk(500))
        out = np.empty(960, dtype=np.float32)
        sink._fill(out)
        assert np.all(out == 0.0)

    def test_partial_block_is_zero_padded(self) -> None:
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(100, is_last=True))

        out = np.empty(960, dtype=np.float32)
        sink._fill(out)
        assert np.all(out[:100] == 0.5)
        assert np.all(out[100:] == 0.0)

    def test_flush_returns_once_drained(self) -> None:
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(100, is_last=True))
        _drain(sink)

        sink.flush(timeout=0.5)  # must not hang

    def test_flush_forces_a_gated_utterance_to_play(self) -> None:
        # flush() on a sub-buffer utterance must release the gate itself.
        sink = _sink(jitter_ms=60)
        sink.write(_chunk(500))

        out = np.empty(960, dtype=np.float32)
        # A concurrent flush would set playing; simulate its first step:
        with sink._lock:
            assert not sink._playing
        import threading

        flusher = threading.Thread(target=sink.flush, kwargs={"timeout": 2.0})
        flusher.start()
        for _ in range(50):
            sink._fill(out)
            if sink.pending_ms == 0.0:
                break
        flusher.join(timeout=3.0)
        assert not flusher.is_alive()


class TestClear:
    """Barge-in."""

    def test_clear_discards_everything_queued(self) -> None:
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(5000))
        sink.clear()

        assert sink.pending_ms == 0.0
        out = np.empty(960, dtype=np.float32)
        sink._fill(out)
        assert np.all(out == 0.0)

    def test_writing_after_clear_works(self) -> None:
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(5000, value=0.2))
        sink.clear()
        sink.write(_chunk(1000, value=0.9, is_last=True))

        played = _drain(sink)
        assert np.all(played[:1000] == 0.9)


class TestResampling:
    """Chunks arriving at a voice's native rate."""

    def test_chunks_are_resampled_to_the_device_rate(self) -> None:
        # A 16 kHz chunk (the MMS voice) into a 48 kHz device triples.
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(1600, rate=SampleRate(16000), is_last=True))

        assert sink.pending_ms == pytest.approx(100.0, abs=15.0)

    def test_rate_changes_between_utterances_are_handled(self) -> None:
        # Language switch: 22.05 kHz Piper then 16 kHz MMS.
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(2205, rate=SampleRate(22050), is_last=True))
        _drain(sink)
        sink.write(_chunk(1600, rate=SampleRate(16000), is_last=True))

        assert sink.pending_ms == pytest.approx(100.0, abs=15.0)

    def test_device_rate_chunks_pass_through_untouched(self) -> None:
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(480, value=0.5, is_last=True))

        played = _drain(sink)
        assert np.all(played[:480] == 0.5)


class TestLifecycle:
    """Open and close behaviour."""

    def test_close_is_idempotent(self) -> None:
        sink = _sink()
        sink.close()
        sink.close()

    def test_is_open_reflects_state(self) -> None:
        sink = VirtualCableSink(_DEVICE, RATE_48K)
        assert not sink.is_open
        sink._stream = object()
        assert sink.is_open

    def test_statistics_accumulate(self) -> None:
        sink = _sink(jitter_ms=0)
        sink.write(_chunk(1000, is_last=True))
        sink.write(_chunk(500, is_last=True))
        _drain(sink)

        assert sink.chunks_written == 2
