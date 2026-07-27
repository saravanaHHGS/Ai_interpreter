"""Unit tests for the audio callback buffer and frame assembler."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ai_interpreter.infrastructure.audio.buffers import AudioBlockBuffer, FrameAssembler

pytestmark = pytest.mark.unit


def _block(value: float, size: int = 4) -> np.ndarray:
    """Build a constant-valued block.

    Args:
        value: Sample value.
        size: Block length.

    Returns:
        A float32 array.
    """
    return np.full(size, value, dtype=np.float32)


class TestAudioBlockBuffer:
    """Hand-off between the driver callback and the worker thread."""

    def test_reads_blocks_in_order(self) -> None:
        buffer = AudioBlockBuffer(max_blocks=4)
        buffer.write(_block(1.0))
        buffer.write(_block(2.0))

        assert buffer.read(timeout=0.1)[0] == pytest.approx(1.0)
        assert buffer.read(timeout=0.1)[0] == pytest.approx(2.0)

    def test_returns_none_on_timeout(self) -> None:
        buffer = AudioBlockBuffer(max_blocks=2)
        start = time.monotonic()
        assert buffer.read(timeout=0.05) is None
        assert time.monotonic() - start >= 0.04

    def test_drops_oldest_when_full(self) -> None:
        # Stale audio is worthless in a live conversation, so the oldest goes.
        buffer = AudioBlockBuffer(max_blocks=2)
        for value in (1.0, 2.0, 3.0):
            buffer.write(_block(value))

        assert buffer.pending_blocks == 2
        assert buffer.dropped_blocks == 1
        assert buffer.read(timeout=0.1)[0] == pytest.approx(2.0)
        assert buffer.read(timeout=0.1)[0] == pytest.approx(3.0)

    def test_counts_writes_and_drops_separately(self) -> None:
        buffer = AudioBlockBuffer(max_blocks=1)
        for _ in range(5):
            buffer.write(_block(1.0))

        assert buffer.written_blocks == 5
        assert buffer.dropped_blocks == 4

    def test_clear_discards_pending(self) -> None:
        buffer = AudioBlockBuffer(max_blocks=4)
        buffer.write(_block(1.0))
        buffer.clear()

        assert buffer.pending_blocks == 0
        assert buffer.read(timeout=0.01) is None

    def test_rejects_zero_capacity(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            AudioBlockBuffer(max_blocks=0)

    def test_reader_wakes_when_a_block_arrives(self) -> None:
        buffer = AudioBlockBuffer(max_blocks=4)
        received: list[np.ndarray] = []

        def consume() -> None:
            block = buffer.read(timeout=2.0)
            if block is not None:
                received.append(block)

        reader = threading.Thread(target=consume)
        reader.start()
        time.sleep(0.05)
        buffer.write(_block(7.0))
        reader.join(timeout=2.0)

        assert len(received) == 1
        assert received[0][0] == pytest.approx(7.0)

    def test_survives_concurrent_producer_and_consumer(self) -> None:
        # Mirrors the real arrangement: driver thread writing, worker reading.
        buffer = AudioBlockBuffer(max_blocks=256)
        total = 500
        received: list[float] = []
        done = threading.Event()

        def produce() -> None:
            for index in range(total):
                buffer.write(_block(float(index)))
            done.set()

        def consume() -> None:
            while not done.is_set() or buffer.pending_blocks:
                block = buffer.read(timeout=0.1)
                if block is not None:
                    received.append(float(block[0]))

        producer = threading.Thread(target=produce)
        consumer = threading.Thread(target=consume)
        consumer.start()
        producer.start()
        producer.join(timeout=5.0)
        consumer.join(timeout=5.0)

        assert received == sorted(received), "blocks arrived out of order"
        assert len(received) + buffer.dropped_blocks == total


class TestFrameAssembler:
    """Re-chunking a variable-length stream into fixed frames."""

    def test_emits_nothing_until_a_frame_is_complete(self) -> None:
        assembler = FrameAssembler(frame_samples=512)
        assert assembler.push(np.zeros(300, dtype=np.float32)) == []
        assert assembler.pending_samples == 300

    def test_emits_one_frame_when_enough_arrives(self) -> None:
        assembler = FrameAssembler(frame_samples=512)
        assembler.push(np.zeros(300, dtype=np.float32))
        frames = assembler.push(np.zeros(300, dtype=np.float32))

        assert len(frames) == 1
        assert frames[0].size == 512
        assert assembler.pending_samples == 88

    def test_emits_multiple_frames_from_one_push(self) -> None:
        assembler = FrameAssembler(frame_samples=512)
        frames = assembler.push(np.zeros(512 * 3 + 10, dtype=np.float32))

        assert len(frames) == 3
        assert assembler.pending_samples == 10

    def test_preserves_sample_order_across_pushes(self) -> None:
        assembler = FrameAssembler(frame_samples=4)
        assembler.push(np.array([1, 2, 3], dtype=np.float32))
        frames = assembler.push(np.array([4, 5, 6, 7, 8], dtype=np.float32))

        assert len(frames) == 2
        np.testing.assert_array_equal(frames[0], [1, 2, 3, 4])
        np.testing.assert_array_equal(frames[1], [5, 6, 7, 8])

    def test_flush_zero_pads_the_remainder(self) -> None:
        assembler = FrameAssembler(frame_samples=4)
        assembler.push(np.array([1, 2], dtype=np.float32))
        tail = assembler.flush()

        assert tail is not None
        np.testing.assert_array_equal(tail, [1, 2, 0, 0])
        assert assembler.pending_samples == 0

    def test_flush_returns_none_when_empty(self) -> None:
        assert FrameAssembler(frame_samples=4).flush() is None

    def test_reset_discards_partial_data(self) -> None:
        assembler = FrameAssembler(frame_samples=4)
        assembler.push(np.array([1, 2], dtype=np.float32))
        assembler.reset()

        assert assembler.pending_samples == 0
        assert assembler.flush() is None

    def test_rejects_zero_frame_size(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            FrameAssembler(frame_samples=0)
