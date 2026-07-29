"""Fixtures for integration tests that use the real models and recording.

These tests run the actual neural networks against the checked-out model
cache and the evaluation recording captured in a live session. Neither is
committed to the repository (models are ~1 GB and re-downloadable; the
recording is the developer's own voice), so every fixture skips cleanly
when its prerequisite is absent instead of failing.

To recreate the evaluation recording: ``.\\run.ps1 --record 30`` speaking
natural mixed Tamil-English sentences, then update ``RECORDING`` below.
"""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ai_interpreter.app.container import Container

REPO_ROOT = Path(__file__).resolve().parents[2]

# The Tanglish evaluation recording: six utterances of natural mixed speech
# ("VALD assessment முடிஞ்சிடுச்சு", "matching மட்டும் pending ல இருக்கு",
# pure Tamil, and one fully English sentence), ground-truthed by the speaker.
RECORDING = REPO_ROOT / "recordings" / "20260729-120325-processed-16000.wav"

# (start_s, duration_s) per utterance, as reported by --record.
UTTERANCE_SPANS = [
    (2.53, 1.09),
    (3.62, 2.37),
    (7.68, 2.08),
    (11.71, 2.05),
    (16.13, 3.49),
    (20.99, 1.82),
]


@pytest.fixture(scope="session")
def real_container() -> Iterator[Container]:
    """A container over the REAL project root, models included.

    Session-scoped: model loading costs seconds and every component is
    cached inside the container anyway (Phase 10).
    """
    if not (REPO_ROOT / "models").is_dir():
        pytest.skip("model cache not present; run any CLI command once to download")
    container = Container.build(root=REPO_ROOT)
    yield container
    container.shutdown()


@pytest.fixture(scope="session")
def recording_path() -> Path:
    """Path to the evaluation recording, skipping when it is absent."""
    if not RECORDING.is_file():
        pytest.skip(f"evaluation recording not present: {RECORDING.name}")
    return RECORDING


@pytest.fixture(scope="session")
def recording_pcm() -> np.ndarray:
    """The evaluation recording as float32 mono 16 kHz samples."""
    if not RECORDING.is_file():
        pytest.skip(f"evaluation recording not present: {RECORDING.name}")
    with wave.open(str(RECORDING), "rb") as reader:
        assert reader.getframerate() == 16000
        raw = reader.readframes(reader.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


@pytest.fixture(scope="session")
def slice_utterance(recording_pcm: np.ndarray):  # type: ignore[no-untyped-def]
    """A cutter for single utterances, with pre/post margins.

    Returns:
        ``cut(index)`` returning the samples of the 1-based utterance
        number, matching the --record report.
    """

    def cut(index: int) -> np.ndarray:
        start, duration = UTTERANCE_SPANS[index - 1]
        lo = max(0, int((start - 0.35) * 16000))
        hi = min(recording_pcm.size, int((start + duration + 0.30) * 16000))
        return np.ascontiguousarray(recording_pcm[lo:hi])

    return cut
