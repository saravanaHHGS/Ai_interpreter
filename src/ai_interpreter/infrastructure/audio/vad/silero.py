"""Silero voice activity detection via ONNX Runtime.

Silero VAD is a small recurrent network that outputs, per 32 ms frame, the
probability that the frame contains human speech. It is the right tool here
for reasons an energy threshold cannot match: it distinguishes speech from
keyboard noise, air conditioning and music, and it holds up at low signal
levels where a threshold either clips the start of words or triggers on hum.

Measured on the development machine (Intel i5-7200U, single ONNX thread):
**0.5 ms per 32 ms frame, about 1.6 % of one core.**

Two constraints are inherent to the model and are enforced rather than
worked around:

* Input must be **exactly 512 samples at 16 kHz**. Silero v5 has a fixed
  input size; other lengths produce meaningless output rather than an error.
  :class:`~ai_interpreter.infrastructure.audio.buffers.FrameAssembler`
  guarantees the size, and this class re-checks it.
* The network is **recurrent**, so it carries hidden state between frames.
  That state must be reset between utterances - leaving one speaker's state
  in place degrades the first frames of the next.

ONNX Runtime is used rather than PyTorch because it loads in milliseconds
instead of seconds, adds no multi-hundred-megabyte dependency, and is already
required for text-to-speech in Phase 6. One runtime for both is one fewer
native library to go wrong - which matters on a machine with Windows Smart App
Control enforced, where unsigned native libraries are blocked outright.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

from ai_interpreter.domain.entities import AudioFrame
from ai_interpreter.domain.errors import ModelLoadError
from ai_interpreter.domain.value_objects import SampleRate

__all__ = ["SileroVad"]

logger = logging.getLogger(__name__)

# Fixed by the model architecture. Not configurable.
_REQUIRED_SAMPLE_RATE: Final[int] = 16000
_REQUIRED_FRAME_SAMPLES: Final[int] = 512

# Recurrent state shape: (2 layers, batch, 128 hidden units).
_STATE_SHAPE: Final[tuple[int, int, int]] = (2, 1, 128)


class SileroVad:
    """Neural voice activity detector, satisfying the ``VoiceActivityDetector`` port.

    Args:
        model_path: Path to ``silero_vad`` in ONNX format.
        num_threads: ONNX Runtime thread count. One is correct: the model is
            far too small to parallelise usefully, and extra threads take
            cores away from speech recognition on a 2-core machine.

    Raises:
        ModelLoadError: If the model cannot be loaded.
    """

    def __init__(self, model_path: Path, num_threads: int = 1) -> None:
        self._model_path = model_path
        self._session: Any = None
        self._num_threads = num_threads
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._sample_rate_input = np.array(_REQUIRED_SAMPLE_RATE, dtype=np.int64)
        self._frames_scored = 0

    # -- port interface ----------------------------------------------------
    @property
    def required_frame_samples(self) -> int:
        """Exact samples per call: 512, fixed by the model."""
        return _REQUIRED_FRAME_SAMPLES

    @property
    def sample_rate(self) -> SampleRate:
        """Required rate: 16 kHz, fixed by the model."""
        return SampleRate(_REQUIRED_SAMPLE_RATE)

    @property
    def frames_scored(self) -> int:
        """Frames processed since the detector was created."""
        return self._frames_scored

    def warmup(self) -> None:
        """Load the model and run one throwaway inference.

        The first inference allocates buffers and is several times slower than
        the rest. Paying that during startup rather than on the user's first
        word is the difference between a clipped first syllable and a clean one.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        self._ensure_session()
        silence = np.zeros(_REQUIRED_FRAME_SAMPLES, dtype=np.float32)
        self._infer(silence)
        self.reset()
        logger.debug("Silero VAD warmed up from %s", self._model_path)

    def speech_probability(self, frame: AudioFrame) -> float:
        """Score a frame for speech.

        Args:
            frame: Exactly 512 samples at 16 kHz.

        Returns:
            Probability in ``[0.0, 1.0]``.

        Raises:
            ValueError: If the frame size or sample rate is wrong.
            ModelLoadError: If the model cannot be loaded.
        """
        if frame.sample_rate.hz != _REQUIRED_SAMPLE_RATE:
            msg = f"Silero VAD requires {_REQUIRED_SAMPLE_RATE} Hz, got {frame.sample_rate.hz} Hz"
            raise ValueError(msg)
        if frame.pcm.size != _REQUIRED_FRAME_SAMPLES:
            msg = (
                f"Silero VAD requires exactly {_REQUIRED_FRAME_SAMPLES} samples per frame, "
                f"got {frame.pcm.size}"
            )
            raise ValueError(msg)

        self._ensure_session()
        return self._infer(frame.pcm)

    def reset(self) -> None:
        """Clear the recurrent state between utterances."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)

    def close(self) -> None:
        """Release the model. Safe to call more than once."""
        self._session = None

    # -- internals ---------------------------------------------------------
    def _ensure_session(self) -> None:
        """Create the inference session on first use.

        Raises:
            ModelLoadError: If the model file is missing or unreadable.
        """
        if self._session is not None:
            return

        if not self._model_path.is_file():
            msg = f"Silero VAD model not found at {self._model_path}"
            raise ModelLoadError(msg)

        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = self._num_threads
            options.inter_op_num_threads = self._num_threads
            # The model is tiny; graph optimisation costs more than it saves.
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            self._session = ort.InferenceSession(
                str(self._model_path),
                options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            msg = f"Could not load the Silero VAD model from {self._model_path}: {exc}"
            raise ModelLoadError(msg) from exc

        logger.info("Silero VAD loaded from %s (%d thread)", self._model_path, self._num_threads)

    def _infer(self, samples: np.ndarray) -> float:
        """Run one inference step, advancing the recurrent state.

        Args:
            samples: Exactly 512 float32 samples.

        Returns:
            Speech probability.
        """
        assert self._session is not None
        outputs = self._session.run(
            None,
            {
                "input": samples.reshape(1, -1),
                "state": self._state,
                "sr": self._sample_rate_input,
            },
        )
        probability, self._state = outputs[0], outputs[1]
        self._frames_scored += 1
        return float(probability[0][0])
