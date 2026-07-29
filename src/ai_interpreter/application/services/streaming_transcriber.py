"""Feeding one utterance's frames into a streaming recogniser, live.

The offline path waits for the segmenter to finish an utterance and only
then decodes it - for a linear-cost model like the IndicConformer that means
the whole decode happens *after* the speaker stops. This helper moves that
work into the speech itself: frames are pushed as they arrive, the
recogniser's chunked-commitment algorithm decodes and commits words while
the speaker is still talking, and at end-of-utterance only the uncommitted
tail remains to decode.

One instance serves exactly one utterance, on its own daemon thread. The
capture thread pushes frames and never blocks on decoding; the pipeline
worker collects the final transcript; interim transcripts surface through
``on_partial`` for live captions.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from queue import SimpleQueue

from ai_interpreter.domain.entities import AudioFrame, Transcript
from ai_interpreter.domain.ports import StreamingSpeechRecognizer
from ai_interpreter.domain.value_objects import LanguageCode

__all__ = ["UtteranceStreamer"]

logger = logging.getLogger(__name__)


class UtteranceStreamer:
    """Streams one utterance through a recogniser on a dedicated thread.

    Args:
        recognizer: The streaming recogniser (chunked-commitment CTC).
        language: Language being spoken.
        on_partial: Receives interim transcripts as words are committed,
            invoked on the streamer's thread. Keep it cheap.
    """

    def __init__(
        self,
        recognizer: StreamingSpeechRecognizer,
        language: LanguageCode,
        on_partial: Callable[[Transcript], None] | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._language = language
        self._on_partial = on_partial
        self._queue: SimpleQueue[AudioFrame | None] = SimpleQueue()
        self._final: Transcript | None = None
        self._error: Exception | None = None
        self._done = threading.Event()
        self._finished = False
        self._thread = threading.Thread(target=self._run, name="stt-stream", daemon=True)
        self._thread.start()

    @property
    def error(self) -> Exception | None:
        """The exception that stopped decoding, if any."""
        return self._error

    def push(self, frame: AudioFrame) -> None:
        """Queue one frame. Never blocks; called on the capture thread.

        Args:
            frame: The next frame of the utterance.
        """
        if not self._finished:
            self._queue.put(frame)

    def finish(self) -> None:
        """Signal end-of-utterance; the tail decode begins. Idempotent."""
        if not self._finished:
            self._finished = True
            self._queue.put(None)

    def result(self, timeout: float | None = None) -> Transcript | None:
        """Wait for the final transcript.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            The final transcript, or ``None`` on timeout or decode failure -
            the caller falls back to a whole-utterance decode either way, so
            a streaming problem can never lose an utterance.
        """
        if not self._done.wait(timeout):
            logger.warning("Streaming decode did not finish in %.1f s", timeout or 0.0)
            return None
        return self._final

    # -- worker -------------------------------------------------------------
    def _frames(self) -> Iterator[AudioFrame]:
        """Yield queued frames until the end-of-utterance sentinel."""
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            yield frame

    def _run(self) -> None:
        """Consume the stream, keeping the final transcript."""
        try:
            for transcript in self._recognizer.transcribe_stream(self._frames(), self._language):
                if transcript.is_final:
                    self._final = transcript
                elif self._on_partial is not None and transcript.text:
                    self._on_partial(transcript)
        except Exception as exc:
            self._error = exc
            logger.warning("Streaming decode failed; falling back to offline: %s", exc)
        finally:
            self._done.set()
