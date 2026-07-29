"""Marshalling pipeline events onto the Qt thread.

The pipeline invokes its ``PipelineEvents`` callbacks on *worker* threads
(the interpretation worker, the capture worker). Qt widgets may only be
touched from the thread that owns them, so nothing in those callbacks may
go near a widget.

Qt's own signal machinery is the bridge: emitting a signal from a foreign
thread automatically queues the delivery onto the receiver's thread (a
``QueuedConnection``). So each callback does exactly one thing - emit a
signal carrying plain values - and every connected slot runs safely on the
UI thread. Domain objects are unpacked to primitives at the boundary, which
also keeps the UI layer free of pipeline imports beyond this module.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ai_interpreter.application.pipeline.interpretation import PipelineEvents, UtteranceTiming
from ai_interpreter.application.services.utterance_segmenter import SegmenterState
from ai_interpreter.domain.entities import Transcript, Translation

__all__ = ["PipelineBridge"]


class PipelineBridge(QObject):
    """Turns pipeline callbacks into Qt signals.

    Create it on the UI thread (its home thread is what queued deliveries
    target), hand :meth:`events` to the pipeline, and connect the signals
    to widget slots.
    """

    transcript_received = Signal(str, str)  # language code, text
    partial_received = Signal(str)  # interim text, grows as words commit
    translation_received = Signal(str, bool)  # text, from_cache
    timing_received = Signal(float, float, float, float)  # eou->audio, stt, mt, tts (ms)
    error_occurred = Signal(str, str)  # stage, message
    state_changed = Signal(str)  # SegmenterState name, lowercase

    def events(self) -> PipelineEvents:
        """Build the callback set to hand to the pipeline.

        Returns:
            Callbacks that emit this bridge's signals.
        """
        return PipelineEvents(
            on_transcript=self._on_transcript,
            on_partial=self._on_partial,
            on_translation=self._on_translation,
            on_timing=self._on_timing,
            on_error=self._on_error,
            on_state=self._on_state,
        )

    # -- callbacks, invoked on pipeline worker threads ---------------------
    def _on_transcript(self, transcript: Transcript) -> None:
        if not transcript.is_empty:
            self.transcript_received.emit(transcript.language.code, transcript.text)

    def _on_partial(self, transcript: Transcript) -> None:
        if not transcript.is_empty:
            self.partial_received.emit(transcript.text)

    def _on_translation(self, translation: Translation) -> None:
        if not translation.is_empty:
            self.translation_received.emit(translation.translated_text, translation.from_cache)

    def _on_timing(self, timing: UtteranceTiming) -> None:
        self.timing_received.emit(
            timing.eou_to_first_audio_ms,
            timing.stt_ms,
            timing.mt_ms,
            timing.tts_first_chunk_ms,
        )

    def _on_error(self, stage: str, error: Exception) -> None:
        self.error_occurred.emit(stage, str(error))

    def _on_state(self, state: SegmenterState) -> None:
        self.state_changed.emit(state.name.lower())
