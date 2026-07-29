"""Headless tests for the desktop UI (Phase 8).

Qt's ``offscreen`` platform renders no windows, so the whole view and the
thread-marshalling bridge are exercised in an ordinary test process - the
same separation that keeps model code out of the window makes the window
testable without models.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ai_interpreter.application.pipeline.interpretation import UtteranceTiming
from ai_interpreter.application.services.utterance_segmenter import SegmenterState
from ai_interpreter.domain.entities import Transcript, Translation, UtteranceId
from ai_interpreter.domain.value_objects import (
    Confidence,
    LanguageCode,
    LanguagePair,
)
from ai_interpreter.presentation.ui.bridge import PipelineBridge
from ai_interpreter.presentation.ui.main_window import MainWindow

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    """One offscreen QApplication for the whole module."""
    application = QApplication.instance() or QApplication([])
    assert isinstance(application, QApplication)
    yield application


def _transcript(text: str, language: str = "ta") -> Transcript:
    return Transcript(
        utterance_id=UtteranceId("u1"),
        text=text,
        language=LanguageCode(language),
        confidence=Confidence(0.9),
        is_final=True,
    )


def _translation(text: str, *, cached: bool = False) -> Translation:
    return Translation(
        utterance_id=UtteranceId("u1"),
        source_text="src",
        translated_text=text,
        pair=LanguagePair.of("ta", "en"),
        from_cache=cached,
    )


class TestBridge:
    """Pipeline callbacks become Qt signals."""

    def test_transcript_callback_emits_language_and_text(self, qt_app: QApplication) -> None:
        bridge = PipelineBridge()
        received: list[tuple[str, str]] = []
        bridge.transcript_received.connect(lambda lang, text: received.append((lang, text)))

        bridge.events().on_transcript(_transcript("வணக்கம்"))  # type: ignore[misc]
        qt_app.processEvents()

        assert received == [("ta", "வணக்கம்")]

    def test_empty_transcript_is_not_forwarded(self, qt_app: QApplication) -> None:
        bridge = PipelineBridge()
        received: list[tuple[str, str]] = []
        bridge.transcript_received.connect(lambda lang, text: received.append((lang, text)))

        bridge.events().on_transcript(_transcript(""))  # type: ignore[misc]
        qt_app.processEvents()

        assert received == []

    def test_partial_transcripts_emit_their_text(self, qt_app: QApplication) -> None:
        bridge = PipelineBridge()
        received: list[str] = []
        bridge.partial_received.connect(received.append)

        bridge.events().on_partial(_transcript("வணக்கம் என்", "ta"))  # type: ignore[misc]
        qt_app.processEvents()

        assert received == ["வணக்கம் என்"]

    def test_translation_carries_the_cache_flag(self, qt_app: QApplication) -> None:
        bridge = PipelineBridge()
        received: list[tuple[str, bool]] = []
        bridge.translation_received.connect(lambda text, cached: received.append((text, cached)))

        bridge.events().on_translation(_translation("hello", cached=True))  # type: ignore[misc]
        qt_app.processEvents()

        assert received == [("hello", True)]

    def test_timing_is_unpacked_to_floats(self, qt_app: QApplication) -> None:
        bridge = PipelineBridge()
        received: list[tuple[float, float, float, float]] = []
        bridge.timing_received.connect(lambda *values: received.append(values))

        timing = UtteranceTiming(
            utterance_id="u1",
            audio_ms=2000.0,
            stt_ms=1200.0,
            mt_ms=300.0,
            tts_first_chunk_ms=400.0,
            eou_to_first_audio_ms=1900.0,
            total_ms=2500.0,
        )
        bridge.events().on_timing(timing)  # type: ignore[misc]
        qt_app.processEvents()

        assert received == [(1900.0, 1200.0, 300.0, 400.0)]

    def test_state_is_lowercased(self, qt_app: QApplication) -> None:
        bridge = PipelineBridge()
        received: list[str] = []
        bridge.state_changed.connect(received.append)

        bridge.events().on_state(SegmenterState.SPEECH)  # type: ignore[misc]
        qt_app.processEvents()

        assert received == ["speech"]

    def test_emission_from_a_worker_thread_reaches_the_main_thread(
        self, qt_app: QApplication
    ) -> None:
        # The property the bridge exists for: pipeline callbacks run on
        # worker threads, slots must run on the UI thread.
        bridge = PipelineBridge()
        received: list[str] = []
        bridge.transcript_received.connect(lambda lang, text: received.append(text))

        worker = threading.Thread(
            target=lambda: bridge.events().on_transcript(_transcript("from-worker"))  # type: ignore[misc]
        )
        worker.start()
        worker.join()

        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            qt_app.processEvents()

        assert received == ["from-worker"]


class TestMainWindow:
    """The view in isolation."""

    def _window(self) -> MainWindow:
        return MainWindow(
            direction="Tamil -> English",
            input_devices=["Microphone (USB2.0 Device)", "Other Mic"],
            output_devices=["Speakers", "CABLE Input (VB-Audio Virtual Cable)"],
        )

    def test_captions_show_transcripts_and_translations(self, qt_app: QApplication) -> None:
        window = self._window()
        window.append_caption("ta", "வணக்கம்")
        window.append_translation("Hello", False)
        window.append_translation("Hello", True)

        text = window.captions.toPlainText()
        assert "[ta] வணக்கம்" in text
        assert "-> Hello" in text
        assert "(cached)" in text

    def test_start_emits_the_selected_devices(self, qt_app: QApplication) -> None:
        window = self._window()
        window.select_devices("Other Mic", "CABLE Input (VB-Audio Virtual Cable)")
        requested: list[tuple[str, str]] = []
        window.start_requested.connect(lambda mic, out: requested.append((mic, out)))

        window.start_button.click()

        assert requested == [("Other Mic", "CABLE Input (VB-Audio Virtual Cable)")]

    def test_stop_is_emitted_while_running(self, qt_app: QApplication) -> None:
        window = self._window()
        stops: list[bool] = []
        window.stop_requested.connect(lambda: stops.append(True))

        window.set_running(True)
        window.start_button.click()

        assert stops == [True]
        assert window.start_button.text() == "Stop"

    def test_busy_window_ignores_clicks(self, qt_app: QApplication) -> None:
        window = self._window()
        requested: list[tuple[str, str]] = []
        window.start_requested.connect(lambda mic, out: requested.append((mic, out)))

        window.set_busy("loading models...")
        window._on_button()  # the button itself is disabled; belt and braces

        assert requested == []

    def test_running_disables_the_device_pickers(self, qt_app: QApplication) -> None:
        window = self._window()
        window.set_running(True)
        assert not window.input_box.isEnabled()
        assert not window.output_box.isEnabled()

        window.set_running(False)
        assert window.input_box.isEnabled()
        assert window.start_button.text() == "Start"

    def test_partial_line_shows_and_final_caption_clears_it(self, qt_app: QApplication) -> None:
        window = self._window()
        window.show_partial("வணக்கம் என்")
        assert "வணக்கம்" in window.partial_label.text()

        window.append_caption("ta", "வணக்கம் என் பெயர்")
        assert window.partial_label.text() == ""

    def test_timing_updates_the_latency_readout(self, qt_app: QApplication) -> None:
        window = self._window()
        window.show_timing(1900.0, 1200.0, 300.0, 400.0)
        window.show_timing(1100.0, 800.0, 200.0, 100.0)

        text = window.latency_label.text()
        assert "latency 1.1 s" in text
        assert "mean 1.5" in text

    def test_speech_state_only_shows_while_running(self, qt_app: QApplication) -> None:
        window = self._window()
        window.set_status("stopped")
        window.set_speech_state("speech")
        assert window.state_label.text() == "stopped"

        window.set_running(True)
        window.set_speech_state("speech")
        assert window.state_label.text() == "hearing speech"
