"""The main window: devices, start/stop, live captions, latency.

Deliberately a *view*: it owns widgets and formatting and nothing else.
It knows the pipeline only through plain-value slots (connected to
:class:`~ai_interpreter.presentation.ui.bridge.PipelineBridge`) and speaks
back only through two signals - ``start_requested`` and ``stop_requested``.
That separation is what lets the whole window be exercised headlessly in
unit tests with Qt's offscreen platform, no models or audio involved.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["MainWindow"]

# Keep the caption view bounded; a day-long meeting must not grow memory.
_MAX_CAPTION_BLOCKS = 500


class MainWindow(QMainWindow):
    """Single-window interface for live interpretation.

    Args:
        direction: Human-readable direction, e.g. ``"Tamil -> English"``.
        input_devices: Microphone names for the input picker.
        output_devices: Playback names for the output picker.
    """

    start_requested = Signal(str, str)  # input device name, output device name
    stop_requested = Signal()

    def __init__(
        self,
        direction: str,
        input_devices: list[str],
        output_devices: list[str],
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"AI Interpreter - {direction}")
        self.resize(760, 540)

        self._running = False
        self._busy = False
        self._latencies: list[float] = []

        # -- top bar: devices and the start/stop button --------------------
        self.input_box = QComboBox()
        self.input_box.addItems(input_devices)
        self.output_box = QComboBox()
        self.output_box.addItems(output_devices)

        self.start_button = QPushButton("Start")
        self.start_button.setMinimumWidth(110)
        self.start_button.clicked.connect(self._on_button)

        top = QHBoxLayout()
        top.addWidget(QLabel("Microphone:"))
        top.addWidget(self.input_box, stretch=2)
        top.addWidget(QLabel("Output:"))
        top.addWidget(self.output_box, stretch=2)
        top.addWidget(self.start_button)

        # -- captions -------------------------------------------------------
        self.captions = QPlainTextEdit()
        self.captions.setReadOnly(True)
        self.captions.setMaximumBlockCount(_MAX_CAPTION_BLOCKS)
        font = QFont()
        font.setPointSize(11)
        self.captions.setFont(font)
        self.captions.setPlaceholderText(
            "Captions appear here.\n\n"
            "Pick your microphone, set Output to 'CABLE Input' for meetings\n"
            "(or your speakers to listen), then press Start."
        )

        # -- status bar -----------------------------------------------------
        self.state_label = QLabel("idle")
        self.latency_label = QLabel("")
        self.latency_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status = self.statusBar()
        status.addWidget(self.state_label, stretch=1)
        status.addPermanentWidget(self.latency_label)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(self.captions, stretch=1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    # -- selections ---------------------------------------------------------
    def select_devices(self, input_name: str | None, output_name: str | None) -> None:
        """Preselect the devices the container resolved as defaults.

        Args:
            input_name: Microphone to preselect, if present in the list.
            output_name: Output to preselect, if present in the list.
        """
        if input_name is not None:
            index = self.input_box.findText(input_name)
            if index >= 0:
                self.input_box.setCurrentIndex(index)
        if output_name is not None:
            index = self.output_box.findText(output_name)
            if index >= 0:
                self.output_box.setCurrentIndex(index)

    # -- slots fed by the bridge (always on the UI thread) -------------------
    def append_caption(self, language: str, text: str) -> None:
        """Show a transcript line.

        Args:
            language: ISO code shown as the line's tag.
            text: Recognised text.
        """
        self._append_line(f"[{language}] {text}")

    def append_translation(self, text: str, from_cache: bool) -> None:
        """Show a translation line.

        Args:
            text: Translated text.
            from_cache: Whether it came from the cache.
        """
        suffix = "  (cached)" if from_cache else ""
        self._append_line(f"     -> {text}{suffix}")

    def show_timing(self, eou_ms: float, stt_ms: float, mt_ms: float, tts_ms: float) -> None:
        """Update the latency readout.

        Args:
            eou_ms: End-of-utterance to first audio, the headline number.
            stt_ms: Recognition time.
            mt_ms: Translation time.
            tts_ms: First-chunk synthesis time.
        """
        self._latencies.append(eou_ms)
        mean = sum(self._latencies) / len(self._latencies)
        self.latency_label.setText(
            f"latency {eou_ms / 1000.0:.1f} s (mean {mean / 1000.0:.1f})   "
            f"stt {stt_ms:.0f} | mt {mt_ms:.0f} | tts {tts_ms:.0f} ms"
        )

    def show_error(self, stage: str, message: str) -> None:
        """Show a stage failure in the captions view.

        Args:
            stage: Failed stage name.
            message: Error text.
        """
        self._append_line(f"[error in {stage}] {message}")

    def set_speech_state(self, state: str) -> None:
        """Reflect the segmenter state while running.

        Args:
            state: ``"speech"`` or ``"silence"``.
        """
        if self._running:
            self.state_label.setText("listening..." if state == "silence" else "hearing speech")

    def set_status(self, text: str) -> None:
        """Show a status-bar message (loading progress, errors).

        Args:
            text: Message to show.
        """
        self.state_label.setText(text)

    # -- lifecycle driven by the controller ----------------------------------
    def show_progress(self, label: str, detail: str) -> None:
        """Show a component-loading line in the captions view.

        Args:
            label: Component label, empty for continuation lines.
            detail: Human-readable detail.
        """
        line = f"{label}: {detail}" if label else f"   {detail}"
        self._append_line(line)
        self.set_status(f"loading - {label or detail}")

    def set_running(self, running: bool) -> None:
        """Flip the window between idle and interpreting.

        Args:
            running: Whether the pipeline is now live.
        """
        self._running = running
        self._busy = False
        self.start_button.setEnabled(True)
        self.start_button.setText("Stop" if running else "Start")
        self.input_box.setEnabled(not running)
        self.output_box.setEnabled(not running)
        self.set_status("listening..." if running else "stopped")

    def set_busy(self, message: str) -> None:
        """Disable controls while a start or stop is in flight.

        Args:
            message: Status message to show meanwhile.
        """
        self._busy = True
        self.start_button.setEnabled(False)
        self.set_status(message)

    @property
    def is_running(self) -> bool:
        """Whether the window believes the pipeline is live."""
        return self._running

    # -- internals -----------------------------------------------------------
    def _on_button(self) -> None:
        """Start or stop, depending on the current state."""
        if self._busy:
            return
        if self._running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit(
                self.input_box.currentText(),
                self.output_box.currentText(),
            )

    def _append_line(self, line: str) -> None:
        """Append one line to the captions view and keep it scrolled.

        Args:
            line: Text to append.
        """
        self.captions.appendPlainText(line)
        self.captions.moveCursor(QTextCursor.MoveOperation.End)
