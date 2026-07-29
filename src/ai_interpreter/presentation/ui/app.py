"""Wiring the desktop UI: controller, entry point, and thread choreography.

The rule that shapes everything here: **model work never runs on the UI
thread.** Building a bundle warms four neural networks (~10 s on the target
CPU) and stopping one drains audio; both happen on plain Python threads,
reporting back through Qt signals - the same queued-delivery mechanism the
pipeline bridge uses, applied to lifecycle instead of captions.

Start builds a fresh bundle and stop tears it down completely. Rebuilding
per session costs a warmup each time but leaves no half-alive component
state between sessions - correctness first; keeping warmed models across
sessions is a Phase 10 optimisation inside the controller only.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from ai_interpreter.app.assembly import InterpretationBundle, build_interpretation_bundle
from ai_interpreter.app.container import Container
from ai_interpreter.domain.errors import InterpreterError
from ai_interpreter.domain.value_objects import DeviceKind, LanguagePair
from ai_interpreter.presentation.ui.bridge import PipelineBridge
from ai_interpreter.presentation.ui.main_window import MainWindow

__all__ = ["InterpreterController", "run_ui"]

logger = logging.getLogger(__name__)


class InterpreterController(QObject):
    """Owns the bundle lifecycle on behalf of the window.

    Args:
        container: Built application container.
        pair: Direction to interpret.
    """

    progress = Signal(str, str)  # component label, detail
    started = Signal(bool)  # captions_only
    stopped = Signal(str)  # session summary line
    failed = Signal(str)  # error message

    def __init__(self, container: Container, pair: LanguagePair) -> None:
        super().__init__()
        self._container = container
        self._pair = pair
        self.bridge = PipelineBridge(self)
        self._bundle: InterpretationBundle | None = None
        self._worker: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        """Whether a bundle is currently live."""
        return self._bundle is not None

    def start(self, input_device: str, output_device: str) -> None:
        """Build, warm and start a session on a background thread.

        Args:
            input_device: Microphone name from the picker.
            output_device: Output name from the picker.
        """
        if self._bundle is not None or self._is_busy():
            return

        def build() -> None:
            try:
                bundle = build_interpretation_bundle(
                    self._container,
                    self._pair,
                    self.bridge.events(),
                    input_device=input_device or None,
                    output_device=output_device or None,
                    on_progress=self.progress.emit,
                )
                bundle.pipeline.start()
            except InterpreterError as exc:
                logger.warning("UI session failed to start: %s", exc)
                self.failed.emit(str(exc))
                return
            self._bundle = bundle
            self.started.emit(bundle.captions_only)

        self._worker = threading.Thread(target=build, name="ui-session-start", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        """Stop and tear down the session on a background thread."""
        bundle, self._bundle = self._bundle, None
        if bundle is None or self._is_busy():
            return

        def teardown() -> None:
            stats = bundle.pipeline.stats()
            bundle.shutdown()
            self.stopped.emit(
                f"session: {stats.utterances_out}/{stats.utterances_in} interpreted, "
                f"{stats.word_fusions} fused, {stats.code_switch_reroutes} rerouted, "
                f"{stats.failures} failed"
            )

        self._worker = threading.Thread(target=teardown, name="ui-session-stop", daemon=True)
        self._worker.start()

    def shutdown(self, timeout: float = 15.0) -> None:
        """Synchronous teardown for window close.

        Args:
            timeout: Seconds to wait for any in-flight worker first.
        """
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
        bundle, self._bundle = self._bundle, None
        if bundle is not None:
            bundle.shutdown()

    def _is_busy(self) -> bool:
        """Whether a start or stop thread is still running."""
        worker = self._worker
        return worker is not None and worker.is_alive()


def run_ui(container: Container) -> int:
    """Launch the desktop interface and block until it closes.

    Args:
        container: Built application container.

    Returns:
        Process exit code from the Qt event loop.
    """
    settings = container.settings
    pair = LanguagePair.of(settings.app.language_pair.source, settings.app.language_pair.target)

    application = QApplication.instance() or QApplication([])

    inputs = [device.name for device in container.devices.list_devices(DeviceKind.INPUT)]
    outputs = [device.name for device in container.devices.list_devices(DeviceKind.OUTPUT)]
    window = MainWindow(
        direction=f"{pair.source.english_name} -> {pair.target.english_name}",
        input_devices=inputs,
        output_devices=outputs,
    )

    # Preselect what the CLI would have used, so pressing Start with no
    # choices made behaves exactly like --interpret.
    try:
        window.select_devices(
            container.resolve_input_device().name,
            container.resolve_output_device().name,
        )
    except InterpreterError as exc:
        logger.warning("Device preselection failed: %s", exc)

    controller = InterpreterController(container, pair)

    # Window -> controller.
    def on_start(microphone: str, output: str) -> None:
        window.set_busy("loading models...")
        controller.start(microphone, output)

    def on_stop() -> None:
        window.set_busy("stopping...")
        controller.stop()

    window.start_requested.connect(on_start)
    window.stop_requested.connect(on_stop)

    # Controller -> window.
    def on_started(captions_only: bool) -> None:
        window.set_running(True)
        if captions_only:
            window.append_caption("info", "captions only - no voice for the target language")

    def on_stopped(summary: str) -> None:
        window.set_running(False)
        window.append_caption("info", summary)

    def on_failed(message: str) -> None:
        window.set_running(False)
        window.show_error("startup", message)

    controller.progress.connect(window.show_progress)
    controller.started.connect(on_started)
    controller.stopped.connect(on_stopped)
    controller.failed.connect(on_failed)

    # Pipeline -> window, via the bridge.
    controller.bridge.transcript_received.connect(window.append_caption)
    controller.bridge.partial_received.connect(window.show_partial)
    controller.bridge.translation_received.connect(window.append_translation)
    controller.bridge.timing_received.connect(window.show_timing)
    controller.bridge.error_occurred.connect(window.show_error)
    controller.bridge.state_changed.connect(window.set_speech_state)

    application.aboutToQuit.connect(controller.shutdown)

    window.show()
    return application.exec()
