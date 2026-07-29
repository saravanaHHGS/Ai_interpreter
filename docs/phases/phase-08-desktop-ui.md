# Phase 8 - Desktop interface

`.\run.ps1 --ui` opens a single window: pick a microphone, set Output to
`CABLE Input (VB-Audio Virtual Cable)` for meetings (or speakers to listen),
press **Start**. Captions stream into the middle pane - `[ta]` transcripts,
`->` translations - and the status bar shows the live latency with its
per-stage breakdown. **Stop** tears the session down and prints a summary
line (utterances, fusions, reroutes) into the captions.

Built after Phase 9 by design: the UI is a *view over the pipeline*, and the
pipeline had to exist and be measured first. `PipelineEvents` was shaped for
this moment - five callbacks, each mapping 1:1 onto a Qt signal.

## The Smart App Control gate

The first act of this phase was not design but a gate check: SAC has blocked
PyTorch permanently and mypy's binaries transiently on this machine (see
`architecture.md` §0, C6). PySide6 6.11.1 (Qt Company signing) **loads and
runs** - verified with an offscreen QApplication exercising widgets, the
event loop and signal delivery before any UI code was written. Only
`PySide6-Essentials` is depended on: QtWidgets/QtCore/QtGui without ~300 MB
of Addons this project will never use.

## Architecture

Three rules produced the three modules:

**One assembly path** (`app/assembly.py`). The moment a second front end
appeared, the pipeline assembly moved out of `--interpret` into a shared
`build_interpretation_bundle()`: same captions-only fallback, same
code-switch fallback with word timestamps, same glossary wiring for CLI and
UI. Progress is a `(label, detail)` callback - the CLI renders rows, the UI
renders status lines, from the same events.

**Signals are the only bridge** (`presentation/ui/bridge.py`). Pipeline
callbacks fire on worker threads; Qt widgets may only be touched from their
own thread. Emitting a Qt signal from a foreign thread queues delivery onto
the receiver's thread, so each callback does exactly one thing: unpack the
domain object to primitives and emit. The cross-thread property has its own
test.

**The window is a view** (`presentation/ui/main_window.py`). It owns widgets
and formatting, speaks back only via `start_requested`/`stop_requested`,
and never imports a model. That is what makes it fully testable under Qt's
`offscreen` platform - 13 headless tests cover captions, device selection,
button state, latency readout and the bridge.

**Model work never runs on the UI thread** (`presentation/ui/app.py`).
`InterpreterController` builds and warms the bundle (~10 s of neural-network
loading) on a plain Python thread and reports back through signals. Start
builds a fresh bundle; Stop tears it down completely - a warmup per session
in exchange for no half-alive state between sessions. Keeping warmed models
across sessions is a Phase 10 optimisation confined to the controller.

## Bounds

The caption view is capped at 500 blocks (`setMaximumBlockCount`) so a
day-long meeting cannot grow memory, and the busy flag makes start/stop
strictly sequential - a second click during a transition is ignored rather
than racing the worker thread.
