"""AI Interpreter - real-time speech-to-speech translation for Windows.

The package is organised as four Clean Architecture layers:

``domain``
    Entities, value objects and ports (interfaces). Depends on the standard
    library and numpy only. Knows nothing about models, audio drivers or Qt.
``application``
    Use cases and orchestration. Depends on ``domain`` abstractions only.
``infrastructure``
    Concrete adapters: audio drivers, ML runtimes, config files, logging.
``presentation``
    PySide6 user interface (added in Phase 8).

``app.container`` is the composition root - the single place where concrete
implementations are bound to the ports they satisfy.
"""

from __future__ import annotations

__version__ = "0.4.0"

__all__ = ["__version__"]
