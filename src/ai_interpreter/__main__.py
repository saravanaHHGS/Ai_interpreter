"""Enables ``python -m ai_interpreter``.

Keeping this separate from :mod:`ai_interpreter.cli` means the CLI's ``main``
can be imported and called by tests without the module-level side effect of
executing it.
"""

from __future__ import annotations

from ai_interpreter.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
