"""Domain layer: entities, value objects, ports and errors.

This layer is deliberately free of framework dependencies. It may import the
standard library and :mod:`numpy` (treated as a primitive data type for audio
buffers, in the same spirit as :mod:`decimal`), and nothing else.

Because nothing here imports a model runtime or an audio driver, the whole
layer is importable in milliseconds and testable without any hardware.
"""

from __future__ import annotations

__all__: list[str] = []
