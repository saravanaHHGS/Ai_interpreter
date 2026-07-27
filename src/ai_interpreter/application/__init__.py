"""Application layer: use cases and orchestration.

Depends on domain abstractions only. No module here may import a model
runtime, an audio driver or Qt - that restriction is what allows the whole
layer to be unit tested in milliseconds with plain fakes.
"""

from __future__ import annotations

__all__: list[str] = []
