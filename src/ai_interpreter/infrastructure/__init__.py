"""Infrastructure layer: concrete adapters for the domain ports.

Everything that touches the outside world lives here - audio drivers, model
runtimes, the file system, the registry. Isolating it means the rest of the
application can be tested without any of it.
"""

from __future__ import annotations

__all__: list[str] = []
