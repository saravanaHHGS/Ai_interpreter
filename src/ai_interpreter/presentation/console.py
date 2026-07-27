"""Console rendering helpers.

Kept apart from :mod:`ai_interpreter.cli` so the command handlers stay about
*what* to report while these functions handle *how* it looks, and so the
formatting can be unit tested without running a command.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from typing import Final

from ai_interpreter.domain.entities import DeviceInfo

__all__ = [
    "WIDTH",
    "format_device_table",
    "heading",
    "level_bar",
    "row",
    "terminal_width",
]

WIDTH: Final[int] = 78

# Narrowest terminal the live meter still renders in without wrapping.
_MIN_TERMINAL_WIDTH: Final[int] = 40
_FALLBACK_TERMINAL_WIDTH: Final[int] = 80


def terminal_width() -> int:
    """Return the usable console width.

    A line that exactly fills the terminal wraps on Windows, and a wrapped
    line cannot be overwritten with a carriage return - the meter then leaves
    a trail of half-drawn rows instead of updating in place. One column is
    therefore held back.

    Returns:
        Usable width in characters, never below :data:`_MIN_TERMINAL_WIDTH`.
    """
    try:
        columns = shutil.get_terminal_size(fallback=(_FALLBACK_TERMINAL_WIDTH, 24)).columns
    except OSError:
        columns = _FALLBACK_TERMINAL_WIDTH
    return max(_MIN_TERMINAL_WIDTH, columns - 1)


# Full block and light shade. Both are in every Windows console font, unlike
# the fractional block characters a finer meter would need.
_BAR_FILLED: Final[str] = "█"
_BAR_EMPTY: Final[str] = "░"


def heading(title: str) -> None:
    """Print a section heading.

    Args:
        title: Heading text.
    """
    print(f"\n{title}")
    print("-" * WIDTH)


def row(label: str, value: str) -> None:
    """Print an aligned label and value.

    Args:
        label: Left-hand label.
        value: Right-hand value.
    """
    print(f"  {label:<26} {value}")


def level_bar(level: float, width: int = 30) -> str:
    """Render an audio level as a text meter.

    The scale is decibels, not amplitude. A linear amplitude meter spends most
    of its length on levels the ear cannot tell apart, so normal speech sits
    near the far left and the meter looks broken when it is working.

    Args:
        level: Peak amplitude in ``[0.0, 1.0]``.
        width: Meter width in characters.

    Returns:
        A bar string of exactly ``width`` characters.
    """
    width = max(1, width)
    if level <= 0.0:
        filled = 0
    else:
        # -60 dB (inaudible) to 0 dB (full scale) mapped across the bar.
        decibels = 20.0 * math.log10(max(level, 1e-6))
        fraction = max(0.0, min(1.0, (decibels + 60.0) / 60.0))
        filled = round(fraction * width)

    return _BAR_FILLED * filled + _BAR_EMPTY * (width - filled)


def format_device_table(devices: Sequence[DeviceInfo]) -> list[str]:
    """Render audio devices as aligned lines.

    Args:
        devices: Endpoints to display.

    Returns:
        One line per device, empty if there are none.
    """
    if not devices:
        return ["  (none found)"]

    host_width = max(len(device.host_api) for device in devices)
    lines: list[str] = []

    for device in devices:
        markers: list[str] = []
        if device.is_default:
            markers.append("default")
        if device.is_virtual_cable:
            markers.append("virtual cable")
        suffix = f"  <- {', '.join(markers)}" if markers else ""

        lines.append(
            f"  {device.index:>3}  [{device.host_api:<{host_width}}]  "
            f"{device.max_channels}ch {device.default_sample_rate:>6.0f} Hz  "
            f"{device.name}{suffix}"
        )

    return lines
