"""Windows audio endpoint discovery via the registry.

Phase 3 introduces the real device layer built on PortAudio. This module
exists so that ``--check`` can already answer one question today: *is VB-CABLE
installed?* That driver must be installed and the machine rebooted before
Phase 7, and finding out early is far better than finding out at the end.

Windows records every audio endpoint under::

    HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\MMDevices\\Audio\\{Render|Capture}

Each endpoint is a GUID-named subkey whose ``Properties`` subkey holds
PROPERTYKEY-named values. Reading it needs no drivers, no audio libraries and
no elevated permissions.

Every failure path degrades to an empty result. The registry layout is a
Windows implementation detail; if it ever changes, ``--check`` should report
"could not determine" rather than crash the application.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Final

__all__ = ["AudioEndpoint", "list_windows_audio_endpoints"]

logger = logging.getLogger(__name__)

# winreg only exists on Windows. Importing it under the platform guard keeps
# this module importable everywhere while letting the type checker resolve it
# normally, which a dynamic import would prevent.
if sys.platform == "win32":
    import winreg

_MMDEVICES_ROOT: Final[str] = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio"

# PROPERTYKEY value names, tried in order of usefulness:
#   PKEY_Device_FriendlyName          -> "Speakers (Conexant ISST Audio)"
#   PKEY_DeviceInterface_FriendlyName -> "Conexant ISST Audio"
#   PKEY_Device_DeviceDesc            -> "Speakers"
_NAME_PROPERTY_KEYS: Final[tuple[str, ...]] = (
    "{a45c254e-df1c-4efd-8020-67d146a850e0},14",
    "{b3f8fa53-0004-438e-9003-51a46e139bfc},6",
    "{a45c254e-df1c-4efd-8020-67d146a850e0},2",
)

# DEVICE_STATE_ACTIVE from mmdeviceapi.h
_DEVICE_STATE_ACTIVE: Final[int] = 1

_DIRECTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("render", "Render"),
    ("capture", "Capture"),
)


@dataclass(frozen=True, slots=True)
class AudioEndpoint:
    """An audio endpoint recorded by Windows.

    Args:
        name: Friendly device name.
        direction: ``"render"`` for playback, ``"capture"`` for recording.
        is_active: Whether the endpoint is currently present and enabled.
    """

    name: str
    direction: str
    is_active: bool

    @property
    def is_virtual_cable(self) -> bool:
        """Whether the name looks like a virtual audio cable."""
        haystack = self.name.casefold()
        return any(marker in haystack for marker in ("cable", "voicemeeter", "virtual"))


def list_windows_audio_endpoints(*, active_only: bool = True) -> tuple[AudioEndpoint, ...]:
    """Enumerate audio endpoints from the Windows registry.

    Args:
        active_only: Exclude endpoints that are disabled or unplugged.

    Returns:
        Detected endpoints sorted by direction and name, or an empty tuple on a
        non-Windows platform or if the registry cannot be read.
    """
    if sys.platform != "win32":
        return ()

    endpoints: list[AudioEndpoint] = []
    for direction, subkey in _DIRECTIONS:
        endpoints.extend(_enumerate_direction(direction, subkey))

    if active_only:
        endpoints = [endpoint for endpoint in endpoints if endpoint.is_active]

    return tuple(sorted(endpoints, key=lambda item: (item.direction, item.name.casefold())))


def _enumerate_direction(direction: str, subkey: str) -> list[AudioEndpoint]:
    """Read every endpoint registered for one direction.

    Args:
        direction: Label recorded on the results.
        subkey: Registry subkey, ``"Render"`` or ``"Capture"``.

    Returns:
        Endpoints found, empty on any failure.
    """
    path = f"{_MMDEVICES_ROOT}\\{subkey}"
    results: list[AudioEndpoint] = []

    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
    except OSError as exc:
        logger.debug("Could not open %s: %s", path, exc)
        return results

    with root:
        index = 0
        while True:
            try:
                guid = winreg.EnumKey(root, index)
            except OSError:
                # Raised once the subkeys are exhausted.
                break
            index += 1

            endpoint = _read_endpoint(root, guid, direction)
            if endpoint is not None:
                results.append(endpoint)

    return results


def _read_endpoint(root: winreg.HKEYType, guid: str, direction: str) -> AudioEndpoint | None:
    """Read one endpoint's name and activation state.

    Args:
        root: Open registry key for the direction.
        guid: GUID subkey identifying the endpoint.
        direction: Label recorded on the result.

    Returns:
        The endpoint, or ``None`` when no readable name was found.
    """
    name = _read_endpoint_name(root, guid)
    if name is None:
        return None

    return AudioEndpoint(name=name, direction=direction, is_active=_read_device_state(root, guid))


def _read_device_state(root: winreg.HKEYType, guid: str) -> bool:
    """Read whether an endpoint is active.

    Args:
        root: Open registry key for the direction.
        guid: GUID subkey identifying the endpoint.

    Returns:
        ``True`` when active. An absent or unreadable state is treated as
        active, so a device the user can see in Windows is never hidden here.
    """
    try:
        with winreg.OpenKey(root, guid) as device_key:
            state, _ = winreg.QueryValueEx(device_key, "DeviceState")
            return int(state) == _DEVICE_STATE_ACTIVE
    except (OSError, TypeError, ValueError):
        return True


def _read_endpoint_name(root: winreg.HKEYType, guid: str) -> str | None:
    """Read the friendliest available name for an endpoint.

    Args:
        root: Open registry key for the direction.
        guid: GUID subkey identifying the endpoint.

    Returns:
        The best available name, or ``None`` when none could be read.
    """
    try:
        properties = winreg.OpenKey(root, f"{guid}\\Properties")
    except OSError:
        return None

    with properties:
        for property_key in _NAME_PROPERTY_KEYS:
            try:
                value, _ = winreg.QueryValueEx(properties, property_key)
            except OSError:
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None
