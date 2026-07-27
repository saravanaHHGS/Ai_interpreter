"""Audio device discovery via PortAudio.

Windows exposes the same physical device through several host APIs, and the
choice between them matters more than it looks:

===============  =========================================================
Host API         Behaviour
===============  =========================================================
**WASAPI**       Full device names, native 48 kHz, lowest latency. Preferred.
DirectSound      Full names, but resamples to 44.1 kHz and adds latency.
MME              **Truncates device names to 31 characters.** Legacy.
WDM-KS           Exclusive access; steals the device from other applications.
===============  =========================================================

The MME truncation is not cosmetic. On the development machine MME reports
``"Internal Microphone (Conexant I"`` while WASAPI reports
``"Internal Microphone (Conexant ISST Audio)"``. Configuration stores device
*names* - indices are reassigned whenever a USB headset is unplugged - so a
truncated name silently fails to match, and the application would fall back to
the wrong microphone with no error.

PortAudio's default device is whatever Windows nominates, which is usually
MME. This module therefore prefers WASAPI explicitly rather than trusting the
default.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Final

from ai_interpreter.domain.entities import DeviceInfo
from ai_interpreter.domain.errors import DeviceError, DeviceNotFoundError
from ai_interpreter.domain.value_objects import DeviceKind

__all__ = ["HOST_API_PREFERENCE", "SounddeviceDeviceEnumerator"]

logger = logging.getLogger(__name__)

# Best first. Matched case-insensitively against the PortAudio host API name.
HOST_API_PREFERENCE: Final[tuple[str, ...]] = (
    "wasapi",
    "directsound",
    "mme",
    "wdm-ks",
)

# Virtual endpoints PortAudio exposes that are not real devices.
_PSEUDO_DEVICE_MARKERS: Final[tuple[str, ...]] = (
    "microsoft sound mapper",
    "primary sound capture driver",
    "primary sound driver",
)


# Shortest prefix treated as identifying a device. Long enough that unrelated
# devices sharing a generic word ("Speakers", "Headset") are not merged.
_MIN_PREFIX_MATCH: Final[int] = 12


def _same_physical_device(name: str, other: str) -> bool:
    """Whether two host APIs are reporting the same physical device.

    MME truncates names to 31 characters, so ``"Internal Microphone (Conexant I"``
    and ``"Internal Microphone (Conexant ISST Audio)"`` are the same microphone
    seen through different host APIs. Equality would miss that; a prefix match
    in either direction catches it.

    Args:
        name: First device name.
        other: Second device name.

    Returns:
        ``True`` when both names refer to the same device.
    """
    first = name.casefold().strip()
    second = other.casefold().strip()

    if first == second:
        return True
    if min(len(first), len(second)) < _MIN_PREFIX_MATCH:
        return False
    return first.startswith(second) or second.startswith(first)


class SounddeviceDeviceEnumerator:
    """Lists audio endpoints, satisfying the ``DeviceEnumerator`` port.

    Args:
        preferred_host_api: Host API name fragment to favour, or ``None`` to
            use :data:`HOST_API_PREFERENCE`.
        include_pseudo_devices: Include PortAudio's aggregate entries such as
            "Microsoft Sound Mapper". Off by default: selecting one hides
            which physical device is actually in use.
    """

    def __init__(
        self,
        preferred_host_api: str | None = None,
        *,
        include_pseudo_devices: bool = False,
    ) -> None:
        self._preferred_host_api = preferred_host_api
        self._include_pseudo = include_pseudo_devices

    # -- listing -----------------------------------------------------------
    def list_devices(self, kind: DeviceKind | None = None) -> Sequence[DeviceInfo]:
        """List audio endpoints.

        Args:
            kind: Restrict to capture or playback endpoints, or ``None`` for
                both.

        Returns:
            Endpoints ordered by host API preference, then by name.

        Raises:
            DeviceError: If PortAudio cannot enumerate devices.
        """
        raw_devices, host_apis, defaults = self._query()

        devices: list[DeviceInfo] = []
        for index, raw in enumerate(raw_devices):
            host_api_name = self._host_api_name(host_apis, raw)
            if not self._include_pseudo and self._is_pseudo_device(str(raw["name"])):
                continue

            for direction in (DeviceKind.INPUT, DeviceKind.OUTPUT):
                if kind is not None and direction is not kind:
                    continue
                channels = int(
                    raw["max_input_channels"]
                    if direction is DeviceKind.INPUT
                    else raw["max_output_channels"]
                )
                if channels < 1:
                    continue
                devices.append(
                    DeviceInfo(
                        index=index,
                        name=str(raw["name"]),
                        kind=direction,
                        max_channels=channels,
                        default_sample_rate=float(raw["default_samplerate"]),
                        host_api=host_api_name,
                        is_default=index == defaults.get(direction),
                    )
                )

        devices.sort(key=lambda d: (self._host_api_rank(d.host_api), d.name.casefold()))
        return tuple(devices)

    def default_device(self, kind: DeviceKind) -> DeviceInfo | None:
        """Return the best available endpoint for a direction.

        Deliberately not PortAudio's own default, which on Windows is usually
        an MME endpoint with a truncated name. The highest-ranked host API
        offering the system default device is chosen instead, falling back to
        the first device of the preferred host API.

        Args:
            kind: Capture or playback.

        Returns:
            The chosen endpoint, or ``None`` if the machine has none.
        """
        devices = self.list_devices(kind)
        if not devices:
            return None

        system_default = next((d for d in devices if d.is_default), None)
        if system_default is None:
            return devices[0]

        # Find the same physical device on a better host API. Names cannot be
        # compared with == here: Windows nominates an MME endpoint as the
        # default, and MME truncates to 31 characters, so the MME and WASAPI
        # entries for one microphone never match exactly. Since list_devices
        # is sorted by host API preference, the first prefix match is the best
        # available version of the same device.
        preferred = next(
            (d for d in devices if _same_physical_device(d.name, system_default.name)),
            system_default,
        )
        if preferred is not system_default:
            logger.debug(
                "Default device %r [%s] upgraded to %r [%s]",
                system_default.name,
                system_default.host_api,
                preferred.name,
                preferred.host_api,
            )
        return preferred

    def find_device(self, name_fragment: str, kind: DeviceKind) -> DeviceInfo | None:
        """Find an endpoint whose name contains a fragment.

        Args:
            name_fragment: Case-insensitive substring, e.g. ``"CABLE Input"``.
            kind: Capture or playback.

        Returns:
            The best-ranked match, or ``None`` when nothing matches.
        """
        needle = name_fragment.strip().casefold()
        if not needle:
            return None

        matches = [d for d in self.list_devices(kind) if needle in d.name.casefold()]
        if not matches:
            return None
        # list_devices is already sorted by host API preference.
        return matches[0]

    def resolve(self, name_fragment: str | None, kind: DeviceKind) -> DeviceInfo:
        """Resolve a configured device name, or fall back to the default.

        Args:
            name_fragment: Configured name fragment, or ``None`` for the
                default device.
            kind: Capture or playback.

        Returns:
            The endpoint to open.

        Raises:
            DeviceNotFoundError: If a name was configured but matches nothing,
                or if the machine has no endpoint of this kind at all. Falling
                back silently would leave the user recording the wrong device.
        """
        if name_fragment:
            found = self.find_device(name_fragment, kind)
            if found is None:
                available = "\n".join(
                    f"    - {d.name}  [{d.host_api}]" for d in self.list_devices(kind)
                )
                msg = (
                    f"No {kind.value} device matching {name_fragment!r}.\n"
                    f"Available {kind.value} devices:\n{available or '    (none)'}"
                )
                raise DeviceNotFoundError(msg)
            return found

        default = self.default_device(kind)
        if default is None:
            msg = f"This machine has no {kind.value} audio device."
            raise DeviceNotFoundError(msg)
        return default

    # -- internals ---------------------------------------------------------
    def _query(self) -> tuple[Sequence[Any], Sequence[Any], dict[DeviceKind, int]]:
        """Read raw device and host API tables from PortAudio.

        Returns:
            The device list, host API list, and default device indices.

        Raises:
            DeviceError: If PortAudio fails.
        """
        try:
            import sounddevice as sd

            raw_devices = sd.query_devices()
            host_apis = sd.query_hostapis()
            default_input, default_output = sd.default.device
        except Exception as exc:
            msg = (
                f"Could not enumerate audio devices: {exc}\n"
                "Check that Windows audio services are running and that at least "
                "one audio device is enabled in Sound settings."
            )
            raise DeviceError(msg) from exc

        defaults: dict[DeviceKind, int] = {}
        if isinstance(default_input, int) and default_input >= 0:
            defaults[DeviceKind.INPUT] = default_input
        if isinstance(default_output, int) and default_output >= 0:
            defaults[DeviceKind.OUTPUT] = default_output

        return raw_devices, host_apis, defaults

    @staticmethod
    def _host_api_name(host_apis: Sequence[Any], raw_device: Any) -> str:
        """Resolve a device's host API name.

        Args:
            host_apis: Host API table from PortAudio.
            raw_device: Raw device mapping.

        Returns:
            The host API name, or an empty string if it cannot be resolved.
        """
        try:
            return str(host_apis[int(raw_device["hostapi"])]["name"])
        except (IndexError, KeyError, TypeError, ValueError):
            return ""

    def _host_api_rank(self, host_api: str) -> int:
        """Rank a host API; lower sorts first.

        Args:
            host_api: Host API name reported by PortAudio.

        Returns:
            Sort rank, with unknown APIs placed last.
        """
        name = host_api.casefold()
        if self._preferred_host_api and self._preferred_host_api.casefold() in name:
            return -1
        for rank, candidate in enumerate(HOST_API_PREFERENCE):
            if candidate in name:
                return rank
        return len(HOST_API_PREFERENCE)

    @staticmethod
    def _is_pseudo_device(name: str) -> bool:
        """Whether a device name is one of PortAudio's aggregate entries.

        Args:
            name: Device name.

        Returns:
            ``True`` for entries that are not real hardware.
        """
        lowered = name.casefold()
        return any(marker in lowered for marker in _PSEUDO_DEVICE_MARKERS)
