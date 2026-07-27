"""Unit tests for audio device discovery and selection.

PortAudio is replaced with a fixture reproducing the real device table from
the development machine, including the MME name truncation that caused a real
bug: the system default is an MME endpoint, and matching it against WASAPI by
exact name silently selected the wrong host API.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_interpreter.domain.errors import DeviceNotFoundError
from ai_interpreter.domain.value_objects import DeviceKind
from ai_interpreter.infrastructure.audio import devices as devices_module
from ai_interpreter.infrastructure.audio.devices import (
    SounddeviceDeviceEnumerator,
    _same_physical_device,
)

pytestmark = pytest.mark.unit

# Reproduces the real table: one microphone and one virtual cable exposed
# through MME (truncated names) and WASAPI (full names).
_HOST_APIS = [
    {"name": "MME"},
    {"name": "Windows DirectSound"},
    {"name": "Windows WASAPI"},
]

_DEVICES = [
    {  # 0 - MME pseudo-device
        "name": "Microsoft Sound Mapper - Input",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {  # 1 - MME microphone, name truncated at 31 characters, system default
        "name": "Internal Microphone (Conexant I",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {  # 2 - MME speakers, system default output
        "name": "Speakers (Conexant ISST Audio)",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
    {  # 3 - WASAPI microphone, full name
        "name": "Internal Microphone (Conexant ISST Audio)",
        "hostapi": 2,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {  # 4 - WASAPI virtual cable capture
        "name": "CABLE Output (VB-Audio Virtual Cable)",
        "hostapi": 2,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {  # 5 - WASAPI virtual cable playback
        "name": "CABLE Input (VB-Audio Virtual Cable)",
        "hostapi": 2,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    {  # 6 - DirectSound microphone
        "name": "Internal Microphone (Conexant ISST Audio)",
        "hostapi": 1,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {  # 7 - WASAPI speakers, the better version of the default output
        "name": "Speakers (Conexant ISST Audio)",
        "hostapi": 2,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
]


@pytest.fixture
def enumerator(monkeypatch: pytest.MonkeyPatch) -> SounddeviceDeviceEnumerator:
    """An enumerator backed by the fake device table.

    Args:
        monkeypatch: pytest patcher.

    Returns:
        The enumerator under test.
    """
    instance = SounddeviceDeviceEnumerator(preferred_host_api="WASAPI")

    def fake_query(_self: Any) -> tuple[Any, Any, dict[DeviceKind, int]]:
        return _DEVICES, _HOST_APIS, {DeviceKind.INPUT: 1, DeviceKind.OUTPUT: 2}

    monkeypatch.setattr(SounddeviceDeviceEnumerator, "_query", fake_query)
    return instance


class TestSamePhysicalDevice:
    """Matching truncated MME names against full WASAPI names."""

    def test_identical_names_match(self) -> None:
        assert _same_physical_device("Speakers (Conexant)", "Speakers (Conexant)")

    def test_truncated_name_matches_full_name(self) -> None:
        assert _same_physical_device(
            "Internal Microphone (Conexant ISST Audio)",
            "Internal Microphone (Conexant I",
        )

    def test_match_is_symmetric(self) -> None:
        assert _same_physical_device(
            "Internal Microphone (Conexant I",
            "Internal Microphone (Conexant ISST Audio)",
        )

    def test_different_devices_do_not_match(self) -> None:
        assert not _same_physical_device(
            "Internal Microphone (Conexant ISST Audio)",
            "CABLE Output (VB-Audio Virtual Cable)",
        )

    def test_short_generic_names_do_not_match(self) -> None:
        # "Speakers" is a prefix of many device names; matching on it would
        # merge unrelated hardware.
        assert not _same_physical_device("Speakers", "Speakers (Conexant ISST Audio)")

    def test_case_and_whitespace_are_ignored(self) -> None:
        assert _same_physical_device("  INTERNAL MICROPHONE (Conexant I  ", "internal microphone")


class TestListing:
    """Enumerating and ordering devices."""

    def test_lists_input_devices_only(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        inputs = enumerator.list_devices(DeviceKind.INPUT)
        assert all(device.kind is DeviceKind.INPUT for device in inputs)

    def test_excludes_pseudo_devices(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        names = [device.name for device in enumerator.list_devices(DeviceKind.INPUT)]
        assert "Microsoft Sound Mapper - Input" not in names

    def test_pseudo_devices_can_be_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_query(_self: Any) -> tuple[Any, Any, dict[DeviceKind, int]]:
            return _DEVICES, _HOST_APIS, {}

        monkeypatch.setattr(SounddeviceDeviceEnumerator, "_query", fake_query)
        instance = SounddeviceDeviceEnumerator(include_pseudo_devices=True)
        names = [device.name for device in instance.list_devices(DeviceKind.INPUT)]

        assert "Microsoft Sound Mapper - Input" in names

    def test_wasapi_sorts_first(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        inputs = enumerator.list_devices(DeviceKind.INPUT)
        assert "WASAPI" in inputs[0].host_api

    def test_reports_host_api_and_channels(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        device = enumerator.list_devices(DeviceKind.INPUT)[0]
        assert device.host_api == "Windows WASAPI"
        assert device.max_channels == 2
        assert device.default_sample_rate == 48000.0


class TestDefaultSelection:
    """Choosing a device when configuration names none."""

    def test_upgrades_the_mme_default_to_wasapi(
        self, enumerator: SounddeviceDeviceEnumerator
    ) -> None:
        # The regression this pins: the system default is the MME endpoint
        # with a truncated name, and the WASAPI entry for the same microphone
        # must be preferred.
        default = enumerator.default_device(DeviceKind.INPUT)

        assert default is not None
        assert default.host_api == "Windows WASAPI"
        assert default.name == "Internal Microphone (Conexant ISST Audio)"

    def test_default_output_prefers_wasapi(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        default = enumerator.default_device(DeviceKind.OUTPUT)
        assert default is not None
        assert "WASAPI" in default.host_api

    def test_returns_none_when_no_devices_exist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            SounddeviceDeviceEnumerator,
            "_query",
            lambda _self: ([], _HOST_APIS, {}),
        )
        assert SounddeviceDeviceEnumerator().default_device(DeviceKind.INPUT) is None


class TestFindAndResolve:
    """Matching a configured device name."""

    def test_finds_by_partial_name(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        found = enumerator.find_device("CABLE Output", DeviceKind.INPUT)
        assert found is not None
        assert found.name.startswith("CABLE Output")

    def test_match_is_case_insensitive(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        assert enumerator.find_device("cable output", DeviceKind.INPUT) is not None

    def test_prefers_wasapi_among_matches(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        found = enumerator.find_device("Internal Microphone", DeviceKind.INPUT)
        assert found is not None
        assert "WASAPI" in found.host_api

    def test_returns_none_when_nothing_matches(
        self, enumerator: SounddeviceDeviceEnumerator
    ) -> None:
        assert enumerator.find_device("Nonexistent Device", DeviceKind.INPUT) is None

    def test_resolve_falls_back_to_the_default(
        self, enumerator: SounddeviceDeviceEnumerator
    ) -> None:
        resolved = enumerator.resolve(None, DeviceKind.INPUT)
        assert "WASAPI" in resolved.host_api

    def test_resolve_uses_the_configured_name(
        self, enumerator: SounddeviceDeviceEnumerator
    ) -> None:
        resolved = enumerator.resolve("CABLE Output", DeviceKind.INPUT)
        assert resolved.name.startswith("CABLE Output")

    def test_resolve_raises_rather_than_silently_falling_back(
        self, enumerator: SounddeviceDeviceEnumerator
    ) -> None:
        # Falling back would leave the user recording the wrong microphone
        # with no indication anything went wrong.
        with pytest.raises(DeviceNotFoundError, match="No input device matching"):
            enumerator.resolve("Nonexistent Device", DeviceKind.INPUT)

    def test_error_lists_available_devices(self, enumerator: SounddeviceDeviceEnumerator) -> None:
        with pytest.raises(DeviceNotFoundError, match="Internal Microphone"):
            enumerator.resolve("Nonexistent Device", DeviceKind.INPUT)


class TestHostApiRanking:
    """Ordering of host APIs."""

    def test_preference_order_is_wasapi_first(self) -> None:
        assert devices_module.HOST_API_PREFERENCE[0] == "wasapi"

    def test_explicit_preference_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_query(_self: Any) -> tuple[Any, Any, dict[DeviceKind, int]]:
            return _DEVICES, _HOST_APIS, {}

        monkeypatch.setattr(SounddeviceDeviceEnumerator, "_query", fake_query)
        instance = SounddeviceDeviceEnumerator(preferred_host_api="DirectSound")
        first = instance.list_devices(DeviceKind.INPUT)[0]

        assert "DirectSound" in first.host_api
