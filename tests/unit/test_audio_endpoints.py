"""Unit tests for Windows audio endpoint naming and classification.

Registry access itself is not unit tested - it is exercised by ``--check`` on a
real machine. The naming rules are, because that is where the logic lives and
where a mistake is invisible until Phase 7 routes audio to the wrong endpoint.
"""

from __future__ import annotations

import pytest

from ai_interpreter.infrastructure.system.audio_endpoints import (
    AudioEndpoint,
    compose_endpoint_name,
)

pytestmark = pytest.mark.unit


class TestComposeEndpointName:
    """Building the name Windows itself would display."""

    def test_prefers_the_friendly_name_when_present(self) -> None:
        name = compose_endpoint_name(
            "CABLE Input (VB-Audio Virtual Cable)", "CABLE Input", "VB-Audio Virtual Cable"
        )
        assert name == "CABLE Input (VB-Audio Virtual Cable)"

    def test_combines_description_and_interface_when_friendly_name_is_absent(self) -> None:
        # The real case on the development machine: PKEY_Device_FriendlyName
        # does not exist for any endpoint.
        name = compose_endpoint_name(None, "CABLE Input", "VB-Audio Virtual Cable")
        assert name == "CABLE Input (VB-Audio Virtual Cable)"

    def test_distinguishes_endpoints_sharing_an_interface(self) -> None:
        # Every VB-CABLE endpoint reports the same interface name, so the
        # description is the only thing that tells the two halves apart.
        cable_in = compose_endpoint_name(None, "CABLE Input", "VB-Audio Virtual Cable")
        cable_out = compose_endpoint_name(None, "CABLE Output", "VB-Audio Virtual Cable")
        cable_16ch = compose_endpoint_name(None, "CABLE In 16ch", "VB-Audio Virtual Cable")

        assert len({cable_in, cable_out, cable_16ch}) == 3

    def test_treats_an_empty_friendly_name_as_absent(self) -> None:
        name = compose_endpoint_name("   ", "Speakers", "Conexant ISST Audio")
        assert name == "Speakers (Conexant ISST Audio)"

    def test_falls_back_to_description_alone(self) -> None:
        assert compose_endpoint_name(None, "Speakers", None) == "Speakers"

    def test_falls_back_to_interface_alone(self) -> None:
        assert compose_endpoint_name(None, None, "Conexant ISST Audio") == "Conexant ISST Audio"

    def test_returns_none_when_nothing_is_available(self) -> None:
        assert compose_endpoint_name(None, None, None) is None
        assert compose_endpoint_name("", "  ", "") is None

    def test_strips_surrounding_whitespace(self) -> None:
        assert compose_endpoint_name(None, " CABLE Input ", " VB-Audio Virtual Cable ") == (
            "CABLE Input (VB-Audio Virtual Cable)"
        )


class TestVirtualCableDetection:
    """Classifying an endpoint as a virtual cable."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("CABLE Input (VB-Audio Virtual Cable)", True),
            ("CABLE Output (VB-Audio Virtual Cable)", True),
            ("CABLE In 16ch (VB-Audio Virtual Cable)", True),
            ("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", True),
            ("Speakers (Conexant ISST Audio)", False),
            ("Headset (ZEB-SOUND BOMB 7 Hands-Free)", False),
            ("Headphones (soundcore R50i)", False),
        ],
    )
    def test_identifies_virtual_cables(self, name: str, expected: bool) -> None:
        endpoint = AudioEndpoint(name=name, direction="render", is_active=True)
        assert endpoint.is_virtual_cable is expected
