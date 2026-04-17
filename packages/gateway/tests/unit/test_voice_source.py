"""Unit tests for the VoiceSource Literal type.

VoiceSource is an application-level enum expressed as ``typing.Literal`` —
the DB column stays ``VARCHAR(32)`` so migrations don't have to juggle enum
types. Pydantic picks up the Literal and validates incoming payloads at
runtime; mypy / ruff use it for static checks.

The M7 design doc (ORCHESTRATION-M7.md §4.1 / §8.4) calls for this as a
prerequisite so the later ``'lora'`` addition is one-line.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from larynx_gateway.db.models import VoiceSource
from larynx_gateway.schemas.voice import VoiceResponse


VALID_SOURCES: list[VoiceSource] = ["uploaded", "designed", "seed", "lora"]


def _make_response_payload(source: str) -> dict[str, object]:
    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "name": "sample",
        "source": source,
        "created_at": "2026-04-17T00:00:00+00:00",
        "updated_at": "2026-04-17T00:00:00+00:00",
    }


@pytest.mark.parametrize("source", VALID_SOURCES)
def test_voice_response_accepts_all_valid_sources(source: str) -> None:
    response = VoiceResponse.model_validate(_make_response_payload(source))
    assert response.source == source


@pytest.mark.parametrize(
    "bad_source",
    ["unknown", "LORA", "", "Uploaded", "cloned"],
)
def test_voice_response_rejects_unknown_sources(bad_source: str) -> None:
    with pytest.raises(ValidationError):
        VoiceResponse.model_validate(_make_response_payload(bad_source))


def test_voice_source_literal_exposes_all_variants() -> None:
    # Sanity-check that the Literal is wired with every variant the design
    # doc names. Uses typing.get_args so a renamed variant still fails
    # here loudly.
    import typing

    variants = typing.get_args(VoiceSource)
    assert set(variants) == {"uploaded", "designed", "seed", "lora"}
