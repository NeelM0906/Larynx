"""Language router tests — ISO code → (model, Fun-ASR language name)."""

from __future__ import annotations

import pytest
from larynx_funasr_worker.language_router import (
    FunASRModel,
    UnsupportedLanguageError,
    resolve,
    supported_languages,
)


def test_none_routes_to_nano_with_autodetect() -> None:
    model, name = resolve(None)
    assert model is FunASRModel.NANO
    assert name is None


def test_empty_string_routes_to_nano_with_autodetect() -> None:
    model, name = resolve("")
    assert model is FunASRModel.NANO
    assert name is None


@pytest.mark.parametrize(
    ("iso", "expected_name"),
    [("zh", "中文"), ("en", "英文"), ("ja", "日文")],
)
def test_nano_languages(iso: str, expected_name: str) -> None:
    model, name = resolve(iso)
    assert model is FunASRModel.NANO
    assert name == expected_name


@pytest.mark.parametrize(
    ("iso", "expected_name"),
    [
        ("yue", "粤语"),
        ("ko", "韩文"),
        ("pt", "葡萄牙语"),
        ("nl", "荷兰语"),
        ("ar", "阿拉伯语"),
        ("hi", "印地语"),
        ("pl", "波兰语"),
    ],
)
def test_mlt_languages(iso: str, expected_name: str) -> None:
    model, name = resolve(iso)
    assert model is FunASRModel.MLT
    assert name == expected_name


def test_region_suffix_stripped() -> None:
    for variant in ["en-US", "en_GB", "EN", "en-gb"]:
        model, name = resolve(variant)
        assert model is FunASRModel.NANO
        assert name == "英文"


def test_unsupported_language_raises() -> None:
    # es/fr are common languages NOT in Fun-ASR's coverage — PRD §5.3
    # says these fall back to Whisper in the future; for now we raise.
    with pytest.raises(UnsupportedLanguageError):
        resolve("es")
    with pytest.raises(UnsupportedLanguageError):
        resolve("fr")


def test_supported_languages_includes_nano_and_mlt() -> None:
    langs = set(supported_languages())
    assert {"zh", "en", "ja"} <= langs
    assert {"yue", "ko", "pt", "nl", "ar"} <= langs
    # sanity: ~30 codes
    assert len(langs) >= 28
