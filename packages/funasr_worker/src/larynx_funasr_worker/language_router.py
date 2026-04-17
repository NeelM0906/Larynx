"""Route an ISO-639 language code to Fun-ASR-Nano vs MLT-Nano.

Fun-ASR exposes two checkpoints that together cover 31 languages:

* **Fun-ASR-Nano-2512** — zh, en, ja (and regional dialects of zh)
* **Fun-ASR-MLT-Nano-2512** — 28 additional languages (plus the Nano three,
  but accuracy on zh/en/ja is lower than the dedicated Nano)

The language param travels through the API as an ISO-639 code; Fun-ASR
itself expects a Chinese-name string (e.g. ``"英文"`` for English) passed
as ``language=`` to ``m.inference``. This module owns both sides of that
translation.

``None`` routes to Nano with no explicit language — Fun-ASR-Nano's built-in
auto-detect handles zh/en/ja cleanly. For an MLT-only language, callers
must pass the ISO code explicitly (MLT can't auto-detect across 31
languages reliably).
"""

from __future__ import annotations

from enum import StrEnum


class FunASRModel(StrEnum):
    NANO = "nano"
    MLT = "mlt"


# ISO-639 → Chinese-name string that Fun-ASR's prompt template expects.
# Source: FunAudioLLM/Fun-ASR README + model.py prompt templates (L553+).
_NANO_LANGUAGES: dict[str, str] = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
}

_MLT_LANGUAGES: dict[str, str] = {
    # The MLT checkpoint also handles zh/en/ja but we prefer Nano for those.
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "yue": "粤语",  # Cantonese — PRD's "Chinese dialect" coverage lives here
    "ko": "韩文",
    "vi": "越南语",
    "id": "印尼语",
    "th": "泰语",
    "ms": "马来语",
    "tl": "菲律宾语",
    "ar": "阿拉伯语",
    "hi": "印地语",
    "bg": "保加利亚语",
    "hr": "克罗地亚语",
    "cs": "捷克语",
    "da": "丹麦语",
    "nl": "荷兰语",
    "et": "爱沙尼亚语",
    "fi": "芬兰语",
    "el": "希腊语",
    "hu": "匈牙利语",
    "ga": "爱尔兰语",
    "lv": "拉脱维亚语",
    "lt": "立陶宛语",
    "mt": "马耳他语",
    "pl": "波兰语",
    "pt": "葡萄牙语",
    "ro": "罗马尼亚语",
    "sk": "斯洛伐克语",
    "sl": "斯洛文尼亚语",
    "sv": "瑞典语",
}


class UnsupportedLanguageError(ValueError):
    """Raised when neither Nano nor MLT covers the requested language.

    PRD §5.3 says these should fall back to Whisper large-v3, but that
    worker is out of scope for M3 — surface a clean error instead so the
    gateway can translate it to 400.
    """


def resolve(language: str | None) -> tuple[FunASRModel, str | None]:
    """Pick the Fun-ASR variant + its language-string for a caller's ISO code.

    Returns ``(model, funasr_language_name_or_None)``. The second item is
    what gets passed to ``m.inference(language=...)``; ``None`` means
    "let Fun-ASR auto-detect" (only allowed on the Nano checkpoint).
    """
    if language is None:
        return FunASRModel.NANO, None

    code = language.strip().lower()
    if not code:
        return FunASRModel.NANO, None

    # Normalise common variants so callers can say "en-US" or "zh_CN".
    code = code.split("-")[0].split("_")[0]

    if code in _NANO_LANGUAGES:
        return FunASRModel.NANO, _NANO_LANGUAGES[code]
    if code in _MLT_LANGUAGES:
        return FunASRModel.MLT, _MLT_LANGUAGES[code]

    raise UnsupportedLanguageError(
        f"language {language!r} is not covered by Fun-ASR-Nano or MLT; "
        "PRD §5.3 fallback (Whisper large-v3) is not available in M3"
    )


def supported_languages() -> list[str]:
    """ISO-639 codes covered by Fun-ASR (both models combined).

    Used by the gateway to advertise language support + by tests to pick
    valid sample languages.
    """
    return sorted(set(_NANO_LANGUAGES) | set(_MLT_LANGUAGES))
