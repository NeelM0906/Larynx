"""Tests for the rolling-buffer streaming helper.

The drop-last-5 trick is central to Fun-ASR streaming (see
``demo2.py`` in the upstream Fun-ASR repo). The real tokenizer is a
Qwen3 BPE, but for the helper's correctness what matters is:

1. Round-trip: ``decode(encode(x))`` reproduces ``x`` (modulo BPE
   quirks for the real tokenizer, which we use a character tokenizer
   stub to sidestep).
2. Dropping N tokens actually removes N tokens' worth of text.
3. When the input has fewer than N tokens, the helper returns ``""``.
4. The UTF-8 replacement character cleanup runs on the decoded output.

If the worker ever accidentally uses a module-level tokenizer instead
of ``kwargs["tokenizer"]`` from the loaded model, these tests still
pass — but the GPU test (test_real_model.py) will catch that because
tokenization IDs won't match what the model used.
"""

from __future__ import annotations

import pytest
from larynx_funasr_worker.streaming_utils import (
    drop_last_n_tokens,
    tokenizer_from_kwargs,
)


class CharTokenizer:
    """Character tokenizer — one token per Python char."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_drop_zero_is_identity() -> None:
    tok = CharTokenizer()
    assert drop_last_n_tokens("hello world", tok, 0) == "hello world"


def test_drop_n_removes_exactly_n_tokens() -> None:
    tok = CharTokenizer()
    # 11 chars -> 11 tokens under the char tokenizer; dropping 5 leaves 6.
    assert drop_last_n_tokens("hello world", tok, 5) == "hello "


def test_drop_all_tokens_returns_empty() -> None:
    tok = CharTokenizer()
    assert drop_last_n_tokens("hi", tok, 5) == ""


def test_drop_equal_to_length_returns_empty() -> None:
    tok = CharTokenizer()
    assert drop_last_n_tokens("hello", tok, 5) == ""


def test_empty_input_is_empty() -> None:
    tok = CharTokenizer()
    assert drop_last_n_tokens("", tok, 5) == ""


def test_replacement_character_scrubbed() -> None:
    """BPE tokenizers can leave a U+FFFD when truncation cuts inside a
    multi-byte codepoint. The helper must remove it before returning."""

    class ReplacementTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text)))

        def decode(self, ids: list[int]) -> str:
            return "hello\ufffd"

    out = drop_last_n_tokens("hello world", ReplacementTokenizer(), 5)
    assert "\ufffd" not in out
    assert out == "hello"


def test_tokenizer_from_kwargs_extracts_from_dict() -> None:
    tok = CharTokenizer()
    kwargs = {"tokenizer": tok, "frontend": object(), "model_path": "/x"}
    got = tokenizer_from_kwargs(kwargs)
    assert got is tok


def test_tokenizer_from_kwargs_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="tokenizer"):
        tokenizer_from_kwargs({})


def test_streaming_sequence_is_stable() -> None:
    """Simulate the rolling buffer: each partial drops last 5; final keeps all.

    Ensures an identical suffix in adjacent partials stays stable after
    the drop — that's the whole point of dropping the tail.
    """
    tok = CharTokenizer()
    # Two overlapping partials where the final tokens differ. With the
    # char tokenizer, "hello world foo" (15 chars) drops last 5 -> 10
    # chars; "hello world foobar!" (19 chars) drops last 5 -> 14 chars.
    partial_a = "hello world foo"
    partial_b = "hello world foobar!"
    a_trimmed = drop_last_n_tokens(partial_a, tok, 5)
    b_trimmed = drop_last_n_tokens(partial_b, tok, 5)
    assert a_trimmed == "hello worl"
    assert b_trimmed == "hello world fo"
    # b_trimmed is a proper superset of a_trimmed: stream never retracts
    # characters that survived a previous drop.
    assert b_trimmed.startswith(a_trimmed)
