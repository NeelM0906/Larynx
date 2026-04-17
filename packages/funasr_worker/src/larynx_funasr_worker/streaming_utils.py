"""Rolling-buffer streaming helpers for Fun-ASR.

The key trick (from ``demo2.py`` in the upstream Fun-ASR repo) is that on
each intermediate partial we drop the last N (typically 5) tokens from
the decoded text before returning it to the client. Those tail tokens
are the ones most likely to be revised when more audio arrives, so
hiding them keeps partials stable at the cost of a small tail lag.

The tokenizer used here MUST be the one the Fun-ASR model produced the
tokens from. In practice that means ``kwargs["tokenizer"]`` returned by
``FunASRNano.from_pretrained`` — not a module-level instance. Using a
different tokenizer (even the same model name loaded separately) will
silently produce wrong token offsets and the drop will cut garbled UTF-8
mid-codepoint.

The ``.replace("\\ufffd", "")`` at the end of :func:`drop_last_n_tokens`
cleans up the lone replacement-byte that Fun-ASR's BPE tokenizer leaves
behind when the truncation lands mid-codepoint (e.g. inside a CJK
character). Matches the upstream demo behaviour exactly.
"""

from __future__ import annotations

from typing import Any, Protocol


class _TokenizerLike(Protocol):
    """Structural type for the Fun-ASR tokenizer interface we depend on.

    Accepts both a transformers tokenizer (encode -> list[int], decode ->
    str) and the FunASR ``CharTokenizer`` — both expose these two methods
    with compatible signatures.
    """

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


def drop_last_n_tokens(text: str, tokenizer: _TokenizerLike, n: int = 5) -> str:
    """Return ``text`` with its last ``n`` tokens removed.

    - ``n <= 0`` is a no-op.
    - If ``text`` has fewer than ``n`` tokens, returns an empty string —
      matches the upstream demo (no partial emitted until enough audio
      has arrived).
    - Stray UTF-8 replacement characters (``\\ufffd``) left by truncating
      inside a multi-byte codepoint are stripped.
    """
    if n <= 0 or not text:
        return text
    ids = tokenizer.encode(text)
    if len(ids) <= n:
        return ""
    return tokenizer.decode(ids[:-n]).replace("\ufffd", "")


def tokenizer_from_kwargs(kwargs: dict[str, Any]) -> _TokenizerLike:
    """Pull the tokenizer off the dict returned by ``FunASRNano.from_pretrained``.

    The upstream demo accesses ``tokenizer`` as a module-level variable;
    when we wrap Fun-ASR in an async worker, the tokenizer must come from
    the ``kwargs`` dict returned alongside the model — using a freshly
    constructed tokenizer would produce mismatched token IDs.
    """
    tok = kwargs.get("tokenizer")
    if tok is None:
        raise RuntimeError(
            "Fun-ASR tokenizer missing from from_pretrained() kwargs — "
            "did the upstream API change? See FunASRNano.from_pretrained in "
            "yuekaizhang/Fun-ASR-vllm/model.py"
        )
    return tok
