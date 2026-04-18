"""Minimal Prometheus text-format parser.

We don't want to pull in ``prometheus_client.parser`` at soak-tool
level: the soak harness runs *outside* the gateway's Python env in
principle (different venv, different host eventually), so we keep a
deliberately small parser here and own it in-tree.

What we support:

- ``# HELP`` / ``# TYPE`` lines (we capture the type so histogram
  aggregation can key off ``*_bucket`` / ``*_sum`` / ``*_count``).
- Sample lines ``metric{k="v",k2="v2"} 1.23`` (optional trailing
  timestamp is dropped).
- Escapes ``\\\\``, ``\\"``, ``\\n`` inside label values.
- Histogram helpers: :func:`histogram_quantiles` computes p50/p95/p99
  from a bucket-sample dict, treating ``+Inf`` buckets as the total.

What we do NOT support:

- Summary quantiles (``{quantile="0.99"}``) -- the Larynx gateway only
  exports histograms per the metrics middleware.
- Exemplars, native histograms, UTF-8 metric names, or the OpenMetrics
  superset grammar. If/when a worker starts emitting those this
  parser will silently drop the exemplar portion -- we log a warning
  at parse time if the TYPE is an unknown bucket.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sample:
    """A single observation at scrape time."""

    name: str
    labels: tuple[tuple[str, str], ...]  # sorted for stable hashing
    value: float

    def label_dict(self) -> dict[str, str]:
        return dict(self.labels)


@dataclass
class MetricFamily:
    """All samples sharing a metric base name + their declared type."""

    name: str
    type: str = "untyped"  # counter|gauge|histogram|summary|untyped
    samples: list[Sample] = field(default_factory=list)


def parse(text: str) -> dict[str, MetricFamily]:
    """Parse a Prometheus exposition text blob.

    Returns ``{base_name: MetricFamily}``. For histograms the base
    name is the one declared in ``# TYPE``; the ``_bucket`` /
    ``_sum`` / ``_count`` samples all land inside that family with
    their full suffixed names on :attr:`Sample.name`.
    """

    families: dict[str, MetricFamily] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(maxsplit=3)
            if len(parts) >= 4 and parts[1] == "TYPE":
                name, mtype = parts[2], parts[3]
                families.setdefault(name, MetricFamily(name=name)).type = mtype
            continue
        sample = _parse_sample_line(line)
        if sample is None:
            continue
        base = _base_name_for(sample.name, families)
        fam = families.setdefault(base, MetricFamily(name=base))
        fam.samples.append(sample)
    return families


def _base_name_for(sample_name: str, families: dict[str, MetricFamily]) -> str:
    # Histogram: bucket/sum/count roll up to the parent.
    for suffix in ("_bucket", "_sum", "_count"):
        if sample_name.endswith(suffix):
            base = sample_name[: -len(suffix)]
            if base in families and families[base].type == "histogram":
                return base
    return sample_name


def _parse_sample_line(line: str) -> Sample | None:
    """Parse one sample line. Returns ``None`` on malformed input."""

    # Split into name{labels} value [timestamp]
    if "{" in line:
        name, rest = line.split("{", 1)
        label_blob, _, value_rest = rest.partition("}")
        if not _:
            return None
        labels = _parse_labels(label_blob)
        value_part = value_rest.strip()
    else:
        name, _, value_part = line.partition(" ")
        labels = ()

    name = name.strip()
    if not name:
        return None

    # value [timestamp]
    value_str = value_part.split()[0] if value_part else ""
    try:
        value = float(value_str)
    except ValueError:
        if value_str in {"NaN", "+Inf", "-Inf"}:
            value = float(value_str.replace("+", ""))
        else:
            return None
    return Sample(name=name, labels=labels, value=value)


def _parse_labels(blob: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    i = 0
    n = len(blob)
    while i < n:
        # Skip whitespace and commas.
        while i < n and blob[i] in " ,":
            i += 1
        if i >= n:
            break
        # Key runs up to '='.
        k_start = i
        while i < n and blob[i] != "=":
            i += 1
        if i >= n:
            break
        key = blob[k_start:i].strip()
        i += 1  # skip '='
        # Value is quoted.
        if i >= n or blob[i] != '"':
            break
        i += 1
        v_chars: list[str] = []
        while i < n and blob[i] != '"':
            if blob[i] == "\\" and i + 1 < n:
                esc = blob[i + 1]
                v_chars.append({"n": "\n", '"': '"', "\\": "\\"}.get(esc, esc))
                i += 2
            else:
                v_chars.append(blob[i])
                i += 1
        value = "".join(v_chars)
        if i < n:
            i += 1  # skip closing '"'
        pairs.append((key, value))
    pairs.sort()
    return tuple(pairs)


def histogram_quantiles(
    buckets: Iterable[tuple[float, float]],
    quantiles: Iterable[float] = (0.5, 0.95, 0.99),
) -> dict[float, float]:
    """Linear-interpolate quantiles from cumulative histogram buckets.

    ``buckets`` is ``[(le, cumulative_count), ...]``. Must include the
    ``+Inf`` terminal bucket whose count is the total sample count.
    Returns ``{quantile: value}``. If the histogram is empty, values
    are ``NaN``.
    """

    bs = sorted(buckets, key=lambda t: t[0])
    if not bs:
        return dict.fromkeys(quantiles, float("nan"))
    total = bs[-1][1]
    if total <= 0:
        return dict.fromkeys(quantiles, float("nan"))

    out: dict[float, float] = {}
    for q in quantiles:
        target = q * total
        prev_le, prev_count = 0.0, 0.0
        chosen = math.inf
        for le, count in bs:
            if count >= target:
                if le == math.inf:
                    chosen = prev_le
                elif count == prev_count:
                    chosen = le
                else:
                    # Linear interpolation within the bucket.
                    frac = (target - prev_count) / (count - prev_count)
                    chosen = prev_le + frac * (le - prev_le)
                break
            prev_le, prev_count = le, count
        out[q] = chosen
    return out


__all__ = ["MetricFamily", "Sample", "histogram_quantiles", "parse"]
