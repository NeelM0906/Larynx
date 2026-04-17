"""Helper modules for ``scripts/soak_test.py``.

Split by responsibility so each file stays under the ~300-line ceiling
and is independently reviewable:

- :mod:`scripts.soak_utils.sampling` -- psutil + nvidia-smi wrappers
- :mod:`scripts.soak_utils.metrics` -- Prometheus text-format parser
- :mod:`scripts.soak_utils.report` -- timeseries -> SOAK_REPORT.md

``corpus.txt`` lives alongside this package and is read by the traffic
streams in ``soak_test.py`` for the TTS workload. It's a fixed set of
sentences (not randomly generated at boot) so that a soak run is
bit-for-bit reproducible given the same random seed.
"""

from __future__ import annotations

import pathlib

CORPUS_PATH = pathlib.Path(__file__).resolve().parent / "corpus.txt"


def load_corpus() -> list[str]:
    """Return every non-blank line from ``corpus.txt``.

    Raises ``FileNotFoundError`` if the corpus file was removed from
    the checkout -- we don't fall back to a hard-coded default because
    that would let a broken install pass ``--help`` but silently run
    an anaemic load profile.
    """

    lines = [line.strip() for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


__all__ = ["CORPUS_PATH", "load_corpus"]
