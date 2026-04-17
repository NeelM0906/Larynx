# Larynx 24h Soak Report (template)

This file is a **template**. `scripts/soak_test.py` overwrites it at the
end of each run with concrete numbers from the timeseries parquet. The
structure below documents what the final report will contain; it does
not claim any specific result.

## Context

- Started / ended / duration. ISO-8601 UTC timestamps.
- Path to `errors.jsonl` captured alongside the report.

## Endpoint traffic

A single table with one row per route the harness exercised plus (when
the gateway was healthy and the `/metrics` scrape worked) a second set
of rows marked `[... , server]` whose timings come from the
`larynx_request_duration_seconds` histogram on the gateway itself.

| Route | Total | Errors | Err% | p50 (ms) | p95 (ms) | p99 (ms) |
|-------|------:|-------:|-----:|---------:|---------:|---------:|

Client-side rows include network time; server-side rows do not. The
two views usually differ by a few milliseconds.

## Process RSS / CPU

One row per PID whose name matched the tracked pattern
(`larynx|voxcpm|funasr|vad_punc|training_worker` by default). The
growth column is the linear-regression slope scaled to the run
duration, expressed as a percentage of the starting RSS.

| Name | PID | RSS start (MiB) | RSS end (MiB) | Growth % | CPU mean | CPU p95 |
|------|-----|----------------:|--------------:|---------:|---------:|--------:|

## GPUs

One row per GPU as reported by `nvidia-smi`. VRAM columns are MiB.

| GPU | VRAM min | VRAM max | VRAM mean | Util mean | Temp max |
|-----|---------:|---------:|----------:|----------:|---------:|

If `nvidia-smi` was absent the section renders a single placeholder
line and the run is not auto-failed on that basis.

## Disk

`shutil.disk_usage(--data-dir)` delta between first and last sample.
On a passing 24h run this number should be bounded by batch-output
accumulation modulo the 7-day cleanup cron.

## Restart storms

Lines from `--stderr-log` containing `critical.restart_storm`. On a
clean run the section reads **_None detected._**. Non-zero counts
immediately fail the run.

## Quality degradation check

At the end of the run the harness requests 10 fresh TTS outputs,
computes RMS + spectral centroid (FFT-based, numpy-only) on each, and
compares the mean centroid of the first half versus the second half.
Drift > 5% flags as `DEGRADED` and fails the verdict.

The section is skipped with a warning when `numpy` or `pyarrow` are
missing, or when fewer than four samples completed successfully.

## Verdict

`Verdict: PASS` iff all of the following hold:

- every endpoint's error rate is `<= error_rate_ceiling` (default 1%)
- every tracked process's RSS growth is `<= rss_growth_ceiling_pct`
  (default 5%)
- the restart-storm list is empty
- the quality-degradation check is not `DEGRADED`

Otherwise `Verdict: FAIL` and one bullet per reason. The exact rules
live in `scripts/soak_utils/report.py::_evaluate`.
