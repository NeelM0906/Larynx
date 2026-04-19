# Staging Verification Report

**Run timestamp:** 2026-04-19T04:41:11Z
**Total duration:** 124.1s
**Verdict:** **PASS**

Replacement for the 24h soak per ORCHESTRATION-M8.md §7.3. This is 
not a substitute for a long-horizon soak — it catches Part C hardening 
issues and gross leaks, not 18-hour GPU accumulation or slow drifts.

## Phase results

### load_run — **PASS** (120.2s)

```json
{
  "total_requests": 120,
  "success_rate": 1.0,
  "ready_checks_ok": 2,
  "ready_checks_degraded": 0,
  "per_minute_buckets": [
    {
      "tts_ok": 60,
      "tts_fail": 0
    },
    {
      "tts_ok": 60,
      "tts_fail": 0
    }
  ]
}
```

### drain_test — **PASS** (2.7s)

```json
{
  "batch_job_id": "fd42d2f7098a48e48110fa9365579c88",
  "ready_503_after_s": 0.234605906996876,
  "process_exit_after_s": 1.4,
  "exit_code": 143
}
```

### mem_delta — **PASS** (1.2s)

```json
{
  "num_requests": 100,
  "requests_ok": 100,
  "requests_failed": 0,
  "per_process_deltas": [
    {
      "pid": 2886229,
      "rss_before_mb": 3.7,
      "rss_after_mb": 3.7,
      "growth_pct": 0.0
    },
    {
      "pid": 2886231,
      "rss_before_mb": 50.4,
      "rss_after_mb": 50.4,
      "growth_pct": 0.0
    },
    {
      "pid": 2886235,
      "rss_before_mb": 299.4,
      "rss_after_mb": 299.4,
      "growth_pct": 0.0
    },
    {
      "pid": 2886288,
      "rss_before_mb": 17.0,
      "rss_after_mb": 17.0,
      "growth_pct": 0.0
    },
    {
      "pid": 2889320,
      "rss_before_mb": 16.5,
      "rss_after_mb": 16.5,
      "growth_pct": 0.0
    },
    {
      "pid": 2890723,
      "rss_before_mb": 3.8,
      "rss_after_mb": 3.8,
      "growth_pct": 0.0
    }
  ],
  "flagged_processes": []
}
```

### restart_alerter — **SKIPPED** (0.0s)

Skipped: supervisord not installed on this host; the harness can't drive the PROCESS_STATE events end-to-end. Unit coverage in packages/gateway/tests/unit/test_restart_alerter.py exercises the threshold + window logic; the end-to-end supervisord → eventlistener path stays a deploy-time manual verification.

## Artifacts

Raw phase JSON: `staging-artifacts/phase_results.json`
