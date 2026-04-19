# Staging Verification Report

**Run timestamp:** 2026-04-19T07:34:30Z
**Total duration:** 147.6s
**Verdict:** **PASS**
**Gateway mode:** workers(funasr=ready, vad_punc=ready, voxcpm=ready) · gpus=2

Replacement for the 24h soak per ORCHESTRATION-M8.md §7.3. This is 
not a substitute for a long-horizon soak — it catches Part C hardening 
issues and gross leaks, not 18-hour GPU accumulation or slow drifts.

## Gateway snapshot (start of run)

```json
{
  "env": {},
  "workers": {
    "voxcpm": {
      "state": "ready"
    },
    "funasr": {
      "state": "ready"
    },
    "vad_punc": {
      "state": "ready"
    }
  },
  "gpus": [
    {
      "gpu": "0",
      "metric": "gpu_utilization_percent",
      "value": 5.0
    },
    {
      "gpu": "0",
      "metric": "gpu_memory_used_mib",
      "value": 80642.0
    },
    {
      "gpu": "0",
      "metric": "gpu_memory_total_mib",
      "value": 97887.0
    },
    {
      "gpu": "0",
      "metric": "gpu_temperature_c",
      "value": 62.0
    },
    {
      "gpu": "1",
      "metric": "gpu_utilization_percent",
      "value": 0.0
    },
    {
      "gpu": "1",
      "metric": "gpu_memory_used_mib",
      "value": 84756.0
    },
    {
      "gpu": "1",
      "metric": "gpu_memory_total_mib",
      "value": 97887.0
    },
    {
      "gpu": "1",
      "metric": "gpu_temperature_c",
      "value": 44.0
    }
  ],
  "version": "0.3.0"
}
```

## Phase results

### load_run — **PASS** (120.1s)

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
  "batch_job_id": "360715d6768c411cbc9f6e6dee6d719a",
  "ready_503_after_s": 0.26886431500315666,
  "process_exit_after_s": 1.43,
  "exit_code": 143
}
```

### mem_delta — **PASS** (24.7s)

```json
{
  "num_requests": 100,
  "requests_ok": 100,
  "requests_failed": 0,
  "per_process_deltas": [
    {
      "pid": 3066655,
      "rss_before_mb": 3.8,
      "rss_after_mb": 3.8,
      "growth_pct": 0.0
    },
    {
      "pid": 3066659,
      "rss_before_mb": 51.0,
      "rss_after_mb": 51.0,
      "growth_pct": 0.0
    },
    {
      "pid": 3066663,
      "rss_before_mb": 4706.7,
      "rss_after_mb": 4706.7,
      "growth_pct": 0.0
    },
    {
      "pid": 3069765,
      "rss_before_mb": 17.4,
      "rss_after_mb": 17.4,
      "growth_pct": 0.0
    },
    {
      "pid": 3073652,
      "rss_before_mb": 16.9,
      "rss_after_mb": 16.9,
      "growth_pct": 0.0
    },
    {
      "pid": 3082613,
      "rss_before_mb": 3.8,
      "rss_after_mb": 3.8,
      "growth_pct": 0.0
    }
  ],
  "flagged_processes": [],
  "gpu_before": [
    {
      "index": null,
      "metric": "gpu_utilization_percent",
      "value": 18.0
    },
    {
      "index": null,
      "metric": "gpu_memory_used_mib",
      "value": 80640.0
    },
    {
      "index": null,
      "metric": "gpu_memory_total_mib",
      "value": 97887.0
    },
    {
      "index": null,
      "metric": "gpu_temperature_c",
      "value": 70.0
    },
    {
      "index": null,
      "metric": "gpu_utilization_percent",
      "value": 0.0
    },
    {
      "index": null,
      "metric": "gpu_memory_used_mib",
      "value": 84756.0
    },
    {
      "index": null,
      "metric": "gpu_memory_total_mib",
      "value": 97887.0
    },
    {
      "index": null,
      "metric": "gpu_temperature_c",
      "value": 44.0
    }
  ],
  "gpu_after": [
    {
      "index": null,
      "metric": "gpu_utilization_percent",
      "value": 2.0
    },
    {
      "index": null,
      "metric": "gpu_memory_used_mib",
      "value": 80642.0
    },
    {
      "index": null,
      "metric": "gpu_memory_total_mib",
      "value": 97887.0
    },
    {
      "index": null,
      "metric": "gpu_temperature_c",
      "value": 73.0
    },
    {
      "index": null,
      "metric": "gpu_utilization_percent",
      "value": 0.0
    },
    {
      "index": null,
      "metric": "gpu_memory_used_mib",
      "value": 84756.0
    },
    {
      "index": null,
      "metric": "gpu_memory_total_mib",
      "value": 97887.0
    },
    {
      "index": null,
      "metric": "gpu_temperature_c",
      "value": 44.0
    }
  ]
}
```

### restart_alerter — **SKIPPED** (0.0s)

Skipped: --skip 4 requested on the CLI

## Artifacts

Raw phase JSON: `staging-artifacts/phase_results.json`
