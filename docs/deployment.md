# Deployment guide

Covers the two supported v1 deployment shapes and the one concrete
cross-cutting operational concern that needs explicit configuration:
log rotation. Everything else (supervisord program definitions,
worker lifecycle, Postgres + Redis dependencies) ships with defaults
that don't need per-site customisation.

See `ORCHESTRATION-M8.md` §3.4 for the design-side requirement.

## § 1. Deployment shapes

### 1.1 Docker compose (default)

The `docker-compose.yml` at the repo root brings up the stateful
dependencies (Postgres + Redis) and — behind the `--profile workers`
flag — the Fun-ASR + VAD+punctuation worker containers. The gateway
process itself runs on the host under `uvicorn` (directly or via
`make run`); the workers live in Docker because their CUDA + FunASR
stack is self-contained there.

```bash
docker compose up -d                      # Postgres + Redis only
docker compose --profile workers up -d    # + funasr_worker + vad_punc_worker
make run                                  # gateway on host
```

The gateway's structured logs go to stdout/stderr. Under systemd
that means the unit file picks them up; under `make run` they land
in the terminal and rotate only if you send them somewhere.

### 1.2 Bare-metal supervisord (single-host production)

For a single-host production box (GPU 0 for VoxCPM2, GPU 1 for
Fun-ASR, everything orchestrated by a single supervisord), use
`supervisord.conf` at the repo root. Every program gets
`autorestart=true` with bounded retries; the
`restart_alerter` eventlistener logs `critical.restart_storm`
after 3 failures in 60s so a monitoring agent can page off that
signal.

In this shape each worker writes to its own stdout, which
supervisord redirects to files under `/var/log/supervisor/`
(default) or wherever `stdout_logfile=` in the program block
points. These are the files the logrotate snippet in §2.2 targets.

## § 2. Log rotation

### 2.1 Docker compose log rotation

Every service block in `docker-compose.yml` carries:

```yaml
logging:
  driver: json-file
  options:
    max-size: "100m"
    max-file: "14"
```

Ceiling per service: 14 × 100MB = 1.4 GB. With four services that's
a 5.6 GB worst-case under normal log volume — well inside any
production disk budget and sized to retain a couple of weeks of
structured JSON logs for post-hoc incident investigation.

The `json-file` driver is Docker's built-in default. No extra
package or daemon reconfiguration is required; Docker rotates in
place.

To inspect rotated files:

```bash
docker inspect --format='{{.LogPath}}' larynx-funasr-worker
# /var/lib/docker/containers/<id>/<id>-json.log
ls -lh /var/lib/docker/containers/<id>/          # includes *-json.log.1 .. *-json.log.14
```

### 2.2 Bare-metal (systemd + logrotate)

When the gateway + workers run under supervisord with file-based
stdout logging, use the stdlib `logrotate` daemon. Drop the
following into `/etc/logrotate.d/larynx`:

```
/var/log/larynx/*.log {
    daily
    rotate 14
    missingok
    notifempty
    copytruncate
    compress
    delaycompress
    dateext
    dateformat -%Y%m%d
}
```

Notes:

- `copytruncate` is required because supervisord keeps a long-lived
  file descriptor open on each program's stdout log. A move-based
  rotation would leave the old file held open and growing; copy +
  truncate rotates the visible file while preserving the fd.
- `dateext` + `dateformat` give human-greppable filenames
  (`larynx.log-20260419`) instead of numeric suffixes.
- `compress` + `delaycompress` gzips everything older than
  yesterday's log. Today's rotated file stays uncompressed so a
  running incident investigation doesn't have to `zcat`.
- `rotate 14` matches the docker-compose ceiling so operators think
  about one retention window, not two.

Point your `supervisord.conf` stdout/stderr to `/var/log/larynx/`:

```ini
[program:gateway]
stdout_logfile=/var/log/larynx/gateway.log
stderr_logfile=/var/log/larynx/gateway.err
```

…and so on for each program. Create the directory with
`install -d -o larynx -g larynx -m 0755 /var/log/larynx` on first
deploy.

Verify logrotate's view of the config with:

```bash
sudo logrotate --debug /etc/logrotate.d/larynx
```

The daily cron in `/etc/cron.daily/logrotate` runs at 03:47 by
default (Debian/Ubuntu) or at a systemd-timer-driven cadence
(Fedora/RHEL). Either is fine — log rotation is not time-critical.

## § 3. Beyond log rotation

v1 does not ship a Grafana dashboard bundle, per-user rate limiting,
or multi-tenant auth. See `ORCHESTRATION-M8.md` §5.3 for the
full descoping list. Production monitoring in v1 is expected to
scrape the gateway's `/metrics` + the `/metrics/workers` proxy, and
page off the `critical.restart_storm` alerter line — that's the
operator contract.
