#!/usr/bin/env bash
# Thin wrapper around `alembic upgrade head`. Invoked by `make migrate`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../packages/gateway"
exec uv run alembic upgrade head
