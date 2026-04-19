.PHONY: help up down test lint fmt migrate smoke run clean

SHELL := /bin/bash
PYTHON ?= uv run python
PORT ?= 8000

help:
	@echo "Larynx M1 — common targets:"
	@echo "  make up       Start Postgres + Redis (docker compose)"
	@echo "  make down     Stop Postgres + Redis"
	@echo "  make migrate  Run alembic upgrade head"
	@echo "  make run      Start the gateway (foreground)"
	@echo "  make smoke    Start the gateway, POST /v1/tts, save WAV"
	@echo "  make test     Run pytest"
	@echo "  make lint     Run ruff check + format --check"
	@echo "  make fmt      Run ruff format"
	@echo "  make clean    Remove caches + venv"

up:
	docker compose up -d
	@echo "waiting for postgres..." && \
	  until docker compose exec -T postgres pg_isready -U larynx -d larynx >/dev/null 2>&1; do sleep 1; done && \
	  echo "postgres ready"
	@echo "waiting for redis..." && \
	  until docker compose exec -T redis redis-cli ping >/dev/null 2>&1; do sleep 1; done && \
	  echo "redis ready"

down:
	docker compose down

migrate:
	cd packages/gateway && uv run alembic upgrade head

run:
	uv run uvicorn larynx_gateway.main:app --host 0.0.0.0 --port $(PORT) --reload

smoke:
	$(PYTHON) scripts/smoke_test.py

test:
	uv run pytest -q

test-real:
	RUN_REAL_MODEL=1 uv run pytest -m real_model -q -s

# Workaround for bugs/002 (real_model GPU accumulation across modules):
# run each real-model file in its own pytest process so vLLM subprocess
# handles are reaped on interpreter exit. Use this instead of `test-real`
# until bugs/002 is closed.
test-real-per-file:
	@set -e; for f in \
	  packages/gateway/tests/integration/test_real_model.py \
	  packages/gateway/tests/integration/test_real_model_stream.py \
	  packages/gateway/tests/integration/test_real_model_conversation.py \
	  packages/gateway/tests/integration/test_real_model_stt.py \
	  packages/gateway/tests/integration/test_m0_smoke_roundtrip.py; do \
	    echo "=== $$f ==="; \
	    RUN_REAL_MODEL=1 uv run pytest "$$f" -m real_model -v; \
	  done

seed:
	uv run python scripts/load_demo_voices.py

measure:
	uv run python scripts/measure_cache.py

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
