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

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
