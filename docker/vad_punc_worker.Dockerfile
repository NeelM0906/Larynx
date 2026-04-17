# VAD + Punctuation worker — CPU only. No CUDA base needed.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ffmpeg ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/shared/pyproject.toml packages/shared/
COPY packages/funasr_worker/pyproject.toml packages/funasr_worker/
COPY packages/voxcpm_worker/pyproject.toml packages/voxcpm_worker/
COPY packages/vad_punc_worker/pyproject.toml packages/vad_punc_worker/
COPY packages/gateway/pyproject.toml packages/gateway/

# Install only the vad_punc_worker extras — avoids pulling torch CUDA
# builds on this lightweight container.
RUN uv sync --no-install-workspace || true

CMD ["uv", "run", "larynx-vad-punc-worker"]
