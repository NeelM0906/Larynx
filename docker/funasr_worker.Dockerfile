# Fun-ASR STT worker container.
#
# Uses the nvidia CUDA base matching the M0 smoke env (torch 2.9 / cu128).
# Installs uv, then the workspace with --extra gpu so torch + vllm +
# flash-attn land. Entry point is supervisord which runs only the
# funasr_worker program (see supervisord.conf).

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git ffmpeg supervisor ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv so workspace resolution matches the host.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# We bind-mount the repo at runtime (see docker-compose.yml); the image
# only needs to prewarm dependency caches. Copying pyproject/uv.lock
# lets docker cache the install layer across repo edits.
COPY pyproject.toml uv.lock ./
COPY packages/shared/pyproject.toml packages/shared/
COPY packages/funasr_worker/pyproject.toml packages/funasr_worker/
COPY packages/voxcpm_worker/pyproject.toml packages/voxcpm_worker/
COPY packages/vad_punc_worker/pyproject.toml packages/vad_punc_worker/
COPY packages/gateway/pyproject.toml packages/gateway/

RUN uv sync --extra gpu --no-install-workspace || true

CMD ["uv", "run", "larynx-funasr-worker"]
