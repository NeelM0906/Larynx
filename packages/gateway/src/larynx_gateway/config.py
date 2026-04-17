"""Environment-driven settings for the gateway."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Auth
    larynx_api_token: str = Field(
        default="change-me-please",
        description="Bearer token required by every /v1/* endpoint.",
    )

    # TTS worker
    larynx_tts_mode: str = Field(default="mock")  # "mock" | "voxcpm"
    larynx_voxcpm_gpu: int = 0
    larynx_voxcpm_model: str = "openbmb/VoxCPM2"
    larynx_voxcpm_inference_timesteps: int = 10
    larynx_default_sample_rate: int = 24000

    # STT worker (Fun-ASR-Nano + Fun-ASR-MLT-Nano)
    larynx_stt_mode: str = Field(default="mock")  # "mock" | "funasr"
    larynx_funasr_gpu: int = 1  # GPU 1 by PRD §6
    larynx_funasr_gpu_mem_util: float = 0.4

    # VAD + Punctuation worker
    larynx_vad_punc_mode: str = Field(default="mock")  # "mock" | "real"
    larynx_vad_model: str = "fsmn-vad"
    larynx_punc_model: str = "ct-punc"

    # LLM (OpenRouter) — used by the M5 conversational loop only. Tests
    # inject their own client; leaving the key blank is fine for the REST
    # and streaming-STT/TTS endpoints.
    larynx_openrouter_api_key: str = Field(default="")
    larynx_openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    larynx_openrouter_http_referer: str = Field(default="")
    larynx_openrouter_x_title: str = Field(default="larynx")
    larynx_llm_default_model: str = Field(default="anthropic/claude-haiku-4.5")

    # Storage
    database_url: str = "postgresql+psycopg://larynx:larynx@localhost:5433/larynx"
    redis_url: str = "redis://localhost:6380/0"
    # Persistent data root — reference audio and latent caches live here.
    larynx_data_dir: str = "./data"

    # Latent cache
    # TTL for the Redis tier. Disk is canonical + permanent; Redis evicts LRU
    # and expires to keep hot voices in RAM without unbounded growth.
    larynx_latent_cache_ttl_s: int = 3600

    # Voice design preview TTL — ephemeral previews are cleaned up after this
    # window unless saved. See POST /v1/voices/design/{preview_id}/save.
    larynx_voice_design_ttl_s: int = 900

    # M7 fine-tuning. ORPHAN_JOB_GRACE (per ORCHESTRATION-M7.md §3.4
    # E-LoRA-2): a non-terminal job older than this at boot belongs to a
    # previous gateway process and gets marked FAILED(gateway_restarted).
    larynx_ft_orphan_grace_seconds: int = 60

    # Logging
    larynx_log_level: str = "INFO"
    larynx_log_json: bool = True

    # Gateway bind
    larynx_host: str = "0.0.0.0"
    larynx_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
