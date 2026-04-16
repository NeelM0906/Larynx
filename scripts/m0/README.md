# M0 — Hardware smoke test

Archived artifacts from the M0 milestone (PRD §9): verifying that flash-attn,
nano-vllm-voxcpm, and Fun-ASR-vllm run on the RTX Pro 6000 Blackwell box.

This directory is **outside the uv workspace** — it has its own isolated
`pyproject.toml` / `uv.lock` with the heavy GPU dependencies that don't belong
in the monorepo root.

## Contents
- `smoke_tts.py` — single-shot VoxCPM2 synthesis
- `smoke_stt.py` — single-shot Fun-ASR transcription
- `smoke_tts.wav` — sample output from the TTS run
- `logs/` — install logs for flash-attn, torch, nano-vllm, Fun-ASR-vllm,
  and the smoke-run outputs
- `../../docs/m0_smoke_report.md` — findings writeup

## Re-running

```bash
cd scripts/m0
uv sync                       # installs from the local pyproject.toml
uv run python smoke_tts.py
uv run python smoke_stt.py
```
