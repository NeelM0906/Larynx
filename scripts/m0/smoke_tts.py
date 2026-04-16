"""TTS smoke test for VoxCPM2 via nano-vllm-voxcpm."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from nanovllm_voxcpm import VoxCPM

TEXT = "Hello from the voice platform smoke test."
OUT = Path(__file__).parent / "smoke_tts.wav"


def main() -> None:
    load_t0 = time.perf_counter()
    server = VoxCPM.from_pretrained(
        model="openbmb/VoxCPM2",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.80,
    )
    load_s = time.perf_counter() - load_t0
    print(f"[tts] model loaded in {load_s:.2f}s")

    sample_rate = getattr(server, "sample_rate", None) or getattr(server, "sr", 16000)

    gen_t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    for chunk in server.generate(target_text=TEXT):
        chunks.append(np.asarray(chunk).reshape(-1))
    gen_s = time.perf_counter() - gen_t0

    wav = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float32)
    sf.write(str(OUT), wav, samplerate=int(sample_rate))

    dur_s = len(wav) / float(sample_rate) if sample_rate else 0.0
    print(f"[tts] generated {len(wav)} samples @ {sample_rate} Hz ({dur_s:.2f}s audio)")
    print(f"[tts] wall-clock generate: {gen_s:.2f}s  load: {load_s:.2f}s  total: {gen_s + load_s:.2f}s")
    print(f"[tts] wrote {OUT}")

    try:
        server.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
