"""TTS smoke test for VoxCPM2 via nano-vllm-voxcpm."""
from __future__ import annotations

import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from nanovllm_voxcpm import VoxCPM

TEXT = "Hello from the voice platform smoke test."
OUT = Path(__file__).parent / "smoke_tts.wav"

# Fun-ASR-Nano consumes 16 kHz mono. Resample VoxCPM2's native output
# here so smoke_stt.py sees audio at the rate its frontend expects —
# mirrors the gateway's _wav_to_pcm16_16k path in
# test_real_model_stream.py. bugs/003 traced the earlier garbled
# transcripts to a getattr() fallback that labelled the WAV as 16 kHz
# while the samples were at VoxCPM2's native 48 kHz.
TARGET_SR = 16000


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

    info = server.get_model_info()
    native_sr = int(info["output_sample_rate"])
    print(f"[tts] VoxCPM2 native output rate: {native_sr} Hz")

    gen_t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    for chunk in server.generate(target_text=TEXT):
        chunks.append(np.asarray(chunk).reshape(-1))
    gen_s = time.perf_counter() - gen_t0

    native_wav = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float32)
    if native_sr != TARGET_SR and native_wav.size:
        # soxr_hq is librosa's bundled high-quality polyphase resampler —
        # no extra dep (kaiser_* would pull in resampy).
        wav = librosa.resample(
            native_wav.astype(np.float32),
            orig_sr=native_sr,
            target_sr=TARGET_SR,
            res_type="soxr_hq",
        )
    else:
        wav = native_wav

    sf.write(str(OUT), wav, samplerate=TARGET_SR)

    dur_s = len(wav) / TARGET_SR if TARGET_SR else 0.0
    print(
        f"[tts] generated {len(native_wav)} samples @ {native_sr} Hz → "
        f"{len(wav)} samples @ {TARGET_SR} Hz ({dur_s:.2f}s audio)"
    )
    print(f"[tts] wall-clock generate: {gen_s:.2f}s  load: {load_s:.2f}s  total: {gen_s + load_s:.2f}s")
    print(f"[tts] wrote {OUT}")

    try:
        server.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
