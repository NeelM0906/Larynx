"""STT smoke test for FunAudioLLM/Fun-ASR-Nano-2512 via Fun-ASR-vllm."""
from __future__ import annotations

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).parent
FUNASR_DIR = THIS_DIR / "Fun-ASR-vllm"
sys.path.insert(0, str(FUNASR_DIR))

from model import FunASRNano  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL_DIR = "FunAudioLLM/Fun-ASR-Nano-2512"
VLLM_REPO = "yuekai/Fun-ASR-Nano-2512-vllm"
WAV = THIS_DIR / "smoke_tts.wav"


def main() -> None:
    if not WAV.exists():
        raise SystemExit(f"missing input wav: {WAV}. Run smoke_tts.py first.")

    load_t0 = time.perf_counter()
    m, kwargs = FunASRNano.from_pretrained(model=MODEL_DIR, device="cuda:0")
    m.eval()
    vllm = LLM(
        model=VLLM_REPO,
        enable_prompt_embeds=True,
        gpu_memory_utilization=0.4,
        dtype="bfloat16",
    )
    m.vllm = vllm
    m.vllm_sampling_params = SamplingParams(top_p=0.001, max_tokens=500)
    load_s = time.perf_counter() - load_t0
    print(f"[stt] model+vllm loaded in {load_s:.2f}s")

    infer_t0 = time.perf_counter()
    res = m.inference(data_in=[str(WAV)], language="英文", **kwargs)
    infer_s = time.perf_counter() - infer_t0

    text = res[0][0]["text"]
    print(f"[stt] transcript: {text!r}")
    print(f"[stt] wall-clock infer: {infer_s:.2f}s  load: {load_s:.2f}s  total: {load_s + infer_s:.2f}s")


if __name__ == "__main__":
    main()
