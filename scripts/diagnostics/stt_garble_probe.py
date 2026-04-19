"""bugs/003 — diagnostic probe matrix for M0 smoke STT garble.

Probes are orchestrated in a single process so TTS + STT models load
once and are reused across P1-P6 (big savings — Fun-ASR alone takes
~230s cold start including CUDA graph capture).

Run from the larynx-smoke venv (has nanovllm_voxcpm, vllm, funasr):

    /home/ripper/larynx-smoke/.venv/bin/python \
        /home/ripper/Desktop/Platform/scripts/diagnostics/stt_garble_probe.py \
        --probes p1,p2,p3

Outputs land in scripts/diagnostics/outputs/ (gitignored). That includes
generated WAVs and per-run log files.

See bugs/003_stt_m0_garble.md §4 for probe definitions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
M0_DIR = REPO_ROOT / "scripts" / "m0"
FUNASR_DIR = M0_DIR / "Fun-ASR-vllm"
OUT_DIR = REPO_ROOT / "scripts" / "diagnostics" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(FUNASR_DIR))

REFERENCE_TEXT = "Hello from the voice platform smoke test."
VOXCPM_MODEL = "openbmb/VoxCPM2"
FUNASR_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
FUNASR_VLLM_REPO = "yuekai/Fun-ASR-Nano-2512-vllm"

NORMALISE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ]
)

PASS_WER = 0.2
GARBLE_WER = 0.8


def wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    rn, hn = NORMALISE(ref), NORMALISE(hyp)
    if not rn.strip():
        return (0.0 if not hn.strip() else 1.0), 0.0
    return jiwer.wer(rn, hn), jiwer.cer(rn, hn)


_RUN_TS = time.strftime("%Y%m%d_%H%M%S")
_RUN_LOG = OUT_DIR / f"probe_run_{_RUN_TS}.log"


def log(msg: str = "") -> None:
    print(msg, flush=True)
    with _RUN_LOG.open("a") as f:
        f.write(msg + "\n")


def log_probe_header(pid: str, desc: str) -> None:
    log("")
    log("=" * 78)
    log(f"[{pid}] {desc}")
    log("=" * 78)


@dataclass
class Models:
    funasr: object
    funasr_kwargs: dict


def load_voxcpm():
    from nanovllm_voxcpm import VoxCPM

    t0 = time.perf_counter()
    voxcpm = VoxCPM.from_pretrained(
        model=VOXCPM_MODEL,
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.40,
    )
    log(f"[load] voxcpm ready in {time.perf_counter() - t0:.1f}s")
    return voxcpm


def release_voxcpm(voxcpm) -> None:
    import gc

    import torch

    try:
        voxcpm.stop()
    except Exception as e:
        log(f"[release] voxcpm.stop() raised: {e!r}")
    del voxcpm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("[release] voxcpm released")


def load_funasr() -> Models:
    from model import FunASRNano  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    t0 = time.perf_counter()
    funasr, funasr_kwargs = FunASRNano.from_pretrained(model=FUNASR_MODEL, device="cuda:0")
    set_inference_mode = getattr(funasr, "eval")  # torch.nn.Module.eval()
    set_inference_mode()
    vllm = LLM(
        model=FUNASR_VLLM_REPO,
        enable_prompt_embeds=True,
        gpu_memory_utilization=0.40,
        dtype="bfloat16",
    )
    funasr.vllm = vllm
    funasr.vllm_sampling_params = SamplingParams(top_p=0.001, max_tokens=500)
    log(f"[load] funasr+vllm ready in {time.perf_counter() - t0:.1f}s")

    return Models(funasr=funasr, funasr_kwargs=funasr_kwargs)


def synth_voxcpm(voxcpm, text: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for chunk in voxcpm.generate(target_text=text):
        chunks.append(np.asarray(chunk).reshape(-1))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), samples, samplerate=int(sample_rate))


def inspect_wav(path: Path) -> dict:
    info = sf.info(str(path))
    with sf.SoundFile(str(path)) as f:
        data = f.read(dtype="float32", always_2d=False)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(data**2))) if data.size else 0.0
    rms_db = 20 * np.log10(rms) if rms > 0 else float("-inf")
    return {
        "path": str(path),
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "frames": info.frames,
        "duration_s": info.frames / info.samplerate if info.samplerate else 0.0,
        "peak": peak,
        "rms_dbfs": rms_db,
    }


def transcribe_file(models: Models, wav_path: Path, language: str | None = "英文") -> str:
    kwargs = dict(models.funasr_kwargs)
    if language is not None:
        kwargs["language"] = language
    res = models.funasr.inference(data_in=[str(wav_path)], **kwargs)
    return res[0][0]["text"]


def probe_p1_synth(voxcpm) -> dict:
    """Phase-1 half of P1: run VoxCPM2 as-shipped; write WAV; no transcribe yet."""
    log_probe_header("P1-synth", "Re-run as-shipped TTS side, write WAV (transcribe in phase 2)")
    sr_guess = (
        getattr(voxcpm, "sample_rate", None)
        or getattr(voxcpm, "sr", 16000)
    )
    t0 = time.perf_counter()
    samples = synth_voxcpm(voxcpm, REFERENCE_TEXT)
    gen_s = time.perf_counter() - t0
    wav = OUT_DIR / f"p1_as_shipped_{_RUN_TS}.wav"
    write_wav(wav, samples, int(sr_guess))
    meta = inspect_wav(wav)
    log(f"  synth: {gen_s:.2f}s  samples={len(samples)}  declared_sr={sr_guess}")
    log(f"  wav meta: {json.dumps(meta, indent=2)}")
    return {"probe": "P1", "wav": str(wav), "meta": meta, "synth_s": gen_s, "declared_sr_used": int(sr_guess)}


def probe_p1_transcribe(models: Models, p1_synth_result: dict) -> dict:
    """Phase-2 half of P1: transcribe the already-written WAV."""
    log_probe_header("P1-transcribe", "Transcribe the as-shipped WAV with FunASR (language=英文)")
    wav = Path(p1_synth_result["wav"])
    t0 = time.perf_counter()
    hyp = transcribe_file(models, wav, language="英文")
    infer_s = time.perf_counter() - t0
    wer, cer = wer_cer(REFERENCE_TEXT, hyp)
    log(f"  transcript: {hyp!r}")
    log(f"  WER={wer:.3f}  CER={cer:.3f}  infer={infer_s:.2f}s")
    status = "PASS (bug gone)" if wer <= PASS_WER else ("GARBLED (bug reproduced)" if wer >= GARBLE_WER else "INTERMEDIATE")
    log(f"  verdict: {status}")
    return {**p1_synth_result, "hyp": hyp, "wer": wer, "cer": cer, "status": status}


def probe_p2(voxcpm, p1_synth_result: dict) -> dict:
    log_probe_header("P2", "Metadata probe — VoxCPM server introspection + WAV header")
    v = voxcpm
    log(f"  type(server).__mro__ = {[c.__name__ for c in type(v).__mro__]}")
    interesting = [a for a in dir(v) if not a.startswith("_") and any(k in a.lower() for k in ("rate", "sr", "fs", "freq", "hz"))]
    log(f"  sr-ish attributes: {interesting}")
    for attr in ("sample_rate", "sr", "output_sample_rate", "fs", "audio_sample_rate"):
        if hasattr(v, attr):
            try:
                log(f"  server.{attr} = {getattr(v, attr)!r}")
            except Exception as e:
                log(f"  server.{attr} access raised: {e!r}")
    from huggingface_hub import try_to_load_from_cache

    for candidate in ("config.json", "vocoder/config.json", "decoder/config.json"):
        hit = try_to_load_from_cache(repo_id=VOXCPM_MODEL, filename=candidate)
        if hit:
            try:
                cfg = json.loads(Path(hit).read_text())
                keys_of_interest = {k: vv for k, vv in cfg.items() if any(s in k.lower() for s in ("rate", "sr", "sampl", "hz", "freq"))}
                log(f"  {candidate} sr-keys: {json.dumps(keys_of_interest, indent=2) if keys_of_interest else '(none)'}")
            except Exception as e:
                log(f"  {candidate} parse failed: {e!r}")
    declared = p1_synth_result["meta"]["samplerate"]
    duration = p1_synth_result["meta"]["duration_s"]
    words = len(REFERENCE_TEXT.split())
    expected_duration = 0.3 * words
    stretch = duration / expected_duration if expected_duration else 0.0
    log(f"  wav declared_sr={declared} Hz  duration={duration:.2f}s")
    log(f"  expected natural duration≈{expected_duration:.2f}s  observed stretch≈{stretch:.2f}×")
    candidate_true_rate = int(declared * stretch)
    log(f"  candidate true rate if mislabelled: ≈{candidate_true_rate} Hz "
        f"(compare against 16000 / 24000 / 48000)")
    return {
        "probe": "P2",
        "declared_sr": declared,
        "stretch": stretch,
        "candidate_true_rate": candidate_true_rate,
    }


def probe_p3(models: Models, p1_result: dict, p2_result: dict) -> dict:
    log_probe_header("P3", "Resample variants: reinterpret + resample to 16k, transcribe each")
    src_wav = Path(p1_result["wav"])
    samples, declared_sr = sf.read(str(src_wav), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    log(f"  source WAV: {src_wav.name}  declared_sr={declared_sr}  samples={len(samples)}")

    variants: list[dict] = []
    variants.append({"label": "A_as_shipped", "wav": src_wav, "note": "declared sr, no resample"})

    for true_sr in (24000, 48000):
        if declared_sr == true_sr:
            continue
        resampled = librosa.resample(samples.astype(np.float32), orig_sr=true_sr, target_sr=16000, res_type="kaiser_best")
        resampled_wav = OUT_DIR / f"p3_from{true_sr}_to16k_{_RUN_TS}.wav"
        write_wav(resampled_wav, resampled, 16000)
        variants.append({"label": f"B_relabel_{true_sr}_then_16k", "wav": resampled_wav, "note": f"interpret as {true_sr} Hz, resample to 16k"})

    if declared_sr != 16000:
        honest = librosa.resample(samples.astype(np.float32), orig_sr=declared_sr, target_sr=16000, res_type="kaiser_best")
        wav_c = OUT_DIR / f"p3_honest_downsample_{_RUN_TS}.wav"
        write_wav(wav_c, honest, 16000)
        variants.append({"label": "C_honest_downsample", "wav": wav_c, "note": f"resample declared {declared_sr}→16k"})

    results = []
    for variant in variants:
        t0 = time.perf_counter()
        hyp = transcribe_file(models, variant["wav"], language="英文")
        wer, cer = wer_cer(REFERENCE_TEXT, hyp)
        dt = time.perf_counter() - t0
        log(f"  [{variant['label']:30s}] {variant['note']}")
        log(f"      transcript: {hyp!r}")
        log(f"      WER={wer:.3f}  CER={cer:.3f}  infer={dt:.2f}s")
        results.append({"label": variant["label"], "wav": str(variant["wav"]), "hyp": hyp, "wer": wer, "cer": cer, "note": variant["note"]})

    as_shipped_wer = next(r["wer"] for r in results if r["label"].startswith("A_"))
    candidates = [r for r in results if r["label"].startswith(("B_", "C_"))]
    best_resampled = min(candidates, key=lambda r: r["wer"]) if candidates else None
    h2_confirmed = (as_shipped_wer >= GARBLE_WER) and (best_resampled is not None) and (best_resampled["wer"] <= PASS_WER)
    log("")
    log(f"  as-shipped WER = {as_shipped_wer:.3f}")
    if best_resampled:
        log(f"  best resampled variant: {best_resampled['label']} WER={best_resampled['wer']:.3f}")
    log(f"  H2 early-stop condition met: {h2_confirmed}")
    return {"probe": "P3", "results": results, "h2_confirmed": h2_confirmed, "best_resampled": best_resampled}


def probe_p4(models: Models) -> dict:
    log_probe_header("P4", "Human-speech control — Fun-ASR on its shipped en.mp3")
    kwargs_iter = dict(models.funasr_kwargs)
    model_path = kwargs_iter.get("model_path") or kwargs_iter.get("model_dir")
    candidates = []
    if model_path:
        candidates.append(Path(model_path) / "example" / "en.mp3")
    candidates.append(Path.home() / ".cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/example/en.mp3")
    en_mp3 = next((p for p in candidates if p.exists()), None)
    if en_mp3 is None:
        log(f"  SKIP: en.mp3 not found. Searched: {[str(c) for c in candidates]}")
        return {"probe": "P4", "status": "SKIPPED", "reason": "en.mp3 not found"}
    log(f"  input: {en_mp3}")
    hyp = transcribe_file(models, en_mp3, language="英文")
    log(f"  transcript: {hyp!r}")
    words = hyp.lower().split()
    short_words = {"the", "a", "an", "of", "to", "in", "is", "and", "it", "that", "for"}
    has_short = any(w.strip(".,!?") in short_words for w in words)
    log(f"  contains common English connector word: {has_short}")
    return {"probe": "P4", "input": str(en_mp3), "hyp": hyp, "coherent": has_short, "status": "DONE"}


def probe_p5(models: Models, p1_result: dict) -> dict:
    log_probe_header("P5", "Language-tag ablation on today's WAV (英文 / None / 中文)")
    wav = Path(p1_result["wav"])
    rows = []
    for lang_label, lang_arg in [("英文(en)", "英文"), ("None(auto)", None), ("中文(zh)", "中文")]:
        hyp = transcribe_file(models, wav, language=lang_arg)
        wer, cer = wer_cer(REFERENCE_TEXT, hyp)
        log(f"  lang={lang_label:12s}  WER={wer:.3f}  CER={cer:.3f}  text={hyp!r}")
        rows.append({"lang_label": lang_label, "lang_arg": lang_arg, "hyp": hyp, "wer": wer, "cer": cer})
    return {"probe": "P5", "rows": rows}


def probe_p6(p1_result: dict, p2_result: dict) -> dict:
    log_probe_header("P6", "Peak / RMS / spectral-centroid inspection")
    wav = Path(p1_result["wav"])
    samples, sr_declared = sf.read(str(wav), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples**2)))
    rms_db = 20 * np.log10(rms) if rms > 0 else float("-inf")
    dc = float(np.mean(samples))
    centroid_declared = float(np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr_declared)))
    log(f"  at declared sr={sr_declared}:  peak={peak:.4f}  rms_dbfs={rms_db:.2f}  dc={dc:+.5f}  centroid_hz={centroid_declared:.1f}")
    true_rate = p2_result.get("candidate_true_rate")
    if true_rate and true_rate != sr_declared and true_rate in (24000, 48000):
        centroid_true = float(np.mean(librosa.feature.spectral_centroid(y=samples, sr=true_rate)))
        log(f"  at candidate true sr={true_rate}:  centroid_hz={centroid_true:.1f} (speech typically 1000-3000 Hz)")
    return {"probe": "P6", "peak": peak, "rms_dbfs": rms_db, "dc": dc, "centroid_declared_hz": centroid_declared}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="p1,p2,p3", help="Comma-separated probes to run (p1..p6)")
    ap.add_argument("--continue-after-p3", action="store_true", help="Ignore early-stop rule; run everything requested")
    args = ap.parse_args()
    wanted = [p.strip().lower() for p in args.probes.split(",") if p.strip()]

    log(f"=== stt_garble_probe.py — bugs/003 ===")
    log(f"timestamp: {_RUN_TS}")
    log(f"log file: {_RUN_LOG}")
    log(f"outputs: {OUT_DIR}")
    log(f"probes requested: {wanted}")

    results: dict[str, dict] = {}

    # Phase 1: TTS-side probes (VoxCPM2 loaded, FunASR not yet).
    voxcpm = load_voxcpm()
    p1_synth = probe_p1_synth(voxcpm) if "p1" in wanted else None
    if p1_synth:
        results["p1"] = p1_synth
    p2 = probe_p2(voxcpm, p1_synth) if "p2" in wanted and p1_synth else None
    if p2:
        results["p2"] = p2
    release_voxcpm(voxcpm)

    # Phase 1.5: CPU-only probes that don't need any model.
    if "p6" in wanted and p1_synth and p2:
        results["p6"] = probe_p6(p1_synth, p2)

    # Phase 2: STT-side probes (FunASR+vLLM loaded).
    need_funasr = any(p in wanted for p in ("p1", "p3", "p4", "p5"))
    models: Models | None = None
    if need_funasr:
        models = load_funasr()

    if p1_synth and models is not None:
        results["p1"] = probe_p1_transcribe(models, p1_synth)

    p3 = None
    if "p3" in wanted and results.get("p1") and p2 and models is not None:
        p3 = probe_p3(models, results["p1"], p2)
        results["p3"] = p3

    early_stop = False
    if p3 and not args.continue_after_p3 and p3.get("h2_confirmed"):
        log("")
        log("EARLY-STOP: H2 confirmed by P3 (as-shipped garbles, resampled variant clean).")
        log("P7 was already in the skip-list by design. Running P4+P5 for completeness.")
        early_stop = True

    run_p4 = "p4" in wanted or early_stop
    run_p5 = "p5" in wanted or early_stop
    if run_p4 and models is not None:
        results["p4"] = probe_p4(models)
    if run_p5 and models is not None and results.get("p1"):
        results["p5"] = probe_p5(models, results["p1"])

    log("")
    log("=" * 78)
    log("SUMMARY")
    log("=" * 78)
    log(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in results.items()}, indent=2, default=str))

    summary_path = OUT_DIR / f"probe_summary_{_RUN_TS}.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
