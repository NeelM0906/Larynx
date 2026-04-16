# Larynx Smoke Test Report

Date: 2026-04-16
Host: Linux 6.17 / Python 3.12.3 / uv 0.11.7
Driver: NVIDIA 580.126.09 (CUDA 13.0)

## Verdict

End-to-end pipeline runs on 2× RTX PRO 6000 Blackwell Max-Q (sm_120, 95 GiB each):
TTS (VoxCPM2 via nano-vllm-voxcpm) and STT (Fun-ASR-Nano-2512 via Fun-ASR-vllm)
both load, execute CUDA kernels on Blackwell, and produce output in a single venv.

**Caveat**: Fun-ASR-Nano-2512 returns a garbled transcript for VoxCPM2 English
audio — see "Open questions" below. Hardware/plumbing is fine; model/prompt
choice may need follow-up.

## GPU Detected by Torch

```
torch 2.9.0+cu128
cuda_available True
device_count 2
  gpu0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition sm_120 mem=95.0GiB
  gpu1: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition sm_120 mem=95.0GiB
```

## Package Versions (final env)

| package           | version                                 |
|-------------------|-----------------------------------------|
| nano-vllm-voxcpm  | 2.0.0                                   |
| vllm              | 0.13.0                                  |
| torch             | 2.9.0+cu128                             |
| torchaudio        | 2.9.0                                   |
| flash-attn        | 2.8.3 (+cu12torch2.9cxx11abiTRUE prebuilt wheel) |
| funasr            | 1.3.1                                   |
| fun-asr-vllm      | *repo clone only*, no pypi package      |
| transformers      | 4.57.6                                  |
| triton            | 3.5.0                                   |
| numpy             | 2.2.6                                   |
| soundfile         | 0.13.1                                  |

Full `uv pip list` captured in `logs/final_versions.txt`.

## TTS Result — `smoke_tts.py`

- Model: `openbmb/VoxCPM2` via `nanovllm_voxcpm.VoxCPM.from_pretrained`
- Input text: `"Hello from the voice platform smoke test."`
- Output: `smoke_tts.wav` — WAVE, 16-bit PCM, mono, 16000 Hz, 145920 samples (9.12 s)
- Wall-clock (second run, weights cached): **generate 1.79 s, load 15.82 s, total 17.62 s**
- First run (with HF download): generate 1.01 s, load 58.94 s, total 59.95 s

## STT Result — `smoke_stt.py`

- Model: `FunAudioLLM/Fun-ASR-Nano-2512` (download via FunASR/ModelScope)
- vLLM weights: `yuekai/Fun-ASR-Nano-2512-vllm`, `gpu_memory_utilization=0.4`, bfloat16
- Attention backend selected by vLLM on Blackwell: `FLASH_ATTN`
- Transcript (`language="英文"`):  
  `"As for the dish, cooks are ashamed to taste it."`
- Wall-clock: **infer 25.06 s (first call, includes CUDA graph capture), load 229.01 s, total 254.07 s**
- Follow-up call in same process with default prompt (no `language` override) also returned garbled English: `"As for the dish, cooks the dish like the fish."` — infer 0.26 s once graphs were warm.

Both runs were streamed on one GPU (cuda:0); gpu1 was idle.

## Issues Encountered and Resolutions

### 1. `uv` was not on `PATH` despite user saying it was installed

`uv`, `uvx` not found anywhere on the system. Installed via
`curl -LsSf https://astral.sh/uv/install.sh | sh` → uv 0.11.7 in `~/.local/bin`.

### 2. `flash-attn==2.8.3` does not declare `torch` as a build dep

Fresh uv build failed with `ModuleNotFoundError: No module named 'torch'`
inside the PEP 517 isolated build. Fixed by installing torch first, then
running `uv pip install --no-build-isolation flash-attn==2.8.3`. Kept this
approach for every subsequent rebuild.

### 3. No CUDA toolkit installed on host — `nvcc` missing

Host has only the NVIDIA driver (580.126.09) / driver CUDA 13.0; no apt
toolkit, no passwordless sudo. Pulled toolkit components from PyPI:
`nvidia-cuda-nvcc`, `nvidia-nvvm`, `nvidia-cuda-cccl` — put `nvcc`,
`cicc`, `ptxas` at
`.venv/lib/python3.12/site-packages/nvidia/cu13/{bin,nvvm/bin}`.

### 4. `fatal error: nv/target: No such file or directory`

`nv/target` ships with libcu++ (CCCL), not with the `nvidia-cuda-nvcc`
package. Installed `nvidia-cuda-cccl` to drop the header into the same
`nvidia/cu13/include/` tree. Fixed.

### 5. `CUDA compiler and CUDA toolkit headers are incompatible`

First cu13 toolchain pulled `nvcc`/`cccl` **13.2.78** but the CUDA
runtime headers shipped with torch 2.11.0 were `CUDART_VERSION 13000` (13.0).
CCCL's `cuda_toolkit.h` enforces equal minor version. Downgraded to
`nvidia-cuda-nvcc==13.0.*` and `nvidia-cuda-cccl==13.0.*`.

### 6. `ptxas fatal : Unsupported .version 9.2; current version is '9.0'`

After fixing CCCL, cicc was still emitting PTX 9.2 (CUDA 13.2) while
ptxas expected 9.0 (CUDA 13.0). Cause: `nvidia-nvvm` and
`nvidia-cuda-crt` were still at **13.2.78** (they provide cicc and the
device CRT). Downgraded both to `==13.0.*`. Build then went past cicc.

### 7. `Killed` during nvcc compilation (OOM with MAX_JOBS=16)

flash-attn's `setup.py` ignores `TORCH_CUDA_ARCH_LIST` and uses its own
`FLASH_ATTN_CUDA_ARCHS` (default `80;90;100;120`). With 4 arches × 4 nvcc
threads × MAX_JOBS=16, template-heavy bwd kernels (`hdim256_*`) thrashed
RAM enough to get OOM-killed even with 251 GiB free. Fixed by setting
`FLASH_ATTN_CUDA_ARCHS=120` (Blackwell only) and `MAX_JOBS=4`. Build
completed for sm_120.

### 8. `ld: cannot find -lcudart`

nvidia pypi packages ship `libcudart.so.13` but not the unversioned
`libcudart.so` soname, and they put libs in `lib/` not `lib64/`. Created
`libcudart.so -> libcudart.so.13` and `lib64 -> lib` symlinks under
`nvidia/cu13/`. Final link succeeded.

### 9. vllm==0.13.0 force-downgrades torch 2.11.0 → 2.9.0+cu128

Fun-ASR-vllm's `requirements.txt` pins `vllm==0.13.0`, and vllm 0.13.0
hard-pins `torch==2.9.0`. Installing those downgraded torch from 2.11.0
(+cu130) to 2.9.0 (+cu128). The flash-attn wheel we built was linked
against torch 2.11 and libcudart.so.13, so it refused to import under
torch 2.9 / cu12.8.

Decision (not a workaround): rather than chase nvcc-12.8 to rebuild
from source (the `nvidia-cuda-nvcc-cu12==12.8.*` PyPI package ships
`ptxas` but **no `nvcc` binary**), I installed the upstream prebuilt
wheel `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
from Dao-AILab's GitHub Releases. It imports cleanly and
`flash_attn_func` runs a live kernel on sm_120 (verified).

Per the task brief, I did **not** silently downgrade vllm to dodge the
torch pin; the vllm version stayed at the Fun-ASR-pinned 0.13.0.

## Open Questions / Follow-Ups

- **STT accuracy**: Fun-ASR-Nano-2512 returns English-sounding but wholly
  wrong transcripts for the VoxCPM2 utterance in both prompt modes. The
  model is marketed as primarily Chinese; the nano variant's English
  coverage may just be weak. Two things to check before the next stage:
  (a) sanity-listen to `smoke_tts.wav` — VoxCPM2 produced 9.12 s for a
  short English sentence, which is long, and (b) retest STT with a
  known-clean English clip to isolate TTS vs. STT.
- **flash-attn from-source build on Blackwell**: the ABI-matching wheel
  side-stepped the cu128 toolkit gap, but if we ever need a custom
  flash-attn build in this stack we'll need `nvcc` 12.8 from somewhere
  other than pypi (apt toolkit, conda, or nvidia's .run installer).
- **ffmpeg**: already present at `/usr/bin/ffmpeg` — did not need
  `apt install -y ffmpeg`.

## Artifacts

```
~/larynx-smoke/
├── pyproject.toml
├── smoke_tts.py
├── smoke_stt.py
├── smoke_tts.wav                 # 16-bit PCM mono 16kHz, 9.12s
├── SMOKE_REPORT.md               # this file
├── Fun-ASR-vllm/                 # cloned for local `from model import FunASRNano`
└── logs/
    ├── install_flash_attn.log    # final (successful) build log
    ├── install_funasr_vllm.log
    ├── install_nano_vllm.log
    ├── install_nvcc.log
    ├── install_torch.log
    ├── smoke_tts_run.log
    ├── smoke_stt_run.log
    ├── final_versions.txt
    └── final_gpu.txt
```
