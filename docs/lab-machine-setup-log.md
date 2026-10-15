# Lab Machine Setup Log — bosgame

Dated log of everything done on the shared lab server. Keep appending; Prof. Cai wants these artifacts linked in the weekly reports.

## 2026-10-11 — First access (Tailscale + SSH)

- Installed Tailscale on the laptop, logged in as `aayushkmodi721b@gmail.com`. The `bosgame` machine (shared by Prof. Cai) showed up in the tailnet as Connected: `100.81.30.97`.
- `ssh localai@bosgame` did not resolve on Windows; connected via `ssh localai@100.81.30.97`. Accepted the host key on first connect. No password was needed.
- Hardware discovery (read-only commands: `lspci`, `free -h`, `nproc`, `df -h`):
  - AMD Strix Halo SoC, Radeon 8060S integrated graphics. **No NVIDIA GPU** (so `nvidia-smi` does not exist; this is expected, not an error).
  - AMD NPU present (not used).
  - 30 GB RAM (25 GB available), 32 CPU cores, 1.7 TB free disk.

## 2026-10-11/12 — Ollama install (user-space, no sudo)

- The official installer (`curl -fsSL https://ollama.com/install.sh | sh`) asks for a sudo password, which we do not have. Abandoned; installed user-local instead so nothing system-wide is touched.
- Download hurdles and fixes:
  - `https://ollama.com/download/ollama-linux-amd64.tgz` returns 404 (only `.tar.zst` is published now).
  - GitHub release asset CDN is blocked on the lab network (both ollama.com `/download/` single path and github `releases/download` returned "Not Found").
  - The working base URL (from the install script's `download_and_extract`, line 171) is `https://ollama.com/download` + `/{filename}.tar.zst`.
  - The ROCm supplement tarball (`ollama-linux-amd64-rocm.tar.zst`, 1.1 GB) contains **only GPU libraries**, not the Ollama binary. The base tarball (`ollama-linux-amd64.tar.zst`, 1.4 GB) must be downloaded and extracted too.
- Steps that worked:
  ```bash
  curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst -o ollama.tar.zst
  mkdir -p ollama && tar --use-compress-program=unzstd -xf ollama.tar.zst -C ollama
  curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama-base.tar.zst
  tar --use-compress-program=unzstd -xf ollama-base.tar.zst -C ollama
  ~/ollama/bin/ollama --version   # -> 0.40.3
  ```
- Server: `tmux new -s ollama`, then `~/ollama/bin/ollama serve` (detach with Ctrl-B, D). Listening on 127.0.0.1:11434.
- GPU detection (from server log): ROCm picked up the 8060S as an inference device (`library=ROCm compute=gfx1151 type=iGPU`). The Vulkan path drops the iGPU by default; ROCm is the working path.
- Caveat: the server reports 96 GB VRAM (unified-memory aperture, not the physical 30 GB) and defaults to `num_ctx=262144`. Real runs must cap context length (e.g. 8192) or the KV cache will exceed physical RAM.

## 2026-10-12 — 27B worker model

- `~/ollama/bin/ollama pull qwen3.8:27b` — tag exists, ~17 GB download, completed.

## 2026-10-12 — Smoke test (PASSED)

- `ollama ps`: qwen3.8:27b, 17 GB, **100% GPU**, context 8192, runner llamacpp. Model fully on the Radeon 8060S via ROCm.
- Test prompt (extract hemoglobin values + dates from a short context): correct answer, both values with dates, formatted as a table.
- Timings: total ~11.2s = load ~5.2s (one-time; model stays loaded) + prompt eval 61 tokens in 0.37s + generation 161 tokens in 5.6s → **~29 tok/s**.
- Note: this model emits a "thinking" chain-of-thought trace before the answer, adding tokens to every call. Factor into throughput and cost math.
- Log-probabilities: `logprobs: True` via `/api/generate` — the RQ1 uncertainty signal is available on this build (v0.40.3).
- Machine state during test: load avg 0.19 (idle); other sessions stale (`weicai` tty1 since Oct 5, `civic` pts/3 since Oct 9).

## 2026-10-12 — Throughput probe (PLAN REVISION NEEDED)

- 8 worker calls over synthetic 665-word clinical chunks (medication-extraction subtask, num_ctx=8192): 793s total → **99.1s per call**.
- Extrapolation at 6 subtasks × 8 chunks = 48 calls/question: **79.3 min/question → 264.3 hours for 200 questions**. The 42-hour lab plan does not survive this.
- Dominant cost driver: the model's chain-of-thought "thinking" trace (likely thousands of tokens per call at ~29 tok/s).
- Next: test `"think": false` via /api/generate. If per-call time collapses, the plan is salvageable with far fewer hours. Chunk-count (8) and subtask-count (6) assumptions are upper bounds; real LongHealth numbers may be lower, but the thinking trace must be addressed regardless.
- Result: `think=True` → 84.1s, 2850 generated tokens; `think=False` → **28.8s**, 974 generated tokens, zero thinking chars. 3x speedup confirmed; the remaining 974 tokens are the answer itself (45 medication events listed).
- Revised extrapolation at 28.8s/call, 48 calls/question: 23 min/question → **76.8h per 200 questions**. Still 1.8x over the 42h budget. Path to fit: (a) concise-output prompting to cut the ~974-token answers, (b) real chunk/subtask counts (likely below the 8×6 upper bound). At 24 calls/question × ~15s/call → ~20h. Decision: all worker calls use `think=False` (standardized protocol choice); verify answer quality holds on the 20-question warmup set.

## Impact on the shared machine (for Adeline / Shutong)

- Everything installed lives under `/home/localai` (~/ollama, ~/ollama-base.tar.zst, ~/ollama.tar.zst). No sudo used, nothing system-wide modified.
- Disk used: ~20 GB of 1.7 TB free. Negligible.
- Idle `ollama serve`: one listening process, negligible CPU/RAM.
- Model download: network bandwidth only, no GPU/RAM load.
- **Running inference is the heavy step**: a 27B model needs ~17-20 GB RAM plus the GPU. On this 30 GB machine that is most of the memory. Heavy runs (smoke test, throughput probe, experiments) must be scheduled in agreed time windows with Adeline and Shutong. Coordination email still to be sent.
