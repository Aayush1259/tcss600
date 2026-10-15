# Lab Machine Setup Log : bosgame

Dated log of everything I did on the shared lab server, October 11 to 14, 2026.

## 2026-10-11 : First access (Tailscale + SSH)

I installed Tailscale on my laptop and logged in with my personal email. The `bosgame` machine (shared with Adeline and Shutong in Prof. Cai's lab) showed up in the tailnet as Connected at `100.81.30.97`.

`ssh localai@bosgame` did not resolve on Windows, so I connected with the IP directly: `ssh localai@100.81.30.97`. I accepted the host key on first connect. No password was needed.

Hardware discovery (read-only commands: `lspci`, `free -h`, `nproc`, `df -h`):
- AMD Strix Halo SoC with Radeon 8060S integrated graphics. There is no NVIDIA GPU, so `nvidia-smi` does not exist. This is expected, not an error.
- AMD NPU is present but I did not use it.
- 30 GB RAM (25 GB available), 32 CPU cores, 1.7 TB free disk.

## 2026-10-11/12 : Ollama install (user-space, no sudo)

The official installer (`curl -fsSL https://ollama.com/install.sh | sh`) asks for a sudo password, which I do not have on this shared machine. I installed Ollama inside my own home folder instead, so nothing system-wide was touched.

Download problems I hit and how I fixed them:
- `https://ollama.com/download/ollama-linux-amd64.tgz` returns 404. Ollama now publishes `.tar.zst` files instead of `.tgz`.
- The GitHub release CDN is blocked on the lab network, so I could not download from GitHub directly.
- I found the working base URL in the install script's `download_and_extract` function: `https://ollama.com/download` plus `/{filename}.tar.zst`.
- The ROCm supplement tarball (`ollama-linux-amd64-rocm.tar.zst`, 1.1 GB) contains only GPU libraries, not the Ollama binary. I had to download the base tarball (`ollama-linux-amd64.tar.zst`, 1.4 GB) as well.

Steps that worked:
```bash
curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst -o ollama.tar.zst
mkdir -p ollama && tar --use-compress-program=unzstd -xf ollama.tar.zst -C ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama-base.tar.zst
tar --use-compress-program=unzstd -xf ollama-base.tar.zst -C ollama
~/ollama/bin/ollama --version   # -> 0.40.3
```

I started the server in a tmux session (`tmux new -s ollama`, then `~/ollama/bin/ollama serve`, detached with Ctrl-B, D). It listens on 127.0.0.1:11434.

From the server log I confirmed ROCm picked up the 8060S as an inference device (`library=ROCm compute=gfx1151 type=iGPU`). The Vulkan path drops the iGPU by default, so ROCm is the working path.

One caveat: the server reports 96 GB VRAM (this is the unified-memory aperture, not physical RAM) and defaults to `num_ctx=262144`. I cap context length on every real run (8192 for the speed tests, 16384 for pilots) so the KV cache never exceeds the physical 30 GB.

## 2026-10-12 : 27B worker model

I ran `~/ollama/bin/ollama pull qwen3.8:27b`. The tag exists; the download is about 17 GB and it completed.

## 2026-10-12 : Smoke test (passed)

`ollama ps` showed qwen3.8:27b loaded at 17 GB and 100% GPU with context 8192, running under llamacpp. The full model sits on the Radeon 8060S through ROCm.

I sent a test prompt asking for hemoglobin values and dates from a short clinical context. The answer was correct: both values with dates, formatted as a table.

Timings: about 11.2 seconds total. Roughly 5.2 seconds one-time model load (the model stays loaded after that), 0.37 seconds to process 61 prompt tokens, and 5.6 seconds to generate 161 tokens. That is about 29 tokens per second.

Two things I noted:
- This model writes a chain-of-thought "thinking" trace before its answer, which adds tokens to every call. I factored this into the throughput math.
- I confirmed `logprobs: true` works through `/api/generate`. The per-token confidence scores are available on this build (v0.40.3), which is what my RQ1 uncertainty signal needs.

Machine state during the test: load average 0.19 (idle). Other sessions were stale (`weicai` on tty1 since Oct 5, `civic` on pts/3 since Oct 9).

## 2026-10-12 : Throughput probe (plan revision needed)

I ran 8 worker calls over synthetic 665-word clinical chunks (a medication-extraction subtask, num_ctx=8192). Total was 793 seconds, or 99.1 seconds per call.

Extrapolated at 6 subtasks times 8 chunks = 48 calls per question, that is 79.3 minutes per question, or 264.3 hours for 200 questions. The 42-hour lab plan does not survive that.

The dominant cost was the model's chain-of-thought thinking trace, likely thousands of tokens per call at 29 tokens per second. So I tested `"think": false` through /api/generate:
- think on: 84.1s, 2850 generated tokens
- think off: 28.8s, 974 generated tokens, zero thinking characters

That is a 3x speedup. The remaining 974 tokens are the actual answer (45 medication events listed). I revised the extrapolation: at 28.8s per call and 48 calls per question, 23 minutes per question, or 76.8 hours for 200 questions. Still 1.8x over the 42-hour budget, but the path to fit is clear: (a) concise-output prompting to cut the 974-token answers, and (b) the real LongHealth chunk and subtask counts, which are likely below my 8x6 upper bound. At 24 calls per question and about 15 seconds per call, the total lands near 20 hours.

Decision I locked in: every worker call uses `think=False`, and I verify answer quality holds on the 20-question warmup set.

## Impact on the shared machine (for Adeline and Shutong)

Everything I installed lives under `/home/localai` (~/ollama and the two tarballs). I never used sudo and modified nothing system-wide.

- Disk used: about 20 GB out of 1.7 TB free. Negligible.
- Idle `ollama serve`: one listening process, negligible CPU and RAM.
- Model download: network bandwidth only, no GPU or RAM load.
- Running inference is the heavy step. A 27B model needs about 17 to 20 GB of RAM plus the GPU, which is most of this 30 GB machine. I schedule heavy runs (smoke test, throughput probe, experiments) in agreed time windows with Adeline and Shutong. The coordination email is still to be sent.
