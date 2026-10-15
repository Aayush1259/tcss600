# What we did on the lab machine, explained simply

Think of tonight like this: you got the keys to a powerful computer in your professor's lab, and we (1) got the keys working, (2) looked at what the computer has, (3) installed the AI program, (4) downloaded an AI brain, and (5) tested how fast it works.

## 1. Getting the keys (Tailscale)

Tailscale creates a private network over the internet. Only invited devices can see each other, like a private club. You installed it on your laptop and logged in with your personal email. The lab computer (`bosgame`) appeared in the club as "Connected" with the address `100.81.30.97`.

## 2. Walking into the computer (SSH)

SSH (Secure Shell) lets you type commands on another computer as if you were sitting in front of it.

`ssh localai@100.81.30.97` means: "connect as the user named `localai` to the computer at address `100.81.30.97`."

- We used the number address because your laptop didn't recognize the nickname `bosgame`.
- The "are you sure you want to continue connecting" question is a first-time safety check. You answer `yes` once, and your laptop remembers the computer from then on.

## 3. Looking around (inspection commands)

- `nvidia-smi` asks "which NVIDIA graphics card is here?" The answer "command not found" told us there is no NVIDIA card at all.
- `lspci | grep ...` lists the computer's hardware and filters for graphics-related lines. It revealed an AMD Strix Halo chip with Radeon 8060S graphics.
- `free -h` shows memory: 30 GB total, 25 GB free. (`-h` means show it in GB instead of raw bytes.)
- `nproc` counts the processor cores: 32.
- `df -h` shows disk space: 1.7 TB free, plenty.
- `ollama list` lists downloaded AI models: none yet, at that point.
- `who` shows who else is logged in; `uptime` shows how busy the machine is (load 0.19 = basically asleep).

## 4. Installing the AI runner (Ollama)

Ollama is the program that loads AI models and answers questions with them. The official installer needed an admin (`sudo`) password, which we don't have, so we installed it inside your own home folder (`~/ollama`). Nothing outside your folder was touched, so the PhD students' setups are unaffected.

- `curl -L <address> -o <filename>` downloads a file. `-L` means "follow redirects", `-o` means "save it under this name".
- Things that went wrong along the way:
  - The old download address returned 404 ("page not found") because Ollama renamed their files from `.tgz` to `.tar.zst`.
  - The GitHub download server is blocked on the lab network.
  - A typo: `release` instead of `releases` in one attempt.
  - The graphics-driver file (`-rocm`, 1.1 GB) turned out to contain only drivers, not the Ollama program itself, so we additionally downloaded the main program file (1.4 GB).
- `mkdir -p ollama` creates a folder (and stays quiet if it already exists).
- `tar --use-compress-program=unzstd -xf file -C folder` unpacks a compressed archive into a folder, like unzipping a zip file.
- `~/ollama/bin/ollama --version` asked the program its version: 0.40.3. That proved the install worked.
- ROCm is AMD's toolkit that lets AI programs use AMD graphics chips (NVIDIA's equivalent is called CUDA).

## 5. Keeping the program running (tmux) and starting the server

- `tmux new -s ollama` opens a terminal session that survives even if your laptop sleeps or disconnects. Like leaving a program running in a locked room you can walk back into later.
- `Ctrl-B` then `D` steps out of the room (the program keeps running). `tmux attach -t ollama` walks back in.
- `~/ollama/bin/ollama serve` starts Ollama's server: a program that sits and waits for requests at port 11434 (think of a port as a numbered door on the computer). Its startup messages confirmed it found the Radeon 8060S graphics chip through ROCm and will use it for AI math.

## 6. Downloading the AI brain

- `~/ollama/bin/ollama pull qwen3.8:27b` downloads the model: Qwen 3.8 with 27 billion parameters (parameters are roughly the model's brain cells), 17 GB in size. Downloading is only network traffic, so it bothered nobody.

## 7. The smoke test (one question through the model)

We used `curl` to knock on Ollama's door (`http://localhost:11434/api/generate`; `localhost` means "this same computer") and hand it a note written in JSON, a structured format:

- `model`: which AI brain to use.
- `prompt`: the question plus the patient context.
- `stream: false`: wait and give the whole answer at once instead of dripping it out word by word.
- `options.num_ctx: 8192`: never try to hold more than 8192 tokens in memory at once. This protects our 30 GB of RAM, because the server otherwise assumed 96 GB and would try something huge.
- `logprobs: true`: also return confidence scores for each word. Your RQ1 research question needs these scores, so we confirmed they exist.

The reply came back with `response` (the answer: a correct table), `thinking` (the model's scratch-pad reasoning), and timings: about 11 seconds total, roughly 29 tokens per second.

## 8. Confirming the graphics chip does the work

`~/ollama/bin/ollama ps` shows what's currently loaded: the model, 17 GB, **100% GPU**. The whole model sits on the graphics chip; the main processor does almost nothing.

## 9. The speed test (throughput probe, the Python script)

Line by line, in plain words:

- `import json, time, urllib.request, random` loads four helper toolboxes: `json` (read and write structured notes), `time` (a stopwatch), `urllib.request` (knock on web doors from inside Python), `random` (dice).
- `random.seed(7)` fixes the dice so the fake data is identical on every run.
- The first loop writes 8 fake patient files (`/tmp/chunk0.txt` through `/tmp/chunk7.txt`; `/tmp` is scratch space). Each file has 45 visit notes, about 665 words, with random medications, dates, and blood pressures.
- `t0 = time.time()` starts the stopwatch.
- The second loop sends each chunk to the model with the instruction "list every medication change with its date", waits for the answer, and discards it. Here we measured speed only, not quality.
- Then simple math: total seconds divided by 8 = seconds per call; times 48 calls = minutes per question; times 200 = hours for the full set.
- Result: 99.1 seconds per call, which extrapolates to 264 hours. Way over the 42-hour budget. The main cost was the model's `thinking` scratch-pad: thousands of extra tokens per call.

## 10. Turning off the thinking

We ran the same chunk twice: once with `"think": true`, once with `"think": false`.

- Thinking on: 84.1 seconds, 2850 tokens generated.
- Thinking off: 28.8 seconds, 974 tokens, zero thinking characters.

Three times faster. The 974 tokens left are the real answer (45 medication events listed). New estimate: about 77 hours at the most pessimistic assumptions, and it fits inside 42 hours once we ask for shorter answers and use the real (smaller) chunk and subtask counts from LongHealth.

## Decisions we locked in

- Every worker call uses `think: false`, with context capped at 8192. We will double-check on the 20 warmup questions that answers stay correct without thinking.
- Installing and downloading never bothered anyone (your folder only; network and disk only). Running the model is the heavy part, so real experiment runs wait for agreed time windows with Adeline and Shutong.
