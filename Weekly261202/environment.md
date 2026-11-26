# Environment

The machines, models and versions behind the results in this repository. Each version below is the one recorded in the run logs or read on the machine.

## Laptop: the 4B worker, cloud alone and every analysis

- Windows 11 with WSL Ubuntu; every script runs in WSL.
- Python 3.11.17 in a virtual environment; the packages the code imports are pinned in `requirements.txt`.
- Ollama 0.40.3 serving `qwen3:4b` (digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`), context 3,072 tokens, thinking off, temperature 0. 27 of the model's 37 layers fit in GPU memory; NVIDIA driver 621.24 from November 13 on.

## Lab machine (bosgame): the 27B worker

- 120 GB of unified memory.
- Ollama 0.40.3 serving `qwen3.8:27b` (digest `aaee06c39dcf2437cde036998d960e1fc1494b8191be7cc9657d01e509097813`, Q4_K_M), thinking off, temperature 0, with the whole model in GPU memory (66 of 66 layers). Context 16,384 tokens for the supervised arms and Option B, 65,536 for Option A.
- Python 3.14.7 in a virtual environment. `llama-cpp-python` 0.3.38 is used only to count a prompt's tokens with the model's own vocabulary (Option A's check on a cut prompt); it loads no model weights.

## Cloud model

- `gemini-3.8-flash` through the Gemini API (paid tier), thinking set to low. Every call records the model version the API reports, and a run stops on any other version.
- The key is read from a `.env` file, which git ignores.

## Data

- LongHealth (Adams et al., 2024), `benchmark_v5.json`, from https://github.com/kbressem/LongHealth. The benchmark is not in this repository. The code expects it at `~/LongHealth/data/benchmark_v5.json`.
- The fixed question sets are in `question_sets/`: `warmup_ids.json` (20 questions) and `eval_ids.json` (200 questions), with no overlap. Each entry is a patient and a question number only. The code expects them at `~/warmup_ids.json` and `~/eval_ids.json`. Their SHA-256 values are in `question_sets/SHA256SUMS.txt`.

## Earlier weeks only

- The MinionS runs of Weeks 1 to 5 used the Minions library (https://github.com/HazyResearch/minions), installed from its source.
- The Question A chart (`Weekly261118/chart_a.py`) uses matplotlib.
