# Note of November 25, 2026: Option A rules, fixed before its probe

This note adds to section 3 of my note of November 23 (27B alone). It is written and pushed before the Option A probe. No Option A call has been made, and no Option A result exists.

## 1. When the probe stops Option A

Option A goes no further than the probe, and nothing is adjusted to make it pass, if any of these holds:

- the chat template overhead measured on the short prompt is outside 1 to 64 tokens;
- the two sends of the largest prompt of the 200 report different prompt tokens;
- either send's prompt tokens differ from the prompt's own count plus the overhead;
- the deliberately oversized prompt is answered instead of refused with an HTTP 4xx error;
- the model's size and its size in GPU memory, as Ollama reports them after the probe calls, are not equal (the whole model must be in GPU memory);
- 220 times the longest probe call is more than 14 hours.

If the largest prompt fails the probe, I may restart the lab machine's Ollama once with parallel requests set to 1 and run the probe once more. The restart and the setting are logged. Nothing else is changed.

## 2. How the prompt tokens are counted

The own count is the tokenizer stored in the model's GGUF file, read by llama-cpp-python with vocab only, as in `whole_record_27b.py`. If llama-cpp-python cannot be installed in the lab machine's current Python environment, I use a separate Python environment for it and run the pushed script unchanged; the interpreter and the package versions are logged.

If that count cannot run by November 27 at 18:00, warns that it does not know the model's tokenizer, or disagrees with Ollama's count on the probe, I use this fallback instead, unless Prof. Cai objects (asked in Weekly Report 8): the run stops on any refused prompt, on any "truncating input prompt" line in Ollama's log for the run, and on any prompt outside 2.5 to 5.0 characters per token as Ollama counts it. "truncate": false is sent on every call either way. The fallback is a small change to the script; it is tested and pushed before the probe, and the change is stated in the report.

## 3. A halted or partial run

- A halt by any check ends Option A. The questions answered before the halt are reported descriptively, with no reading against cloud alone.
- One restart is allowed, only for a cause outside the run (a lost connection, a power cut, the GPU taken by another job), and only if all 200 can still be run and checked by December 4. The run resumes from its log; no answered question is sent again.
- Anything not run and checked by December 4 is reported as a limitation, as the note of November 23 says.

These rules do not change after any Option A result is seen.

## 4. Added November 25, before the probe: how the own count treats the end of a prompt

Two checks made before any Option A call, with no study text sent to the model:

- On all 4,350 Option B prompts, rebuilt exactly and counted with the model's own tokenizer, the prompt tokens Ollama logged were the own count plus 12, on every prompt, the first of the run included. So the own count agrees with Ollama, and the prompt cache never lowers Ollama's count.
- Ollama builds this model's chat format with a built-in renderer that cannot be read. Three short prompts, "alpha beta gamma" with 0, 1 and 2 final newlines, all gave 15 prompt tokens, against an own count of 3, 4 and 4. So the renderer drops whitespace at the end of a message.

Every Option A prompt ends with a newline. The own count therefore counts each prompt with its trailing whitespace removed, as the model receives it (a one-line change to `GgufCounter` in `whole_record_27b.py`, with a test). Nothing else in sections 1 to 3 changes.
