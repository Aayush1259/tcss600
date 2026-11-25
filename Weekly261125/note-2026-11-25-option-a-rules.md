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

## 5. Added November 25, before the warmup: a second, descriptive reading of the reply

The one traced warmup question (patient_01 q2) passed every check. Its reply held a JSON answer field with one letter, plus other text, so cloud alone's reading, which needs the whole reply to be JSON, found no letter. Whether that letter is correct has not been looked at. No other Option A answer exists.

- The primary reading does not change: cloud alone's rule (`grader.grade_json`); a reply with no letter is wrong. The verdict against cloud alone ("close", "worse" or "inconclusive", section 4 of the note of November 23) uses only this reading.
- A second reading is reported beside it, descriptive, with no verdict: every place in the reply where an "answer" field is followed by a colon and a quoted single letter A to E (either case) is read; if all of them give the same letter, that letter is the answer; if there is none, or there are two different letters, the reply has no letter and is wrong. The letter is then graded against the gold option exactly as in the primary reading.
- The second reading is applied the same way to cloud alone's replies, and the difference and the patient cluster interval are reported for both readings, on the 91 and on the 200.
- I also report how many Option A replies have no letter in the primary reading but one letter in the second.
- The run does not change. The log keeps every full reply, so the second reading is computed when scoring.

## 6. Added November 25, after the failed warmup and before any evaluation call: the reply schema

What happened. The warmup `wr27_warm` (the 20 warmup questions, which are not part of the 200) passed every count, GPU and truncation check, but 8 of its 20 replies used all 300 reply tokens, against the limit of 1 percent (amendment 2, section 3). I then counted the shape of the replies and looked at no grade: no reply started with "{"; the model wrote its reasoning first and the JSON answer last; 12 of the 20 reached one letter in the answer field and none of the 8 cut replies did; the strict reading found a letter in none of the 20. The cause is my own setting of November 23 (thinking off, 300 tokens, no reply schema), which did not match cloud alone: cloud alone ran with thinking low and no reply limit, so it reasoned silently and returned JSON only. My trace of one question checked every count but not that the strict rule could read the reply; it should have.

The change, made once and before any evaluation call:

- Every call sends a reply schema: a JSON object with a "reasoning" text field first and an "answer" field last, which must be one letter from A to E. The reply budget is 2,048 tokens; the largest prompt of the 200 (39,051 tokens) plus 2,048 fits the context of 65,536.
- Unchanged: cloud alone's prompt, qwen3.8:27b (digest `aaee06c39dcf`), Ollama 0.40.3, the padded record at N = 2, the context of 65,536, thinking off, temperature 0, "truncate": false, the template overhead of 12 and the timeout of 600 seconds from the probe, the GPU reference, the strict reading (`grader.grade_json`, which reads the "answer" field of a JSON object), the second reading of section 5, and every rule of sections 1 to 4.
- Differences I state now: the reasoning is written in the reply, while cloud alone's was silent; the schema forces a letter, so a guess scores 20 percent, while a cloud alone reply could be unreadable and count wrong; Option B's schema allowed "none" for each chunk.
- The failed warmup stays on record under its run id `wr27_warm` and is not used.
- No rule of sections 1 to 3 changes, so the last sentence of section 3 holds. The reply schema and budget were settings of my note of November 23, and the 4B work set the precedent for a change found in a warmup: that warmup was set aside, the rule was written, and a fresh warmup passed before the evaluation (amendment 2, section 5).

Steps and gates:

- One question traced (`wr27_trace2`): the reply starts with "{", the reasoning comes before the answer, the answer is one letter, the call stops normally under 2,048 tokens, and the prompt count equals the own count plus 12 and the count of the same question in `wr27_warm`. A fault in my code found here may be fixed and traced again; if the model does not follow the schema, Option A stops.
- The 20 warmup questions (`wr27_warm2`): every check of the script passes, with no reply at 2,048 tokens, a letter read by the strict rule in all 20, and the median call times 200 under 14 hours. If any of these fails, Option A stops and is reported as a limitation; no other setting is tried.
- The 200 start only after Prof. Cai's explicit yes, asked in Weekly Report 8, and no later than November 30 at 12:00. Without a yes by November 30 at 10:00, Option A is not run and is reported as a limitation.

How the result is read, fixed now. The verdict against cloud alone keeps the rule of section 4 of my note of November 23, on the 91, with the 200 reported beside it. Because the reasoning is written and the letter is forced, a result close to cloud alone is read as evidence that 27B reading the whole record may not need supervision; a result that is worse or inconclusive is read as a lower bound, not as evidence that 27B cannot read the whole record. I also report how many cloud alone replies the strict rule could not read, and the question by question comparison with Option B and with the supervised 27B arms.

This section does not change after the fresh warmup's result is seen.
