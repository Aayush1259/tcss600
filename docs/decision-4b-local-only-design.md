# Decision: 4B Local-Only Baseline Design (frozen 2026-10-17)

## What I decided

The 4B local-only baseline runs the worker over the record in ~1,500-token document chunks (about 8 to 9 chunks per 11,000 to 13,000 token record), with one worker call per chunk. Each call gets the question, the options, and that chunk only. The final answer is the majority vote over the per-chunk letters, with ties broken by the earliest chunk. No supervisor is involved at any step.

Labels for every number from this baseline: protocol local-only (chunked) | worker qwen3:4b, ctx 2048 | supervisor none | grading exact match against gold, disputed answers go to the hand-check queue.

## Why this decision was needed

The 4B worker's context window is 2,048 tokens. The Minion protocol hands the worker the whole patient record at once (11,000 to 13,000 tokens). On October 13 I found and measured this exact failure on the 27B side: the server silently truncated the prompt and the worker answered from a document it never fully saw. Prof. Cai's October 15 feedback ruled out running the 4B pilot this way. So a local-only 4B baseline needs a chunking design before it can run at all.

## How it works, step by step

1. Split the patient's record into consecutive ~1,500-token chunks. The ~500 remaining tokens of the 2,048 window are headroom for the question, the options, and the instructions.
2. For each chunk, call the 4B worker once with the same multiple-choice question.
3. Collect one letter (A to E) per chunk.
4. Count votes. The most common letter is the final answer. If two letters tie, the one from the earliest chunk wins.

## Why I chose this approach

- It keeps the baseline truly local-only. RQ2 needs a clean "no review" reference point: worker alone, no supervisor. Any design that lets the supervisor combine the chunk answers would contaminate that reference.
- It is deterministic and statable in one sentence, which is what the new labeling rule demands.
- Majority vote mirrors the self-consistency idea and needs no extra machinery or tuning.
- It is expected to be weak (most chunks do not contain the answer, so most votes are guesses). That is what a baseline is for: the point of RQ2 is the gap between no review, review every step, and MinionS.

## Why not the alternatives

- **Full record handed to the 4B:** silent truncation, already measured, ruled out by Cai.
- **Supervisor combines the chunk answers:** stops being local-only and breaks the RQ2 comparison.
- **Naive concatenation of chunk outputs:** does not produce a single letter without a judge doing the picking, which reintroduces the supervisor through the back door.

## Corrections to my own earlier notes

- An earlier draft said "2,000-token chunks." That was wrong: 2,000 tokens of document plus ~500 tokens of prompt exceeds the 2,048 window and re-truncates. The correct chunk size is ~1,500 tokens of document text.
- An earlier draft said "naive concatenation." That was replaced by majority vote for the reason above.
