"""Test that the Ollama guards work: one normal prompt, one oversized prompt.

The normal prompt must come back with done_reason "stop" and its prompt
tokens under the limit. The oversized prompt (about twice the limit) must
be refused with an HTTP 4xx error. If it comes back with an answer, the
prompt was cut and the guards do not work: exit code 2, and no run uses
them. If Ollama cannot be reached, exit code 3; that is not a pass.

Laptop WSL, from ~/pipeline-4b-ctx3072 with ollama_guard.py next to it:
    ~/is-env/bin/python ollama_guards_test.py --model qwen3:4b --num-ctx 3072
Afterwards Ollama's own log must show no cut in the same minutes:
    journalctl -u ollama --since "10 minutes ago" --no-pager | grep -c "truncating input prompt"
"""
import argparse
import sys
import urllib.error

import ollama_guard

# Same shape as the real calls: a JSON schema and a 300-token reply budget.
# Without a schema, qwen3:4b with thinking off writes its reasoning into the
# reply and runs past a small budget (seen Nov 10: done_reason "length").
SCHEMA = {"type": "object",
          "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D", "E", "none"]}},
          "required": ["answer"]}
BUDGET = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--num-ctx", type=int, default=3072)
    args = ap.parse_args()
    print(f"ollama at {ollama_guard.host()}, model {args.model}, num_ctx {args.num_ctx}")

    try:
        r = ollama_guard.chat(args.model, 'Reply with JSON only: {"answer": "A"}',
                              args.num_ctx, BUDGET, fmt=SCHEMA, retries=1)
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        print(f"FAIL: Ollama not reachable: {e}")
        sys.exit(3)
    print(f"NORMAL: prompt_tokens={r['prompt_tokens']} completion_tokens={r['completion_tokens']} "
          f"done_reason={r['done_reason']} text={r['text'][:80]!r}")

    big = "word " * (args.num_ctx * 2)
    try:
        r2 = ollama_guard.chat(args.model, big, args.num_ctx, BUDGET, fmt=SCHEMA, retries=1)
        print(f"FAIL: the oversized prompt came back (prompt_tokens={r2['prompt_tokens']}, "
              f"done_reason={r2['done_reason']}): it was cut, so the guards do not work.")
        sys.exit(2)
    except ollama_guard.GuardError as e:
        print(f"FAIL: the oversized prompt came back and then failed a check ({e}): it was cut.")
        sys.exit(2)
    except ollama_guard.Rejected as e:
        print(f"OVERSIZED: refused as required, {e}")
    print("PASS: the normal prompt stops normally and the oversized prompt is refused. "
          "Now check Ollama's log for 'truncating input prompt' (must be 0).")


if __name__ == "__main__":
    main()
