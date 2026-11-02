"""Extracts every needs_handcheck row from the three bosgame warmup
baseline logs into a readable digest, and saves a full copy to
~/handcheck_warmup.md. Run on bosgame: ~/is-env/bin/python handcheck_extract.py
"""
import glob
import json
import os

LOG_PATTERNS = [
    "local_only_27b_warmup_think-off_20261018_001239.jsonl",
    "local_only_27b_warmup_think-on_20261018_002952.jsonl",
    "cloud_only_warmup_20261018_003103.jsonl",
]
LOG_DIR = os.path.expanduser("~/baseline_logs")
OUT = os.path.expanduser("~/handcheck_warmup.md")


def tok_summary(r):
    parts = []
    for key, label in (("worker_completion_tokens", "worker out"),
                       ("completion_tokens", "out"),
                       ("thinking_tokens", "think")):
        if r.get(key) is not None:
            parts.append(f"{label}={r[key]}")
    return ", ".join(parts) or "tokens n/a"


def digest(answer, limit=1400):
    a = (answer or "").strip()
    if len(a) <= limit:
        return a
    return a[:450] + "\n...[middle omitted]...\n" + a[-850:]


def main():
    lines = []
    total = 0
    for pat in LOG_PATTERNS:
        for path in sorted(glob.glob(os.path.join(LOG_DIR, pat))):
            for raw in open(path):
                r = json.loads(raw)
                if not r.get("needs_handcheck"):
                    continue
                total += 1
                head = (f"## {os.path.basename(path)} | {r.get('patient')} "
                        f"q{r.get('q_idx')} | gold={r.get('gold')!r} | "
                        f"pred={r.get('predicted')} | {tok_summary(r)}")
                lines.append(head)
                lines.append("")
                lines.append(digest(r.get("answer")))
                lines.append("")
    with open(OUT, "w") as f:
        f.write("# Warmup hand-check queue\n\n" + "\n".join(lines))
    print(f"{total} flagged rows written to {OUT}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
