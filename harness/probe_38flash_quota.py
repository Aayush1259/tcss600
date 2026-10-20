"""10-call quota probe for gemini-3.8-flash. Machine: laptop or bosgame (either works).

Purpose: measure THIS project's actual free-tier daily cap on the frozen
supervisor model before the number goes into Weekly261021. The Sep probe
echoed "limit: 20 requests per day"; Oct community listings say ~500/day
for flash-class. Ten tiny calls settle which one is real.

The probe does NOT retry on 429: a 429 is the signal, not a failure.
Pacing is 15 s between calls to stay under the 5 req/min free-tier cap,
so per-minute throttling cannot masquerade as a daily cap.

Usage:
    ~/is-env/bin/python probe_38flash_quota.py        # laptop (WSL2)
    /home/localai/is-env/bin/python probe_38flash_quota.py   # bosgame

Needs GEMINI_API_KEY in the environment. Writes
probe_38flash_quota_<UTC timestamp>.jsonl in the current directory
and prints a verdict summary at the end.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

for _sp in ("/home/localai/is-env/lib/python3.14/site-packages",
            os.path.expanduser("~/is-env/lib/python3.14/site-packages")):
    if os.path.isdir(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

from google import genai  # noqa: E402

MODEL = "gemini-3.8-flash"  # frozen supervisor, 2026-10-17
N_CALLS = 10
SPACING_S = 15
PROMPT = "Reply with exactly one word: ok"


def extract_daily_limit(msg):
    m = re.search(r"limit:\s*(\d+)\s*requests?\s*per\s*day", msg, re.I)
    return int(m.group(1)) if m else None


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in the environment.", flush=True)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = f"probe_38flash_quota_{ts}.jsonl"

    ok = 0
    rate_limited = 0
    other_errors = 0
    stated_limit = None

    print(f"Probe: {N_CALLS} calls to {MODEL}, {SPACING_S}s spacing.", flush=True)
    with open(log_path, "w") as log:
        for i in range(1, N_CALLS + 1):
            rec = {"call": i, "utc": datetime.now(timezone.utc).isoformat()}
            try:
                t0 = time.time()
                r = client.models.generate_content(model=MODEL, contents=PROMPT)
                dt = time.time() - t0
                u = getattr(r, "usage_metadata", None)
                rec.update({
                    "status": "ok",
                    "wall_time_s": round(dt, 1),
                    "prompt_tokens": getattr(u, "prompt_token_count", None),
                    "completion_tokens": getattr(u, "candidates_token_count", None),
                    "thinking_tokens": getattr(u, "thoughts_token_count", 0) or 0,
                    "text": (r.text or "").strip()[:50],
                })
                ok += 1
                print(f"  call {i}: ok ({rec['wall_time_s']}s)", flush=True)
            except Exception as e:  # noqa: BLE001 - the error IS the data
                msg = str(e)
                lim = extract_daily_limit(msg)
                if lim:
                    stated_limit = lim
                if "429" in msg:
                    rate_limited += 1
                    rec.update({"status": "429", "error": msg[:600]})
                    print(f"  call {i}: 429 rate-limited"
                          + (f" (stated daily limit: {lim})" if lim else ""), flush=True)
                    print(f"    error excerpt: {msg[:220]}", flush=True)
                else:
                    other_errors += 1
                    rec.update({"status": "error", "error": msg[:600]})
                    print(f"  call {i}: ERROR {msg[:220]}", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if i < N_CALLS:
                time.sleep(SPACING_S)

    print(f"\nLog: {log_path}", flush=True)
    print(f"Result: {ok} ok / {rate_limited} 429 / {other_errors} other errors",
          flush=True)
    if stated_limit:
        days = 11700 / stated_limit
        print(f"API-stated daily cap: {stated_limit} requests/day", flush=True)
        print(f"At that cap, ~11,700 supervisor calls take ~{days:.0f} days.",
              flush=True)
        print("Write THIS measured number (not 20, not 500) into Weekly261021.",
              flush=True)
    elif rate_limited:
        print("429s seen but no daily limit in the error text; paste the log "
              "excerpt and I will interpret it.", flush=True)
    else:
        print("No 429s: the free tier carried 10 calls. The daily cap is "
              "higher than 10; the report should state the probe result as "
              "a lower bound, not as proof of 500/day.", flush=True)


if __name__ == "__main__":
    main()
