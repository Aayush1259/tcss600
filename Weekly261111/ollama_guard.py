"""Ollama worker calls that fail instead of being cut.

On Nov 4 I found that Ollama keeps only the start and the end of a prompt
that reaches num_ctx, and raises no error. Every call here sends
"truncate": false and "shift": false at the top level of the /api/chat
request, so a prompt that does not fit is refused with an HTTP error.
The Python client in is-env (ollama 0.6.3) has no truncate or shift
argument, so the request goes straight to the HTTP API through the
standard library.

Checks after every call (each one stops the run):
- an HTTP 4xx answer (for example a prompt longer than the limit);
- done_reason other than "stop";
- prompt tokens plus the reply budget over num_ctx;
- for prompts of 2,000 characters or more, more than 5 characters per
  token (a cut prompt counts fewer tokens for the same text).
There is no lower bound: on Nov 11 a padding chunk of lab values gave
2.666 characters per token, because the tokenizer splits numbers into
single digits. Dense text lowers the ratio; it cannot mean a cut.
Connection errors and HTTP 5xx answers (Ollama restarting after a memory
kill) are retried, as the frozen worker does; nothing else is retried.
"""
import json
import os
import time
import urllib.error
import urllib.request

CPT_MAX, CPT_MIN_CHARS = 5.0, 2000


class Rejected(RuntimeError):
    """Ollama refused the request (HTTP 4xx)."""

    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


class GuardError(RuntimeError):
    """A call came back but failed a check."""


def host():
    h = os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
    return h if h.startswith("http") else "http://" + h


def _post(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _mean_logprob(items):
    vals = [float(it["logprob"]) for it in items or []
            if isinstance(it, dict) and isinstance(it.get("logprob"), (int, float))]
    return sum(vals) / len(vals) if vals else None


def build_payload(model, prompt, num_ctx, num_predict, fmt=None, logprobs=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "truncate": False,
        "shift": False,
        "keep_alive": "30m",
        "options": {"num_ctx": num_ctx, "temperature": 0, "num_predict": num_predict},
    }
    if fmt is not None:
        payload["format"] = fmt
    if logprobs:
        payload["logprobs"] = True
    return payload


def chat(model, prompt, num_ctx, num_predict, fmt=None, logprobs=False,
         post=None, retries=6, sleep=time.sleep, timeout=900):
    """One guarded chat call. Returns the same fields as the frozen OllamaWorker."""
    post = post or _post
    url = host() + "/api/chat"
    payload = build_payload(model, prompt, num_ctx, num_predict, fmt, logprobs)
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            status, body = post(url, payload, timeout)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last = e
            print(f"    ollama unreachable ({type(e).__name__}: {e}); retry {attempt + 1}/{retries}", flush=True)
            sleep(10 * (attempt + 1))
            continue
        if status >= 500:
            last = GuardError(f"HTTP {status}: {body[:300]}")
            print(f"    ollama HTTP {status}; retry {attempt + 1}/{retries}", flush=True)
            sleep(10 * (attempt + 1))
            continue
        if status >= 400:
            raise Rejected(status, body)
        r = json.loads(body)
        out = {
            "text": (r.get("message") or {}).get("content") or "",
            "mean_logprob": _mean_logprob(r.get("logprobs")),
            "prompt_tokens": r.get("prompt_eval_count"),
            "completion_tokens": r.get("eval_count"),
            "thinking_tokens": 0,
            "done_reason": r.get("done_reason"),
            "prompt_chars": len(prompt),
            "wall_time_s": round(time.time() - t0, 3),
        }
        pt = out["prompt_tokens"]
        out["chars_per_token"] = round(len(prompt) / pt, 3) if pt else None
        if out["done_reason"] != "stop":
            raise GuardError(f"done_reason {out['done_reason']!r}, not 'stop'")
        if not pt:
            raise GuardError("no prompt_eval_count in the reply")
        if pt + num_predict > num_ctx:
            raise GuardError(f"prompt {pt} + reply budget {num_predict} > num_ctx {num_ctx}")
        if len(prompt) >= CPT_MIN_CHARS and out["chars_per_token"] > CPT_MAX:
            raise GuardError(f"{out['chars_per_token']} characters per token, over {CPT_MAX}: the prompt may have been cut")
        return out
    raise last


def guarded_worker(model, num_ctx, num_predict=300):
    """The frozen OllamaWorker with call() routed through chat(). Not wired into
    run_pipeline.py yet: that change goes in the dated amendment first."""
    import backends

    class GuardedOllamaWorker(backends.OllamaWorker):
        def call(self, prompt, logprobs=False):
            out = chat(self.model, prompt, self.num_ctx, self.num_predict,
                       fmt=backends.JOB_SCHEMA, logprobs=logprobs, retries=self.retries)
            if logprobs and out["mean_logprob"] is None:
                raise backends.BackendError("log-probabilities requested but Ollama returned none")
            return out

    return GuardedOllamaWorker(model, num_ctx, num_predict, client=object())
