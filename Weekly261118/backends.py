"""Backends for the shared-parts pipeline: a cache wrapper, the real
Ollama worker, and the real Gemini supervisor. (Mocks live in mocks.py.)

A raw backend has: .kind, .model, .params() and .call(prompt, **kw) ->
dict with text, prompt_tokens, completion_tokens, thinking_tokens,
wall_time_s (workers may add mean_logprob). CachedBackend wraps one:

  - A request seen before is answered from the DiskCache with the usage
    and latency it had when first measured, so cost and latency numbers
    are reproducible and nothing is paid twice.
  - Every FRESH call is appended to a calls log (component, tokens,
    time). cost_model.py --calls-log turns that into measured tokens per
    call, replacing the assumed ones in the cost estimate.

The live backends are NOT exercised by the unit tests (no model or key on
the dev machine). Their first run is the verification: use
`run_pipeline.py shared --limit 1` on the target machine.
"""
import json
import time

from cache import DiskCache  # noqa: F401  (re-exported for callers)

JOB_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["answer", "quote"],
}


class BackendError(RuntimeError):
    pass


class CachedBackend:
    def __init__(self, raw, cache, calls_log=None):
        self.raw = raw
        self.cache = cache
        self.calls_log = calls_log
        self.hits = 0
        self.fresh = 0
        # Fresh supervisor dollars, list price, for --max-usd caps.
        # Cached calls add 0 here, because they were paid on first use.
        # Worker calls are local and cost 0 dollars; this field is only
        # meaningful for the supervisor backend, where tokens are priced.
        self.fresh_usd = 0.0

    @property
    def model(self):
        return self.raw.model

    def generate(self, prompt, component, **kw):
        """Cached call. `component` labels the fresh-call log only; it is
        deliberately NOT part of the cache key, so an identical request made
        for two components is answered once."""
        payload = {"backend": self.raw.kind, "model": self.raw.model,
                   "params": self.raw.params(), "prompt": prompt, "kw": kw}
        hit = self.cache.get(payload)
        if hit is not None:
            self.hits += 1
            return {**hit, "cached": True}
        resp = self.raw.call(prompt, **kw)
        resp = {**resp, "model": self.raw.model}
        self.cache.put(payload, resp)
        self.fresh += 1
        if self.raw.kind == "gemini":
            # Same pricing as runlog.supervisor_cost_usd, $0.75 in,
            # $3.75 out per 1M, thinking bills as output.
            pt = resp.get("prompt_tokens") or 0
            out = (resp.get("completion_tokens") or 0) + (resp.get("thinking_tokens") or 0)
            self.fresh_usd += pt / 1e6 * 0.75 + out / 1e6 * 3.75
        if self.calls_log:
            row = {"component": component, "model": self.raw.model,
                   "prompt_tokens": resp.get("prompt_tokens"),
                   "completion_tokens": resp.get("completion_tokens"),
                   "thinking_tokens": resp.get("thinking_tokens", 0),
                   "wall_time_s": resp.get("wall_time_s"),
                   "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(self.calls_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return {**resp, "cached": False}


def mean_logprob(items):
    """Mean of per-token log-probabilities. Tolerates objects, dicts, floats."""
    vals = []
    for it in items or []:
        if isinstance(it, (int, float)):
            vals.append(float(it))
        elif isinstance(it, dict) and isinstance(it.get("logprob"), (int, float)):
            vals.append(float(it["logprob"]))
        elif isinstance(getattr(it, "logprob", None), (int, float)):
            vals.append(float(it.logprob))
    return sum(vals) / len(vals) if vals else None


class OllamaWorker:
    """Local worker (qwen3 via Ollama): JSON-mode, think off, temperature 0.

    NOTE (Oct 17): `think` is a TOP-LEVEL chat parameter; inside options it
    is silently ignored. JSON mode (format=schema) is what made the 4B emit
    parseable output (Oct 14 / Oct 18). If log-probabilities are requested
    and Ollama returns none, this raises instead of returning a signal of
    None: the log-prob signal must never be silently missing.
    """
    kind = "ollama"

    def __init__(self, model, num_ctx, num_predict=300, client=None, retries=6):
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.retries = retries
        if client is None:
            import ollama  # lazy: unit tests and dry runs need no Ollama
            client = ollama.Client(timeout=900)
        self.client = client

    def params(self):
        return {"num_ctx": self.num_ctx, "num_predict": self.num_predict,
                "temperature": 0, "think": False, "format": "job-json"}

    def call(self, prompt, logprobs=False):
        last = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                extra = {"logprobs": True} if logprobs else {}
                resp = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    think=False, format=JOB_SCHEMA, keep_alive="30m",
                    options={"num_ctx": self.num_ctx, "temperature": 0,
                             "num_predict": self.num_predict},
                    **extra)
                lp = mean_logprob(getattr(resp, "logprobs", None))
                if logprobs and lp is None:
                    raise BackendError(
                        "log-probabilities requested but Ollama returned none "
                        "(check the Ollama version and the client's logprobs field)")
                return {
                    "text": resp.message.content or "",
                    "mean_logprob": lp,
                    "prompt_tokens": getattr(resp, "prompt_eval_count", None),
                    "completion_tokens": getattr(resp, "eval_count", None),
                    "thinking_tokens": 0,
                    "done_reason": getattr(resp, "done_reason", None),
                    "wall_time_s": round(time.time() - t0, 3),
                }
            except BackendError:
                raise
            except Exception as e:  # noqa: BLE001
                last = e
                wait = 10 * (attempt + 1)
                print(f"    worker call failed ({type(e).__name__}: {e}); "
                      f"retry {attempt+1}/{self.retries} in {wait}s", flush=True)
                time.sleep(wait)
        raise last


class GeminiSupervisor:
    """Cloud supervisor. Reuses cloud_only.ask_supervisor, which already
    handles 503/429 retry, the daily-quota stop, and None responses."""
    kind = "gemini"

    def __init__(self, model, client, pace_s=15):
        self.model = model
        self.client = client
        self.pace_s = pace_s  # free tier is 5 requests/minute
        self._last = None

    def params(self):
        return {}

    def call(self, prompt, json_mode=True):
        # The freeze cloud_only fixes the model (gemini-3.8-flash) and
        # thinking low internally and takes no config: it is called with
        # (client, prompt) only. json_mode is accepted for interface
        # compatibility; the prompts request JSON in text and the
        # pipeline parsers extract it defensively (fallback flags in
        # the log show when a reply was not JSON). finish_reason is
        # absent from the freeze return, so it defaults to None.
        import cloud_only  # lazy: needs google-genai
        if self._last is not None:
            wait = self.pace_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
        out = cloud_only.ask_supervisor(self.client, prompt)
        self._last = time.time()
        return {
            "text": out["text"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "thinking_tokens": out["thinking_tokens"],
            "finish_reason": out.get("finish_reason"),
            "wall_time_s": out["wall_time_s"],
        }
