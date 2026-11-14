"""Guarded backends for the 4B shared-parts runs (code change C2, November 13).

The frozen pipeline (Weekly261104) stores a reply in its cache before anything checks it, calls
Ollama at a context of 2,048 with no guard, and does not record the model version the cloud API
reports. Every frozen file stays unchanged; this module adds three classes that the new runner,
run_shared_c2.py, uses instead of the frozen ones.

GuardedOllamaWorker
    Every worker call goes through ollama_guard.chat: "truncate": false and "shift": false, so a
    prompt that does not fit is refused instead of cut, and the call stops the run on a refused
    request, a stop reason other than "stop", a prompt that leaves no room for the 300-token reply,
    or more than 5 characters per token. Context 3,072, temperature 0, think off, the frozen job
    schema. A 120 second timeout and 3 attempts for connection errors (the frozen worker waited
    900 seconds, 6 times). Each reply carries a guard tag, the Ollama version and the model digest.
    The tag is part of params(), so a guarded request never shares a cache entry with an unguarded
    one. It is written new, not as a subclass of the frozen OllamaWorker.
    One change on November 13, after the guarded warmup halted on patient 2: a reply that stops
    because it used its whole 300-token budget (done reason "length" with exactly 300 reply
    tokens, and a prompt that leaves room for all 300) is kept and marked reply_cut, instead of
    stopping the run. Its input was complete; only the reply ran long. The frozen parser usually
    reads such a reply as unreadable, so the job counts as no finding (all 24 in the stopped
    evaluation run did); a cut reply that still holds complete JSON is read as given, and the count
    check reports how many cut replies were read as an answer. As in every earlier run with this
    budget (0 of 5,044 worker replies in the warmup at 3,072, 24 of 44,729 in the stopped
    evaluation run). Every other stop that is not "stop" still halts the run.
    A second change on November 13, after warmup3 lost the GPU (Ollama reloaded the model on the
    processor after a CUDA error and the run went on six times slower, which no check caught):
    after every worker call, also after a retried one and before the reply is kept, the worker
    reads Ollama's /api/ps (the model's size and the bytes of it in GPU memory). It halts if the
    model has no bytes in GPU memory, or if size or size_vram differ from the run's reference.
    The reference is given on the command line (for the evaluation, the warmup4 values) or, if
    not, taken from the run's calls log or its first call. Both values are logged on every row,
    with the number of attempts. A job over 3 times the running median only prints a warning.
    The guard tag is now c2-v2, so no reply from earlier runs can be replayed.

PinnedGeminiSupervisor
    The frozen call (cloud_only.ask_supervisor: the supervisor model, thinking low, retries for
    429 and 503), with the model version the API reports read from the response. A missing or
    different version stops the run before the reply is stored. An empty reply is an error and is
    never stored, so its one rerun is a fresh call. Two seconds between calls. params() is the
    frozen one, so supervisor cache keys are the frozen pipeline's.

GuardedCachedBackend
    The frozen CachedBackend with the order fixed: the cap is checked before every fresh
    supervisor call, the reply is checked, and only then stored. A cached reply is used only if it
    carries the marks (worker: guard tag, the same Ollama version and digest, a normal stop;
    supervisor: the expected version). Every call, cached or fresh, writes a calls-log row with
    the run id, the question, the attempt, the component, the dollars and the marks. A paid reply
    that is then rejected (wrong version, empty) is logged with its dollars. Each fresh worker call
    refreshes a heartbeat file.
"""
import json
import os
import time
import urllib.error
import urllib.request

import backends
import cloud_only
import ollama_guard

GUARD_TAG = "c2-v2"   # v2 (Nov 13): GPU residency marks; no v1 reply can be replayed under v2
SLOW_FACTOR = 3.0     # a job over 3 times the running median only warns
PRICE_IN_PER_1M, PRICE_OUT_PER_1M = 0.75, 3.75  # the frozen price table; thinking bills as output


class Halt(RuntimeError):
    """Stop the run now. The question gets a halt row; it is never graded from this reply."""


class VersionHalt(Halt):
    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage


class CacheMarkHalt(Halt):
    """A cached reply without the marks this run requires."""


class GpuHalt(Halt):
    """The model is not in GPU memory, or its share differs from the run's reference."""


class EmptyReply(RuntimeError):
    """The supervisor answered with no text. A failure: logged, not stored, rerun once."""

    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage


class CapReached(RuntimeError):
    """The run's cap on fresh supervisor calls or dollars is reached."""


def supervisor_usd(usage):
    usage = usage or {}
    out = (usage.get("completion_tokens") or 0) + (usage.get("thinking_tokens") or 0)
    return (usage.get("prompt_tokens") or 0) / 1e6 * PRICE_IN_PER_1M + out / 1e6 * PRICE_OUT_PER_1M


def _get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_env(model, get=None):
    """The Ollama server version and the model's digest, read from Ollama's HTTP API."""
    get = get or _get_json
    base = ollama_guard.host()
    version = get(base + "/api/version").get("version")
    digest = None
    for m in get(base + "/api/tags").get("models", []):
        if m.get("name") == model or m.get("model") == model:
            digest = m.get("digest")
    return {"ollama_version": version, "model_digest": digest}


def guarded_chat(model, prompt, num_ctx, num_predict, fmt=None, logprobs=False,
                 post=None, retries=3, sleep=time.sleep, timeout=120):
    """ollama_guard.chat (frozen, unchanged) with one difference: a reply that reached its own
    num_predict tokens is returned with reply_cut True instead of raising GuardError. The request
    is the frozen one (ollama_guard.build_payload: "truncate": false, "shift": false), and every
    other check is the same and in the same order."""
    post = post or ollama_guard._post
    url = ollama_guard.host() + "/api/chat"
    payload = ollama_guard.build_payload(model, prompt, num_ctx, num_predict, fmt, logprobs)
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
            last = ollama_guard.GuardError(f"HTTP {status}: {body[:300]}")
            print(f"    ollama HTTP {status}; retry {attempt + 1}/{retries}", flush=True)
            sleep(10 * (attempt + 1))
            continue
        if status >= 400:
            raise ollama_guard.Rejected(status, body)
        r = json.loads(body)
        out = {
            "text": (r.get("message") or {}).get("content") or "",
            "mean_logprob": ollama_guard._mean_logprob(r.get("logprobs")),
            "prompt_tokens": r.get("prompt_eval_count"),
            "completion_tokens": r.get("eval_count"),
            "thinking_tokens": 0,
            "done_reason": r.get("done_reason"),
            "prompt_chars": len(prompt),
            "wall_time_s": round(time.time() - t0, 3),
        }
        pt = out["prompt_tokens"]
        out["chars_per_token"] = round(len(prompt) / pt, 3) if pt else None
        out["attempts"] = attempt + 1
        out["reply_cut"] = out["done_reason"] == "length" and out["completion_tokens"] == num_predict
        if out["done_reason"] != "stop" and not out["reply_cut"]:
            raise ollama_guard.GuardError(f"done_reason {out['done_reason']!r} with {out['completion_tokens']} "
                                          f"reply tokens, not 'stop' and not the {num_predict}-token reply cap")
        if not pt:
            raise ollama_guard.GuardError("no prompt_eval_count in the reply")
        if pt + num_predict > num_ctx:
            raise ollama_guard.GuardError(f"prompt {pt} + reply budget {num_predict} > num_ctx {num_ctx}")
        if len(prompt) >= ollama_guard.CPT_MIN_CHARS and out["chars_per_token"] > ollama_guard.CPT_MAX:
            raise ollama_guard.GuardError(f"{out['chars_per_token']} characters per token, over "
                                          f"{ollama_guard.CPT_MAX}: the prompt may have been cut")
        return out
    raise last


def stop_problem(resp, num_predict):
    """None for a normal stop or a reply that used its whole budget; otherwise the reason."""
    if resp.get("done_reason") == "stop":
        return None
    if resp.get("done_reason") == "length" and resp.get("reply_cut") and resp.get("completion_tokens") == num_predict:
        return None
    return f"done_reason {resp.get('done_reason')!r} with {resp.get('completion_tokens')} reply tokens"


class GuardedOllamaWorker:
    kind = "ollama"

    def __init__(self, model, num_ctx=3072, num_predict=300, env=None, post=None,
                 timeout=120, attempts=3, sleep=time.sleep, ps=None, reference=None):
        self.model, self.num_ctx, self.num_predict = model, num_ctx, num_predict
        self.env = env if env is not None else ollama_env(model)
        if not self.env.get("ollama_version") or not self.env.get("model_digest"):
            raise Halt(f"could not read the Ollama version and the digest of {model}: {self.env}")
        self.post, self.timeout, self.attempts, self.sleep = post, timeout, attempts, sleep
        self.ps = ps or (lambda: _get_json(ollama_guard.host() + "/api/ps"))
        self.reference = reference          # {"model_size", "model_size_vram"}; None: the first call sets it
        self._times = []

    def params(self):
        return {"num_ctx": self.num_ctx, "num_predict": self.num_predict, "temperature": 0,
                "think": False, "format": "job-json", "guard": GUARD_TAG}

    def residency(self):
        """(size, size_vram) of the loaded model from Ollama's /api/ps, or (None, None). A read that
        fails three times is a halt (never a failed question): the reply is not kept and not counted."""
        last = None
        for attempt in range(3):
            try:
                ps = self.ps() or {}
                break
            except (OSError, ValueError) as e:     # URLError, timeouts and bad JSON are OSError or ValueError
                last = e
                self.sleep(5 * (attempt + 1))
        else:
            raise GpuHalt(f"could not read Ollama's /api/ps after 3 tries ({type(last).__name__}: {last})")
        for m in ps.get("models", []):
            if m.get("name") == self.model or m.get("model") == self.model:
                return m.get("size"), m.get("size_vram")
        return None, None

    def call(self, prompt, logprobs=False):
        out = guarded_chat(self.model, prompt, self.num_ctx, self.num_predict,
                           fmt=backends.JOB_SCHEMA, logprobs=logprobs, post=self.post,
                           retries=self.attempts, sleep=self.sleep, timeout=self.timeout)
        if logprobs and out["mean_logprob"] is None:
            raise ollama_guard.GuardError("log-probabilities requested but Ollama returned none")
        # where the model ran: read after every call, also after a retried one, before the reply is kept
        size, vram = self.residency()
        if not vram:
            raise GpuHalt(f"the model is not in GPU memory after this call (size {size}, size_vram {vram}); "
                          f"it may have fallen back to the processor")
        if self.reference is None:
            self.reference = {"model_size": size, "model_size_vram": vram}
            print(f"    GPU residency reference for this run: size {size}, size_vram {vram} "
                  f"({100 * vram / size:.1f} percent of the model in GPU memory)", flush=True)
        elif (size, vram) != (self.reference["model_size"], self.reference["model_size_vram"]):
            raise GpuHalt(f"GPU residency changed: size {size}, size_vram {vram}; reference "
                          f"size {self.reference['model_size']}, size_vram {self.reference['model_size_vram']}")
        self._times.append(out["wall_time_s"])
        if len(self._times) >= 20:
            med = sorted(self._times[-200:])[len(self._times[-200:]) // 2]
            if out["wall_time_s"] > SLOW_FACTOR * med:
                print(f"    warning: a job took {out['wall_time_s']:.1f} s, over {SLOW_FACTOR} times the "
                      f"running median {med:.1f} s", flush=True)
        return {**out, "guard": GUARD_TAG, **self.env, "model_size": size, "model_size_vram": vram}


class _VersionRecorder:
    """Passes calls to the real client and keeps the last response, so its model version can be read."""

    def __init__(self, client):
        self._client = client
        self.last = None
        self.models = self

    def generate_content(self, **kwargs):
        self.last = self._client.models.generate_content(**kwargs)
        return self.last


class PinnedGeminiSupervisor:
    kind = "gemini"

    def __init__(self, model, client, expect_version, pace_s=2.0, sleep=time.sleep, clock=time.time):
        self.model, self.expect_version = model, expect_version
        self.client = _VersionRecorder(client)
        self.pace_s, self.sleep, self.clock = pace_s, sleep, clock
        self._last = None

    def params(self):
        return {}

    def call(self, prompt, json_mode=True):
        if self._last is not None:
            wait = self.pace_s - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        self.client.last = None
        out = cloud_only.ask_supervisor(self.client, prompt)
        self._last = self.clock()
        version = getattr(self.client.last, "model_version", None)
        usage = {k: out.get(k) for k in ("prompt_tokens", "completion_tokens", "thinking_tokens")}
        if version != self.expect_version:
            raise VersionHalt(f"the API reported model version {version!r}, expected {self.expect_version!r}",
                              usage=usage)
        if not (out.get("text") or "").strip():
            raise EmptyReply("the supervisor reply was empty", usage=usage)
        return {"text": out["text"], **usage, "finish_reason": None,
                "wall_time_s": out["wall_time_s"], "model_version": version}


class Budget:
    """Fresh supervisor calls and dollars for the whole run, seeded from the run's own calls log."""

    def __init__(self, max_calls, max_usd, calls=0, usd=0.0):
        self.max_calls, self.max_usd, self.calls, self.usd = max_calls, max_usd, calls, usd

    @classmethod
    def from_calls_log(cls, path, max_calls, max_usd):
        calls, usd = 0, 0.0
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("kind") == "gemini" and not r.get("cached"):
                        calls += 1
                        usd += r.get("usd") or 0.0
        return cls(max_calls, max_usd, calls, usd)

    def before_fresh_call(self):
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise CapReached(f"{self.calls} fresh supervisor calls, cap {self.max_calls}")
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise CapReached(f"{self.usd:.4f} dollars of fresh supervisor calls, cap {self.max_usd}")

    def add(self, usd):
        self.calls += 1
        self.usd += usd


class GuardedCachedBackend(backends.CachedBackend):
    def __init__(self, raw, cache, calls_log, budget, run_id, expect_version=None, heartbeat=None):
        super().__init__(raw, cache, calls_log)
        self.budget, self.run_id = budget, run_id
        self.expect_version, self.heartbeat = expect_version, heartbeat
        self.context = {}

    def _marks_problem(self, resp):
        if self.raw.kind == "ollama":
            env = self.raw.env
            if resp.get("guard") != GUARD_TAG:
                return "no guard tag"
            stop = stop_problem(resp, self.raw.num_predict)
            if stop:
                return stop
            if resp.get("ollama_version") != env["ollama_version"] or resp.get("model_digest") != env["model_digest"]:
                return (f"Ollama {resp.get('ollama_version')} / {resp.get('model_digest')}, "
                        f"this run {env['ollama_version']} / {env['model_digest']}")
            if not resp.get("model_size_vram"):
                return "no GPU residency mark"
            ref = self.raw.reference
            if ref and (resp.get("model_size"), resp.get("model_size_vram")) != (ref["model_size"], ref["model_size_vram"]):
                return (f"GPU residency {resp.get('model_size')} / {resp.get('model_size_vram')}, "
                        f"reference {ref['model_size']} / {ref['model_size_vram']}")
        elif self.raw.kind == "gemini":
            if resp.get("model_version") != self.expect_version:
                return f"model version {resp.get('model_version')!r}, expected {self.expect_version!r}"
        return None

    def _log(self, component, resp, cached, usd, status="ok"):
        if not self.calls_log:
            return
        row = {"run_id": self.run_id, **self.context, "component": component, "kind": self.raw.kind,
               "model": self.raw.model, "cached": cached, "status": status,
               "prompt_tokens": resp.get("prompt_tokens"), "completion_tokens": resp.get("completion_tokens"),
               "thinking_tokens": resp.get("thinking_tokens", 0), "wall_time_s": resp.get("wall_time_s"),
               "usd": round(usd, 8)}
        if self.raw.kind == "ollama":
            row.update({k: resp.get(k) for k in ("done_reason", "reply_cut", "guard", "ollama_version",
                                                  "model_digest", "chars_per_token", "model_size",
                                                  "model_size_vram", "attempts")})
        else:
            row["model_version"] = resp.get("model_version")
        row["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.calls_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _beat(self, resp):
        if not self.heartbeat:
            return
        tmp = self.heartbeat + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": self.run_id, **self.context,
                       "fresh_worker_calls": self.fresh, "last_job_s": resp.get("wall_time_s")}, f)
        os.replace(tmp, self.heartbeat)

    def generate(self, prompt, component, **kw):
        payload = {"backend": self.raw.kind, "model": self.raw.model,
                   "params": self.raw.params(), "prompt": prompt, "kw": kw}
        hit = self.cache.get(payload)
        if hit is not None:
            problem = self._marks_problem(hit)
            if problem:
                raise CacheMarkHalt(f"cached {self.raw.kind} reply not usable in this run: {problem}")
            self.hits += 1
            self._log(component, hit, cached=True, usd=0.0)
            return {**hit, "cached": True}
        gemini = self.raw.kind == "gemini"
        if gemini:
            self.budget.before_fresh_call()
        try:
            resp = self.raw.call(prompt, **kw)
        except (VersionHalt, EmptyReply) as e:
            usd = supervisor_usd(e.usage)
            if gemini:
                self.budget.add(usd)
            self._log(component, e.usage or {}, cached=False, usd=usd, status=type(e).__name__)
            raise
        resp = {**resp, "model": self.raw.model}
        problem = self._marks_problem(resp)
        if problem:
            raise Halt(f"fresh {self.raw.kind} reply failed its check: {problem}")
        self.cache.put(payload, resp)
        self.fresh += 1
        usd = supervisor_usd(resp) if gemini else 0.0
        if gemini:
            self.fresh_usd += usd
            self.budget.add(usd)
        self._log(component, resp, cached=False, usd=usd)
        if not gemini:
            self._beat(resp)
        return {**resp, "cached": False}
