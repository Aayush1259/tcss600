"""Fake-backend tests for the guarded 27B alone run (local_alone_27b.py), from the 4B alone tests. No model, key or benchmark text.
Run from the folder with local_alone_27b.py, local_alone_vote.py (pushed November 12),
run_shared_c2.py, guarded_backend.py and the frozen modules:
    python test_local_alone_27b.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="la_test_")
os.environ["BASELINE_LOG_DIR"] = TMP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import local_alone_27b as la  # noqa: E402
import run_shared_c2 as rs  # noqa: E402
import runlog  # noqa: E402

ENV = {"ollama_version": "0.40.3", "model_digest": "sha256:abc123"}


def rec(i, split="eval"):
    p = f"patient_{i:02d}"
    return {"patient": p, "q_idx": 3, "q_no": 4, "question": f"Which option is documented for {p}?",
            "options": {L: f"option {L}" for L in "ABCDE"}, "gold": "option B", "split": split,
            "documents": ["intro text " * 600, "FACT: the answer is B " * 5, "closing text " * 600],
            "padding_n": 2, "padding_patients": [f"patient_{i + 1:02d}", f"patient_{i + 2:02d}"]}


class FakeOllama:
    def __init__(self, mode="ok"):
        self.mode, self.calls = mode, 0

    def __call__(self, url, payload, timeout):
        self.calls += 1
        if self.mode == "down":
            raise ConnectionError("refused")
        prompt = payload["messages"][0]["content"]
        assert payload["format"]["properties"]["answer"]["enum"][-1] == "none"   # the vote schema, not the job schema
        ans = "B" if "FACT" in prompt.split("Excerpt:", 1)[-1] else "none"
        return 200, json.dumps({"message": {"content": json.dumps({"answer": ans})},
                                "prompt_eval_count": max(1, len(prompt) // 4), "eval_count": 6,
                                "done_reason": "stop"})


class FakePs:
    def __init__(self, *values):
        self.values, self.calls = list(values or [(4800, 3500)]), 0

    def __call__(self):
        v = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return {"models": [{"name": "qwen3.8:27b", "size": v[0], "size_vram": v[1]}]}


class LATests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=TMP)
        runlog.LOG_DIR = self.dir
        self.ollama, self.ps = FakeOllama(), FakePs()
        self.recs = [rec(1), rec(2)]

    def run_la(self, *extra, split="eval", ref=True, run_id="la1"):
        def make():
            return la.VoteWorker("qwen3.8:27b", 16384, 300, env=ENV, post=self.ollama, sleep=lambda s: None, ps=self.ps)
        argv = ["run", "--split", split, "--run-id", run_id, *extra]
        if ref:
            argv += ["--expect-size", "4800", "--expect-size-vram", "3500"]
        out = io.StringIO()
        with mock.patch.object(rs, "load_records", return_value=[dict(r, split=split) for r in self.recs]), \
                mock.patch("time.sleep"), contextlib.redirect_stdout(out):
            code = la.main(argv, make_worker=make)
        return code, out.getvalue()

    def check(self, *extra, split="eval", run_id="la1"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = la.main(["check", "--split", split, "--run-id", run_id, *extra])
        return code, out.getvalue()

    def rows(self, split="eval", run_id="la1"):
        return runlog.read_rows(os.path.join(self.dir, f"local_alone_27b_{split}_{run_id}.jsonl"))

    def test_normal_run_votes_counts_only_and_check(self):
        code, out = self.run_la()
        self.assertEqual(code, 0, out)
        rows = self.rows()
        self.assertEqual([r["status"] for r in rows], ["ok", "ok"])
        r = rows[0]
        self.assertEqual(r["n_chunks"], 3)
        self.assertEqual(r["requests"], 3)
        self.assertEqual(r["predicted"], "B")
        self.assertTrue(r["match"])
        self.assertEqual(len(r["documents_sha256"]), 64)
        self.assertNotIn("gold", out)
        self.assertNotRegex(out, r"pred\w*[ =:]+[A-E]\b")
        self.assertNotIn("match", out)
        self.assertNotIn("correct", out.lower())
        code, out = self.check("--expect-questions", "2", "--expect-size", "4800", "--expect-size-vram", "3500")
        self.assertEqual(code, 0, out)

    def test_resume_makes_no_new_call(self):
        self.run_la()
        n = self.ollama.calls
        code, out = self.run_la()
        self.assertEqual(code, 0)
        self.assertEqual(self.ollama.calls, n)

    def test_fallback_to_processor_halts(self):
        self.ps = FakePs((4800, 3500), (4800, 0))
        code, out = self.run_la()
        self.assertEqual(code, 3, out)
        self.assertEqual(self.rows()[0]["error_type"], "GpuHalt")

    def test_eval_refuses_without_reference(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_la(ref=False)

    def test_connection_failures_then_stop(self):
        self.ollama = FakeOllama("down")
        self.recs = [rec(1), rec(2), rec(3)]
        code, out = self.run_la()
        self.assertEqual(code, 4, out)
        self.assertEqual([r["status"] for r in self.rows()], ["failed"] * 3)

    def test_a_code_error_halts_not_fails(self):
        with mock.patch.object(la, "run_question", side_effect=KeyError("documents")):
            code, out = self.run_la()
        self.assertEqual(code, 3, out)
        self.assertEqual([r["status"] for r in self.rows()], ["halt"])

    def test_frozen_vote_script_hash_checked(self):
        with mock.patch.object(la, "LAV_SHA256", "0" * 64):
            code, out = self.run_la()
        self.assertEqual(code, 5, out)

    def test_compare_log_counts_same_predictions(self):
        self.run_la(split="warmup", ref=False)
        old = os.path.join(self.dir, "old.jsonl")
        runlog.log_row(old, {"patient": "patient_01", "q_idx": 3, "predicted": "B"})
        runlog.log_row(old, {"patient": "patient_02", "q_idx": 3, "predicted": "C"})
        code, out = self.check("--compare-log", old, split="warmup")
        self.assertEqual(code, 0, out)
        self.assertIn("same prediction as old.jsonl: 1 of 2 questions", out)

    def journal(self, lines):
        path = os.path.join(self.dir, "journal.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    LOAD = "llama_model_loader: load_tensors: offloaded 27/37 layers to GPU"
    CHAT = '[GIN] 2026/11/15 - 10:00:00 | 200 |  3.1s | 127.0.0.1 | POST "/api/chat"'
    CHAT500 = '[GIN] 2026/11/15 - 10:00:00 | 500 |  0.2s | 127.0.0.1 | POST "/api/chat"'
    CRASH = "CUDA error: an illegal memory access was encountered"

    def test_journal_clean_and_a_recovered_crash_pass(self):
        self.run_la()                                               # 2 questions, 6 chunk calls
        code, out = self.check("--ollama-journal", self.journal([self.LOAD] + [self.CHAT] * 6))
        self.assertEqual(code, 0, out)
        self.assertIn("model process crashes in the journal: 0", out)
        code, out = self.check("--ollama-journal", self.journal(
            [self.LOAD] + [self.CHAT] * 3 + [self.CRASH, self.LOAD] + [self.CHAT] * 3))
        self.assertEqual(code, 0, out)
        self.assertIn("model process crashes in the journal: 1", out)

    def test_journal_failures(self):
        self.run_la()
        cases = {
            "answered before a reload": [self.LOAD] + [self.CHAT] * 3 + [self.CRASH] + [self.CHAT] * 3,
            "truncating": [self.LOAD, "truncating input prompt limit=3072"] + [self.CHAT] * 6,
            "one layer split": [self.LOAD, self.LOAD.replace("27/37", "0/37")] + [self.CHAT] * 6,
            "HTTP 500": [self.LOAD, self.CHAT500] + [self.CHAT] * 6,          # no retry was logged
            "unlogged": [self.LOAD] + [self.CHAT] * 10,                      # 4 more than 6 calls, 1 launch of 3
            "no model load seen": [self.CHAT] * 6,
            "runner terminated, no reload": [self.LOAD] + [self.CHAT] * 3
            + ["llama runner process has terminated: exit status 2"] + [self.CHAT] * 3,
        }
        for name, lines in cases.items():
            code, out = self.check("--ollama-journal", self.journal(lines))
            self.assertEqual(code, 1, f"{name}: {out}")

    def test_reply_cuts_over_one_percent_fail_the_check(self):
        self.run_la()
        path = os.path.join(self.dir, "local_alone_27b_eval_la1.jsonl")
        rows = self.rows()
        rows[0]["chunks"][0].update(done_reason="length", reply_cut=True, completion_tokens=300)
        os.remove(path)
        for r in rows:
            runlog.log_row(path, r)
        code, out = self.check()
        self.assertEqual(code, 1, out)
        self.assertIn("reply cuts at 300 tokens: 1 of 6 calls", out)

    def test_shares_the_gpu_lock_with_the_pipeline(self):
        lock = os.path.join(rs.HERE, ".gpu.lock")
        with open(lock, "w") as f:
            f.write(str(os.getppid()))
        try:
            code, out = self.run_la()
            self.assertEqual(code, 5, out)
            self.assertIn("lock", out)
        finally:
            if os.path.exists(lock):
                os.remove(lock)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
