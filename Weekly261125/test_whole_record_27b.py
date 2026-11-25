"""Fake-backend tests for 27B alone, Option A (whole_record_27b.py). No model, key or benchmark text.
Run from the folder with whole_record_27b.py, local_alone_27b.py and the frozen modules:
    python test_whole_record_27b.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="wr_test_")
os.environ["BASELINE_LOG_DIR"] = TMP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_shared_c2 as rs  # noqa: E402
import runlog  # noqa: E402
import whole_record_27b as wr  # noqa: E402

ENV = {"ollama_version": "0.40.3", "model_digest": "sha256:abc123"}
OFFSET = 7                     # the fake chat template adds 7 tokens


def count(text):
    return len(text) // 4


def rec(i, fact=True, split="eval"):
    p = f"patient_{i:02d}"
    return {"patient": p, "q_idx": 3, "q_no": 4, "question": f"Which option is documented for {p}?",
            "options": {L: f"option {L}" for L in "ABCDE"}, "gold": "option B", "split": split,
            "documents": ["intro text " * 600, ("FACT: the answer is B " if fact else "nothing here ") * 5,
                          "closing text " * 600],
            "padding_n": 2, "padding_patients": [f"patient_{i + 1:02d}", f"patient_{i + 2:02d}"]}


class FakeOllama:
    """mode ok: refuses a prompt over the context (as "truncate": false should); cut: answers it with a
    cut count; cache: a prompt sent again reports 50 fewer tokens. A prompt with the fact gets a reply in
    the reply schema (reasoning, then the letter); one without it gets free text that breaks the schema."""

    def __init__(self, mode="ok"):
        self.mode, self.calls, self.seen, self.timeouts = mode, 0, set(), []

    def __call__(self, url, payload, timeout):
        self.calls += 1
        self.timeouts.append(timeout)
        if self.mode == "down":
            raise ConnectionError("refused")
        assert payload["format"] == wr.REPLY_SCHEMA and payload["truncate"] is False and payload["think"] is False
        assert payload["options"]["num_predict"] == wr.NUM_PREDICT == 2048
        prompt, ctx = payload["messages"][0]["content"], payload["options"]["num_ctx"]
        pt = count(prompt) + OFFSET
        if pt > ctx:
            if self.mode != "cut":
                return 400, json.dumps({"error": "the input length exceeds the context length"})
            pt = ctx - wr.NUM_PREDICT - 400
        if self.mode == "cache" and prompt in self.seen:
            pt -= 50
        self.seen.add(prompt)
        ans = (json.dumps({"reasoning": "The record states the answer is B.", "answer": "B"}) if "FACT" in prompt
               else "I cannot tell from these records.")
        return 200, json.dumps({"message": {"content": ans}, "prompt_eval_count": pt, "eval_count": 6,
                                "done_reason": "stop"})


class FakePs:
    def __init__(self, *values):
        self.values, self.calls = list(values or [(18000, 17500)]), 0

    def __call__(self):
        v = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return {"models": [{"name": "qwen3.8:27b", "size": v[0], "size_vram": v[1]}]}


class WRTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=TMP)
        runlog.LOG_DIR = self.dir
        self.ollama, self.ps = FakeOllama(), FakePs()
        self.recs = [rec(1), rec(2)]
        self.counter = count

    def main(self, argv):
        def make(timeout):
            return wr.WholeWorker("qwen3.8:27b", wr.NUM_CTX, wr.NUM_PREDICT, env=ENV, post=self.ollama,
                                  sleep=lambda s: None, ps=self.ps, timeout=timeout)
        out = io.StringIO()
        with mock.patch.object(rs, "load_records", side_effect=lambda split: [dict(r, split=split) for r in self.recs]), \
                mock.patch("time.sleep"), contextlib.redirect_stdout(out):
            code = wr.main(argv, make_worker=make, make_counter=lambda args: self.counter)
        return code, out.getvalue()

    def probe(self):
        return self.main(["probe", "--run-id", "p1"])

    def run_wr(self, *extra, split="eval", ref=True, timeout="900"):
        argv = ["run", "--split", split, "--run-id", "wr1", "--template-offset", str(OFFSET), "--timeout", timeout, *extra]
        if ref:
            argv += ["--expect-size", "18000", "--expect-size-vram", "17500"]
        return self.main(argv)

    def check(self, *extra, split="eval"):
        return self.main(["check", "--split", split, "--run-id", "wr1", *extra])

    def rows(self, name="eval_wr1"):
        return runlog.read_rows(os.path.join(self.dir, f"whole_record_27b_{name}.jsonl"))

    # probe
    def test_probe_passes_and_gives_the_run_settings(self):
        code, out = self.probe()
        self.assertEqual(code, 0, out)
        self.assertIn("PROBE PASSED", out)
        self.assertIn(f"--template-offset {OFFSET} --timeout 600", out)
        self.assertIn("refused, HTTP 400", out)
        steps = [r["step"] for r in self.rows("probe_p1")]
        self.assertEqual(steps, ["template overhead", "largest prompt, first", "largest prompt, second",
                                 "oversized prompt", "result"])
        self.assertTrue(all(t == wr.PROBE_TIMEOUT for t in self.ollama.timeouts))

    def test_probe_fails_if_an_oversized_prompt_is_answered(self):
        self.ollama = FakeOllama("cut")
        code, out = self.probe()
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL  an oversized prompt", out)

    def test_probe_fails_if_a_prompt_sent_twice_changes_its_count(self):
        self.ollama = FakeOllama("cache")
        code, out = self.probe()
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL  Ollama's count", out)

    # run
    def test_normal_run_counts_only_and_check(self):
        code, out = self.run_wr()
        self.assertEqual(code, 0, out)
        rows = self.rows()
        self.assertEqual([r["status"] for r in rows], ["ok", "ok"])
        r = rows[0]
        self.assertEqual(r["prompt_tokens"], r["expected_tokens"])
        self.assertEqual((r["predicted"], r["match"], r["timeout_s"]), ("B", True, 900))
        self.assertEqual(len(r["documents_sha256"]), 64)
        self.assertTrue(all(t == 900 for t in self.ollama.timeouts))
        self.assertNotIn("gold", out)
        self.assertNotRegex(out, r"pred\w*[ =:]+[A-E]\b")
        self.assertNotIn("match", out)
        code, out = self.check("--expect-questions", "2", "--expect-size", "18000", "--expect-size-vram", "17500")
        self.assertEqual(code, 0, out)
        self.assertIn("largest prompt:", out)

    def test_a_count_difference_halts_before_the_reply_is_kept(self):
        self.counter = lambda t: count(t) + 1
        code, out = self.run_wr()
        self.assertEqual(code, 3, out)
        self.assertEqual([r["error_type"] for r in self.rows()], ["TokenCountHalt"])

    def test_a_short_timeout_or_a_missing_offset_is_refused(self):
        for argv in (["run", "--split", "warmup", "--run-id", "x", "--template-offset", "7", "--timeout", "120"],
                     ["run", "--split", "warmup", "--run-id", "x", "--template-offset", "0", "--timeout", "900"],
                     ["run", "--split", "warmup", "--run-id", "x", "--timeout", "900"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self.main(argv)

    def test_a_reply_with_no_letter_is_wrong(self):
        self.recs = [rec(1), rec(2, fact=False)]
        self.run_wr()
        rows = self.rows()
        self.assertEqual((rows[1]["predicted"], rows[1]["match"]), (None, False))
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertIn("replies with no letter (counted wrong) 1", out)

    def test_the_reply_schema_is_sent_and_its_reply_is_read_by_the_strict_rule(self):
        code, out = self.run_wr()
        self.assertEqual(code, 0, out)
        r = self.rows()[0]
        self.assertEqual((r["reply_format"], r["reply_budget"]), (wr.REPLY_FORMAT, 2048))
        self.assertTrue(r["answer"].startswith('{"reasoning"'))
        self.assertEqual((r["predicted"], r["grading"]), ("B", "json (json)"))
        self.assertEqual(list(wr.REPLY_SCHEMA["properties"]), ["reasoning", "answer"])   # reasoning first

    def test_the_warmup_check_needs_every_reply_read(self):
        self.recs = [rec(1, split="warmup"), rec(2, fact=False, split="warmup")]
        self.run_wr(split="warmup")
        code, out = self.check(split="warmup")
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL  the strict rule reads a letter in every reply: 1 of 2", out)
        self.recs = [rec(1, split="warmup"), rec(3, split="warmup")]
        self.dir = tempfile.mkdtemp(dir=TMP)
        runlog.LOG_DIR = self.dir
        self.run_wr(split="warmup")
        code, out = self.check(split="warmup")
        self.assertEqual(code, 0, out)
        self.assertIn("PASS  the strict rule reads a letter in every reply: 2 of 2", out)

    def test_a_log_from_the_old_setting_fails_the_check(self):
        self.run_wr()
        path = os.path.join(self.dir, "whole_record_27b_eval_wr1.jsonl")
        rows = runlog.read_rows(path)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                r.pop("reply_format", None)
                f.write(json.dumps(r) + "\n")
        code, out = self.check()
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL  every call sent the reply schema", out)

    def test_resume_makes_no_new_call(self):
        self.run_wr()
        n = self.ollama.calls
        code, out = self.run_wr()
        self.assertEqual((code, self.ollama.calls), (0, n))

    def test_fallback_to_processor_halts(self):
        self.ps = FakePs((18000, 17500), (18000, 0))
        code, out = self.run_wr()
        self.assertEqual(code, 3, out)
        self.assertEqual(self.rows()[-1]["error_type"], "GpuHalt")

    def test_eval_refuses_without_reference(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_wr(ref=False)

    def test_a_changed_frozen_module_is_refused(self):
        with mock.patch.dict(rs.FROZEN, {"cloud_only": "0" * 64}):
            code, out = self.run_wr()
            self.assertEqual(code, 5, out)
            code, out = self.probe()
            self.assertEqual(code, 5, out)
        self.assertEqual(self.ollama.calls, 0)

    def test_journal_checks(self):
        self.run_wr()
        path = os.path.join(self.dir, "journal.txt")
        load = "llama_model_loader: load_tensors: offloaded 65/65 layers to GPU"
        chat = '[GIN] 2026/11/28 - 10:00:00 | 200 | 300.1s | 127.0.0.1 | POST "/api/chat"'
        for lines, want in (([load, chat, chat], 0), ([chat, chat], 1),
                            ([load, "truncating input prompt limit=65536", chat, chat], 1)):
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            code, out = self.check("--ollama-journal", path)
            self.assertEqual(code, want, out)

    def test_own_count_drops_final_whitespace_as_the_renderer_does(self):
        class FakeLlama:                     # one token per byte
            def __init__(self, model_path, vocab_only, verbose):
                self.args = (model_path, vocab_only, verbose)

            def tokenize(self, data, add_bos, special):
                assert (add_bos, special) == (False, False)
                return list(data)
        fake = types.SimpleNamespace(Llama=FakeLlama, __version__="test")
        with mock.patch.dict(sys.modules, {"llama_cpp": fake}):
            c = wr.GgufCounter("/models/blob")
        self.assertEqual(c.llm.args, ("/models/blob", True, False))
        self.assertEqual([c("abc"), c("abc\n"), c("abc\n\n"), c("abc \n")], [3, 3, 3, 3])
        self.assertEqual(c("a\nb\n"), 3)     # only the end is trimmed

    def test_shares_the_gpu_lock(self):
        lock = os.path.join(rs.HERE, ".gpu.lock")
        with open(lock, "w") as f:
            f.write(str(os.getppid()))
        try:
            code, out = self.run_wr()
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
