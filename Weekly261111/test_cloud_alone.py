"""Fake-client tests for cloud_alone.py: no API call, no key, no benchmark file.

Run from the folder that holds cloud_alone.py and the four frozen modules:
    python3 test_cloud_alone.py
Logs go to a temporary folder, never to the real log folder.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="cloud_alone_test_")
os.environ["BASELINE_LOG_DIR"] = TMP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cloud_alone  # noqa: E402
import cloud_only  # noqa: E402
import runlog  # noqa: E402

VERSION = "gemini-3.8-flash"
OK = {"text": '{"answer": "A"}', "version": VERSION}
COST = 10000 / 1e6 * runlog.FLASH_IN_PER_1M + 30 / 1e6 * runlog.FLASH_OUT_PER_1M


def rec(i, pad=None, pad_n=2):
    return {
        "patient": f"patient_{i:02d}", "q_idx": i, "q_no": i, "question": f"Question {i}?",
        "options": {L: f"option {L} of {i}" for L in "ABCDE"}, "gold": "A", "documents": [f"record text {i}"],
        "padding_n": pad_n,
        "padding_patients": pad if pad is not None else [f"patient_{i + 1:02d}", f"patient_{i + 2:02d}"],
    }


RECS = [rec(i) for i in range(1, 6)]


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0) if self.script else OK
        if isinstance(item, BaseException):
            raise item
        usage = types.SimpleNamespace(prompt_token_count=10000, candidates_token_count=10, thoughts_token_count=20)
        return types.SimpleNamespace(text=item["text"], model_version=item["version"], usage_metadata=usage)


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


class CloudAloneTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=TMP)
        runlog.LOG_DIR = self.dir
        cloud_only.ATTEMPTS = 0
        cloud_only.MAX_CALLS = None

    def launch(self, argv, script=(), recs=None):
        fake = FakeClient(script)
        out = io.StringIO()
        with mock.patch.object(cloud_alone, "make_client", return_value=fake), \
                mock.patch.object(cloud_alone.data_loader, "load_split",
                                  return_value=[dict(r) for r in (recs or RECS)]), \
                mock.patch("time.sleep"), contextlib.redirect_stdout(out):
            try:
                code = cloud_alone.main(argv)
            except SystemExit as e:
                code = e.code
        return code, fake, out.getvalue()

    def logs(self):
        return sorted(f for f in os.listdir(self.dir) if f.endswith(".jsonl"))

    def rows(self):
        logs = [f for f in self.logs() if f.startswith("cloud_alone_")]
        return runlog.read_rows(os.path.join(self.dir, logs[-1])) if logs else []

    def test_normal_run_logs_version_padding_and_code(self):
        code, fake, _ = self.launch(["--split", "eval"])
        rows = self.rows()
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 5)
        self.assertEqual(fake.models.calls, 5)
        self.assertEqual(self.logs()[0][:len("cloud_alone_eval_pad2_json_")], "cloud_alone_eval_pad2_json_")
        for r in rows:
            self.assertFalse(r["failed"])
            self.assertEqual(r["supervisor_model_version"], VERSION)
            self.assertEqual(r["padding_n"], 2)
            self.assertEqual(len(r["padding_patients"]), 2)
            self.assertEqual(r["attempts"], 1)
            self.assertEqual(set(r["code_sha256"]), {"cloud_alone.py", "cloud_only.py", "data_loader.py",
                                                     "grader.py", "runlog.py"})
            self.assertIn("google_genai", r["environment"])
            self.assertEqual(r["prompt"], cloud_only.build_prompt(RECS[r["q_idx"] - 1]))
            self.assertAlmostEqual(r["est_cost_usd"], round(COST, 6))

    def test_version_change_stops_before_the_row(self):
        code, fake, _ = self.launch(["--split", "eval"], [OK, OK, {"text": OK["text"], "version": VERSION + "-002"}])
        self.assertIn("expected", str(code))
        self.assertEqual(len(self.rows()), 2)

    def test_missing_version_stops_before_the_row(self):
        code, fake, _ = self.launch(["--split", "eval"], [{"text": OK["text"], "version": None}])
        self.assertIn("no model version", str(code))
        self.assertEqual(len(self.rows()), 0)
        self.assertEqual(fake.models.calls, 1)

    def test_expect_version_mismatch_stops(self):
        code, _, _ = self.launch(["--split", "eval", "--expect-version", "gemini-other"])
        self.assertIn("expected gemini-other", str(code))
        self.assertEqual(len(self.rows()), 0)

    def test_call_cap_covers_the_whole_run(self):
        code, fake, _ = self.launch(["--split", "eval", "--max-calls", "2"])
        self.assertEqual(len(self.rows()), 2)
        code, fake, _ = self.launch(["--split", "eval", "--max-calls", "3"])
        self.assertEqual(fake.models.calls, 1)
        self.assertEqual(len(self.rows()), 3)

    def test_usd_cap_covers_the_whole_run(self):
        self.launch(["--split", "eval", "--limit", "1"])
        self.assertEqual(len(self.rows()), 1)
        code, fake, _ = self.launch(["--split", "eval", "--max-usd", f"{COST * 1.5:.6f}"])
        self.assertEqual(fake.models.calls, 1)
        self.assertEqual(len(self.rows()), 2)

    def test_failure_row_then_one_rerun_then_final(self):
        code, fake, _ = self.launch(["--split", "eval"], [OK, RuntimeError("boom"), OK, OK, OK])
        rows = self.rows()
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 5)
        failed = [r for r in rows if r["failed"]]
        self.assertEqual(len(failed), 1)
        self.assertEqual((failed[0]["q_idx"], failed[0]["error_type"], failed[0]["match"]), (2, "RuntimeError", False))
        code, fake, _ = self.launch(["--split", "eval"], [RuntimeError("boom again")])
        self.assertEqual(fake.models.calls, 1)
        self.assertEqual(len(self.rows()), 6)
        code, fake, out = self.launch(["--split", "eval"])
        self.assertEqual(fake.models.calls, 0)
        self.assertIn("failed twice, final", out)
        done, failures, final_failed = cloud_alone.log_state(self.rows())
        self.assertEqual(final_failed, {("patient_02", 2)})

    def test_rerun_that_succeeds_counts_as_answered(self):
        self.launch(["--split", "eval"], [OK, RuntimeError("boom"), OK, OK, OK])
        code, fake, _ = self.launch(["--split", "eval"])
        self.assertEqual(fake.models.calls, 1)
        done, failures, final_failed = cloud_alone.log_state(self.rows())
        self.assertEqual(len(done), 5)
        self.assertEqual(final_failed, set())

    def test_three_failures_in_a_row_stop_the_run(self):
        code, fake, _ = self.launch(["--split", "eval"], [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
        self.assertIn("3 failures in a row", str(code))
        self.assertEqual(fake.models.calls, 3)
        self.assertEqual(len(self.rows()), 3)

    def test_retry_on_429_counts_both_attempts(self):
        code, fake, _ = self.launch(["--split", "eval", "--limit", "1"], [Exception("429 RESOURCE_EXHAUSTED"), OK])
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts"], 2)
        self.assertFalse(rows[0]["failed"])

    def test_padding_problems_stop_before_any_call(self):
        for bad in (rec(1, pad_n=0, pad=[]), rec(1, pad=["patient_01", "patient_02"]),
                    rec(1, pad=["patient_02", "patient_02"])):
            code, fake, _ = self.launch(["--split", "eval"], recs=[bad] + RECS[1:])
            self.assertIn("padding check failed", str(code))
            self.assertEqual(fake.models.calls, 0)
        self.assertEqual(self.logs(), [])

    def test_resume_refuses_a_log_it_should_not_extend(self):
        good = {"patient": "patient_01", "q_idx": 1, "failed": False, "padding_n": 2,
                "padding_patients": ["patient_02", "patient_03"], "supervisor_model_version": VERSION,
                "code_sha256": cloud_alone.code_hashes(), "est_cost_usd": 0.01, "attempts": 1}
        cases = {
            "more than one model version": [good, dict(good, patient="patient_02", q_idx=2,
                                                       supervisor_model_version=VERSION + "-002")],
            "no model version": [dict(good, supervisor_model_version=None)],
            "without padding 2": [dict(good, padding_n=1)],
            "different code": [dict(good, code_sha256={"cloud_alone.py": "0" * 64})],
        }
        for i, (expected, rows) in enumerate(cases.items()):
            path = os.path.join(self.dir, f"cloud_alone_eval_pad2_json_2026111{i}_000000.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            code, fake, _ = self.launch(["--split", "eval"])
            self.assertIn(expected, str(code))
            self.assertEqual(fake.models.calls, 0)
            os.remove(path)

    def test_never_resumes_a_frozen_script_log(self):
        with open(os.path.join(self.dir, "cloud_only_eval_pad2_json_20261024_000000.jsonl"), "w") as f:
            f.write(json.dumps({"patient": "patient_01", "q_idx": 1, "padding_n": 2}) + "\n")
        code, fake, _ = self.launch(["--split", "eval", "--limit", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(fake.models.calls, 1)
        self.assertEqual(len(self.rows()), 1)

    def test_dry_run_compares_prompts_and_makes_no_call(self):
        ref = os.path.join(self.dir, "ref.jsonl")
        with open(ref, "w", encoding="utf-8") as f:
            for r in RECS[:3]:
                f.write(json.dumps({"patient": r["patient"], "q_idx": r["q_idx"], "prompt": cloud_only.build_prompt(r),
                                    "supervisor_prompt_tokens": 1000, "supervisor_completion_tokens": 5,
                                    "supervisor_thinking_tokens": 50}) + "\n")
        with mock.patch.object(cloud_alone, "make_client", side_effect=AssertionError("no client in a dry run")):
            code, _, out = self.launch(["--split", "eval", "--dry-run", "--ref-log", ref])
        self.assertEqual(code, 0)
        self.assertIn("3 identical prompts, 0 different", out)
        self.assertIn("Projected cost of these 5 questions", out)
        with open(ref, "a", encoding="utf-8") as f:
            f.write(json.dumps({"patient": "patient_04", "q_idx": 4, "prompt": "something else",
                                "supervisor_prompt_tokens": 10}) + "\n")
        code, _, out = self.launch(["--split", "eval", "--dry-run", "--ref-log", ref])
        self.assertEqual(code, 1)
        self.assertIn("3 identical prompts, 1 different", out)
        self.assertEqual([f for f in self.logs() if f.startswith("cloud_alone_")], [])

    def test_ref_log_only_with_dry_run(self):
        code, fake, _ = self.launch(["--split", "eval", "--ref-log", "x.jsonl"])
        self.assertIn("only with --dry-run", str(code))
        self.assertEqual(fake.models.calls, 0)

    def test_key_never_written_in_a_failure_row(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "FAKEKEY12345"}):
            self.launch(["--split", "eval", "--limit", "1"], [RuntimeError("bad request with key FAKEKEY12345")])
        row = self.rows()[0]
        self.assertTrue(row["failed"])
        self.assertNotIn("FAKEKEY12345", json.dumps(row))
        self.assertIn("[key]", row["error_message"])

    def test_check_prints_counts_and_no_accuracy(self):
        self.launch(["--split", "eval"], [OK, RuntimeError("boom"), OK, OK, OK])
        path = os.path.join(self.dir, self.logs()[-1])
        code, _, out = self.launch(["--check", path])
        self.assertEqual(code, 0)
        self.assertIn("rows 5: answered 4, failure rows 1", out)
        self.assertIn("{'gemini-3.8-flash': 4}", out)
        self.assertNotIn("correct", out)
        self.assertNotIn("match", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
