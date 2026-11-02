"""Prompts and parsers for the shared-parts pipeline (proposal v4, sec. 2).

STATUS: frozen in pre-registration 2026-11-03. No edits after the freeze
commit. Edits change the cache keys, so old cached responses are not reused.

Every parser is defensive and reports whether it had to fall back, so a
malformed model reply is visible in the logs instead of silently turning
into an answer (the harness rule since Oct 18).

Component names are shared with cost_model.py (--calls-log):
plan, review, aggregate, oracle_aggregate, worker.
"""
import hashlib
import json
import re

import grader

MAX_STEPS = 6  # proposal: the supervisor writes at most six subtasks

# Fixed, cloud-free paraphrase templates for the agreement signal. The
# proposal says the worker reruns the step "with two paraphrased
# instructions" without saying who writes them; fixed templates keep the
# signal free of extra cloud calls and fully reproducible. Confirmed in
# pre-registration 2026-11-03 as the two fixed templates for agreement.
PARAPHRASES = (
    "Look through this excerpt and report the following: {instr}",
    "From the excerpt only, find the answer to this request: {instr}",
)


def extract_json(text):
    """First JSON object/array in `text`, or None. Tolerates ```json fences."""
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            return None
    return None


# ------------------------------------------------------------------ plan
def plan_prompt(question, options):
    opts = "\n".join(f"({L}) {options[L]}" for L in "ABCDE")
    return (
        "You are planning how to answer a multiple-choice question about a "
        "patient's medical records. You cannot see the records. A worker "
        "will run each subtask you write over every excerpt of the records "
        "and report a short answer with a supporting quote, or 'none'.\n\n"
        f"Write at most {MAX_STEPS} short, independent extraction subtasks "
        "that together provide the evidence needed. Reply with a JSON list "
        'of strings, for example ["List every blood pressure reading with '
        'its date", "List every medication change with its date"].\n\n'
        f"Question: {question}\n{opts}"
    )


def parse_plan(text, question):
    """-> (steps, fallback). Falls back to one step = the question itself."""
    data = extract_json(text)
    if isinstance(data, dict):
        data = data.get("subtasks") or data.get("steps")
    if isinstance(data, list):
        steps = [str(s).strip() for s in data if str(s).strip()][:MAX_STEPS]
        if steps:
            return steps, False
    return [question], True


# ---------------------------------------------------------------- worker
def job_prompt(instruction, question, chunk):
    return (
        "You are reading ONE excerpt of a patient's medical records. Do the "
        "subtask using only this excerpt. If the excerpt does not contain "
        'what is asked, answer exactly "none".\n\n'
        f"Subtask: {instruction}\n"
        f"(Overall question, for context: {question})\n\n"
        f"Excerpt:\n{chunk}\n\n"
        'Reply as JSON: {"answer": "<short answer or none>", '
        '"quote": "<verbatim supporting quote, or empty>"}'
    )


def parse_job(text):
    """-> (answer or None, quote, ok). answer None means 'none'/no finding."""
    data = extract_json(text)
    if not isinstance(data, dict) or "answer" not in data:
        return None, "", False
    ans = str(data.get("answer") or "").strip()
    quote = str(data.get("quote") or "").strip()
    if not ans or ans.lower().strip(" .\"'") in ("none", "n/a", "not found"):
        return None, "", True
    return ans, quote, True


def findings(jobs):
    """Human-readable lines for the non-'none' jobs of one step."""
    lines = []
    for j in jobs:
        if j.get("answer"):
            q = f' (quote: "{j["quote"]}")' if j.get("quote") else ""
            lines.append(f"- excerpt {j['chunk_idx']}: {j['answer']}{q}")
    return "\n".join(lines) if lines else "- (no excerpt contained anything for this subtask)"


# ---------------------------------------------------------------- review
def review_prompt(question, instruction, jobs):
    return (
        "You are reviewing one step of an extraction plan. You see only the "
        "worker's short answers and quotes for this step, never the raw "
        "excerpts.\n\n"
        f"Question being answered: {question}\n"
        f"Step: {instruction}\n\n"
        f"Worker findings:\n{findings(jobs)}\n\n"
        "If the findings look complete and correct for this step, accept "
        "them. If something is missing, wrong, or too vague, ask for ONE "
        "rerun with a sharper instruction. Reply as JSON: "
        '{"verdict": "accept"} or {"verdict": "redo", "instruction": '
        '"<sharper instruction>"}'
    )


def parse_review(text):
    """-> (verdict 'accept'|'redo', new_instruction or None, ok)."""
    data = extract_json(text)
    if isinstance(data, dict):
        v = str(data.get("verdict", "")).strip().lower()
        if v == "redo" and str(data.get("instruction") or "").strip():
            return "redo", str(data["instruction"]).strip(), True
        if v == "accept":
            return "accept", None, True
    return "accept", None, False  # unreadable: keep the original, but flag it


# ------------------------------------------------------------- aggregate
def aggregate_prompt(question, options, steps):
    """steps: [(instruction, jobs)] with the answers chosen by the protocol."""
    opts = "\n".join(f"({L}) {options[L]}" for L in "ABCDE")
    blocks = "\n\n".join(
        f"Step {i+1}: {instr}\n{findings(jobs)}"
        for i, (instr, jobs) in enumerate(steps))
    return (
        "Answer a multiple-choice question about a patient's medical "
        "records. You cannot see the records; use only the findings below.\n\n"
        f"Question: {question}\n{opts}\n\n"
        f"Findings:\n{blocks}\n\n"
        'Reply as JSON with the letter of the correct option: {"letter": "B"}'
    )


def parse_letter(text):
    """-> (letter or None, how). JSON first, then the rule-based grader."""
    data = extract_json(text)
    if isinstance(data, dict):
        letter = grader.normalize_letter(data.get("letter"))
        if letter:
            return letter, "json"
    letter, _ = grader.extract_letter_fixed(text or "")
    return letter, "rule-fallback"


def fingerprint():
    """Short hash of all pipeline prompt wordings (for config ids/resume)."""
    opts = {L: f"OPT{L}" for L in "ABCDE"}
    blob = "\n".join([
        plan_prompt("Q", opts), job_prompt("I", "Q", "C"),
        review_prompt("Q", "I", []), aggregate_prompt("Q", opts, [("I", [])]),
        *PARAPHRASES])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]
