"""Deterministic mock backends and synthetic records for the pipeline.

Used by the unit tests and by `run_pipeline.py --mock` (a dry run that
exercises every script with no model, key, or benchmark). Numbers from
mocks are SIMULATED and must never be reported.

The synthetic world, built so each mechanism has something to do:
  q_easy    the evidence is in one excerpt, confident answer      -> correct
            with no review at all
  q_hidden  the first instruction finds nothing; only a sharper rerun
            ("redo") finds the evidence                           -> needs a
            review; the log-prob signal CANNOT flag it (every excerpt said
            'none', so there is no log-prob to be low)
  q_noisy   a low-confidence WRONG answer that a rerun corrects  -> the
            log-prob and agreement signals both flag it
"""
import re

LETTERS = "ABCDE"


def _opts(correct):
    return {L: f"option {L}" for L in LETTERS}


def synthetic_records():
    base = {"options": _opts(None), "patient": "mock", "q_no": 0}
    recs = [
        {**base, "q_idx": 0, "question": "Which option is documented? (easy)",
         "gold": "B",
         "documents": ["intro text", "FACT: letter=B", "closing text"]},
        {**base, "q_idx": 1, "question": "Which option is documented? (hidden)",
         "gold": "D",
         "documents": ["intro text", "HIDDEN: letter=D", "closing text"]},
        {**base, "q_idx": 2, "question": "Which option is documented? (noisy)",
         "gold": "C",
         "documents": ["intro text", "NOISY FACT: letter=E", "closing text"]},
    ]
    return recs


def mock_chunker(text):
    """One chunk per document (documents are joined with this separator)."""
    return text.split("\n\n---\n\n")


class MockWorker:
    kind = "mock-worker"

    def __init__(self, model="mock-4b"):
        self.model = model

    def params(self):
        return {}

    def call(self, prompt, logprobs=False):
        m = re.search(r"Subtask: (.*)\n", prompt)
        instr = m.group(1) if m else ""
        m = re.search(r"Excerpt:\n(.*)\n\nReply as JSON", prompt, re.DOTALL)
        excerpt = m.group(1) if m else ""
        sharper = "sharper" in instr
        answer, lp = "none", -0.05
        if "NOISY" in excerpt:
            lp = -2.0
            if sharper:
                answer = "letter=C"
            elif instr.startswith("Look through"):
                answer = "letter=E (a)"  # paraphrase 1 words it differently
            elif instr.startswith("From the excerpt only"):
                answer = "letter=E (b)"  # paraphrase 2 too -> runs disagree
            else:
                answer = "letter=E"
        elif "HIDDEN:" in excerpt and sharper:
            answer = "letter=D"
        elif "FACT:" in excerpt:
            answer = re.search(r"FACT: (letter=\w)", excerpt).group(1)
        text = '{"answer": "%s", "quote": "%s"}' % (
            answer, "" if answer == "none" else "from excerpt")
        return {"text": text, "mean_logprob": lp if logprobs else None,
                "prompt_tokens": len(prompt) // 4, "completion_tokens": 12,
                "thinking_tokens": 0, "wall_time_s": 0.5}


class MockSupervisor:
    kind = "mock-supervisor"

    def __init__(self, model="mock-flash"):
        self.model = model

    def params(self):
        return {}

    def call(self, prompt, json_mode=True):
        if prompt.startswith("You are planning"):
            text = ('["find the documented fact", "find any hidden detail", '
                    '"list relevant notes"]')
        elif prompt.startswith("You are reviewing"):
            findings = prompt.split("Worker findings:\n", 1)[1]
            if "no excerpt contained anything" in findings or "letter=E" in findings:
                text = ('{"verdict": "redo", "instruction": '
                        '"sharper: search again for the documented letter"}')
            else:
                text = '{"verdict": "accept"}'
        else:  # aggregate
            findings = prompt.split("Findings:\n", 1)[1]
            letters = re.findall(r"letter=([A-E])", findings)
            text = '{"letter": "%s"}' % (letters[-1] if letters else "A")
        return {"text": text, "prompt_tokens": len(prompt) // 4,
                "completion_tokens": 20, "thinking_tokens": 50,
                "finish_reason": "STOP", "wall_time_s": 0.2}
