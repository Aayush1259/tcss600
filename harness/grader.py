"""Grader for the Week 3 baselines.

Rule-based letter extraction, fixed after the Oct 14 pilot bug.

The bug: the old extractor grabbed the first '(A)' in the text, which was
the options list, not the supervisor's stated choice. The pilot first
scored 6/10 because of it. The fix: look for choice-indicating phrases
("the correct option is (B)", "matches option (D)", ...) and prefer the
LAST match in the text, because the model states its choice at the end,
after the options list.

Oct 17 follow-up: the direct 27B worker (no supervisor) states its choice
as bold markdown at the very start ("**A**"), not "(A)". A leading
standalone choice ("**A**" or "(B)" on the first line) is checked first,
since it is the worker's final answer, not a distractor.

Rows the rule-based grader cannot read return letter=None and are flagged
for the hand-check queue. The hardened API grader (grade_hardened.py)
remains available as a cross-check, not the primary path, so grading does
not burn supervisor budget.

Oct 20 follow-up (v2): all 22 flagged warmup rows were hand-checked
(docs/handcheck-results.md). Twelve think-off answers and five others
stated their choice in formats v1 had no pattern for: a bare trailing
letter, a bare "(C)" line, "(A) <option text>" as the final line,
"The answer is **(C)**", and "**Correct Answer:** **(A)**". v2 adds
patterns for those formats. The Week 3 report's numbers stand as
published (v1 plus hand-checks); v2 exists so the 200-question eval runs
do not produce an unusable hand-check queue. Regrading the warmup logs
with v2 and reporting both gradings is a Week 4 disclosure item.
"""
import re

# A bold choice stated alone at the very start of the answer ("**A**").
# Checked before the pool below. Note: a parenthesized "(A)" at the start
# is the OPTIONS LIST, not the answer (the original Oct 14 bug), so the
# leading check is bold-only.
LEADING_CHOICE = re.compile(r"^\s*\*\*([A-E])\*\*")

# The entire answer is one bare letter ("C"). Safe: a real prose answer
# is never exactly one letter. Seen in the 27B warmup run, Oct 17.
WHOLE_LETTER = re.compile(r"^\s*([A-E])\s*\.?\s*$")

# The entire answer is one parenthesized letter ("(B)"). Same safety
# argument. Seen in the think-on warmup run, Oct 18 (v2, Oct 20).
WHOLE_PAREN = re.compile(r"^\s*\(([A-E])\)\s*\.?\s*$")

# Ordered by specificity; all matches are collected and the latest
# position in the text wins.
CHOICE_PATTERNS = [
    # Bold-tolerant choice phrases. Covers the supervisor's formats:
    # "The correct option is **(A)**." and
    # "The correct option is **(A) Multiple sclerosis**."
    r"correct (?:option|choice|answer) is \*\*\(?([A-E])\)?",
    r"correct (?:option|choice|answer) is \(([A-E])\)",
    # Real pilot phrasing: "the correct option corresponding to the
    # location where osteolysis was detected is (B)." The qualifier
    # between 'option' and 'is' stays within one sentence (no period).
    r"correct option[^.]{0,80}?is \(([A-E])\)",
    r"the correct choice is \(([A-E])\)",
    r"matches option \(([A-E])\)",
    r"[Aa]nswer:\s*\(?([A-E])\)?",
    # "## Answer: **(C) CDKN2A/B deletion**" (27B warmup, Oct 17)
    r"[Aa]nswer:\s*\*\*\(?([A-E])\)?",
    r"choice is \(([A-E])\)",
    r"option \(([A-E])\)",
    # Direct worker format: **A**
    r"\*\*([A-E])\*\*",
    # v2 (Oct 20), from the warmup hand-checks:
    # "The answer is **(C)**." (think-on warmup)
    r"[Aa]nswer is \*\*\(?([A-E])\)?",
    r"[Aa]nswer is \(([A-E])\)",
    # Cloud-only format: "**Correct Answer:** **(A)**"
    r"[Cc]orrect [Aa]nswer:\s*\*\*\s*\*\*\(?([A-E])\)?",
    r"[Cc]orrect [Aa]nswer:\s*\(?([A-E])\)?",
]

# v2 fallback (Oct 20), used only when no choice phrase matched anywhere:
# the conclusion as the FINAL line of the answer. Formats proven by the
# hand-checks: a bare letter ("B"), a bare "(C)", or "(A) <option text>".
# Only the last non-empty line is read, so an options list earlier in the
# answer can never win. When text follows the letter, parentheses are
# required, so a prose line that merely starts with a capital letter
# ("A total of 5 mg was given.") cannot match.
TRAILING_BARE = re.compile(r"^\(?([A-E])\)?$")
TRAILING_PAREN_TEXT = re.compile(r"^\(([A-E])\)\s+\S.{0,78}$")


def _trailing_line_letter(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1].strip("*").strip()
    m = TRAILING_BARE.match(last) or TRAILING_PAREN_TEXT.match(last)
    return m.group(1) if m else None


def extract_letter_fixed(text):
    """Returns (letter_or_None, method). method is 'rule' or 'manual-review'."""
    m = WHOLE_LETTER.match(text) or WHOLE_PAREN.match(text)
    if m:
        return m.group(1), "rule"
    m = LEADING_CHOICE.match(text)
    if m:
        return m.group(1), "rule"
    hits = []
    for pat in CHOICE_PATTERNS:
        for m in re.finditer(pat, text):
            hits.append((m.start(), m.group(1)))
    if hits:
        hits.sort()
        return hits[-1][1], "rule"
    letter = _trailing_line_letter(text)
    if letter:
        return letter, "rule"
    return None, "manual-review"


def matches(letter, gold, options):
    """Same matching logic the pilot's hardened grader used.

    Gold is sometimes a bare letter/combo ('B', 'B and C') and sometimes
    the full option text ('Neutropenic fever').
    """
    if not letter:
        return False
    g = str(gold).strip()
    if len(g) <= 8:
        return letter in g
    return options.get(letter, "").strip().lower() == g.lower()


def grade(text, gold, options):
    """Returns dict(letter, match, needs_handcheck, method)."""
    letter, method = extract_letter_fixed(text)
    return {
        "letter": letter,
        "match": matches(letter, gold, options),
        "needs_handcheck": letter is None,
        "method": method,
    }


if __name__ == "__main__":
    # Regression checks from the real pilot outputs (Oct 13-14).
    cases = [
        # (text snippet, expected letter)
        ("the correct option corresponding to the location is (B).", "B"),
        ("it corresponds to option (B).\n\n(B)", "B"),
        ("the correct choice is (C).", "C"),
        ("This matches option (D).\n\n(D)", "D"),
        # Options list at the top must NOT win:
        ("(A) Luxation of the shoulder\n(B) Fracture\nTherefore, the correct option is (E).", "E"),
        ("no choice stated anywhere here", None),
        # Direct worker format (Oct 17): bold letter at the very start.
        ("**A**\n\nThe medical records consistently identify", "A"),
        # gemini-3.8-flash supervisor format (Oct 17).
        ("The correct option is **(A)**.", "A"),
        ("The correct option is **(A) Multiple sclerosis**.", "A"),
        # 27B warmup formats (Oct 17): bare letter, ## Answer: **(C) text**.
        ("C", "C"),
        ("A\n", "A"),
        ("## Answer: **(C) CDKN2A/B deletion**\n\n### Explanation", "C"),
        # v2 formats (Oct 20), verbatim classes from the warmup hand-checks:
        ("(B)", "B"),
        ("(C)", "C"),
        ("Long reasoning about the records.\n\nB", "B"),
        ("Long reasoning about the records.\n\n(C)", "C"),
        ("Long reasoning about the records.\n\n(A) Multiple sclerosis", "A"),
        ("Therefore, the correct option is:\n\n(C)", "C"),
        ("The answer is **(C)**.", "C"),
        ("**Correct Answer:** **(A)**", "A"),
        # v2 guards: deliberation and prose must still return None.
        ("Option A includes Brain.\nOption D excludes Heart.\nOften 'showed", None),
        ("(A) Luxation\n(B) Fracture\nThe discussion continues and never picks one", None),
        ("A total of 5 mg was given daily.", None),
    ]
    ok = True
    for text, exp in cases:
        got, method = extract_letter_fixed(text)
        status = "OK " if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(f"{status} expected={exp} got={got} ({method}) :: {text[:60]!r}")
    print("ALL OK" if ok else "FAILURES PRESENT")
