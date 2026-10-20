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
]


def extract_letter_fixed(text):
    """Returns (letter_or_None, method). method is 'rule' or 'manual-review'."""
    m = WHOLE_LETTER.match(text)
    if m:
        return m.group(1), "rule"
    m = LEADING_CHOICE.match(text)
    if m:
        return m.group(1), "rule"
    hits = []
    for pat in CHOICE_PATTERNS:
        for m in re.finditer(pat, text):
            hits.append((m.start(), m.group(1)))
    if not hits:
        return None, "manual-review"
    hits.sort()
    return hits[-1][1], "rule"


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
    ]
    ok = True
    for text, exp in cases:
        got, method = extract_letter_fixed(text)
        status = "OK " if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(f"{status} expected={exp} got={got} ({method}) :: {text[:60]!r}")
    print("ALL OK" if ok else "FAILURES PRESENT")
