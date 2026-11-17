"""The cost versus accuracy chart for Question A (amendment 2, section 7). Read-only.

Reads only the JSON written by analyze_a.py and draws two panels for the 91 record-dependent questions.

Panel A, cost and accuracy: x = cloud dollars per question (protocol accounting; 4B alone at 0; a
linear axis), y = accuracy on the 91, one point per arm, with the 95 percent patient cluster interval
of the arm's difference from cloud alone drawn around the arm's point, and a line at cloud alone's
accuracy minus 10 points (the margin of the November 12 rule). Labels carry the laptop minutes per
question where the JSON has them. Three hues only (cloud alone, plan once, check every step); 4B
alone, which uses no cloud, is a gray reference.

Panel B, question by question against cloud alone: for each arm, the 91 split into both right,
lost (cloud alone right, arm wrong), rescued (arm right, cloud alone wrong) and both wrong, from the
JSON's arm_correct, only_arm_right and only_cloud_alone_right. Every segment carries its count.

A table with the same numbers is written beside the image, so no reading depends on color.

    python chart_a.py --results question_a_results.json --out question_a_chart
writes question_a_chart.png, question_a_chart.svg and question_a_chart_table.md.
"""
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, TEXT, TEXT2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# panel A: categorical slots 1 to 3 of the validated palette (all-pairs safe), fixed per arm; 4B alone gray
COLOR = {"cloud_alone": "#2a78d6", "plan_once": "#eb6834", "check_every_step": "#1baf7a", "local_alone": TEXT2,
         "triggered_logprob_mean": MUTED}
LABEL = {"cloud_alone": "Cloud alone", "plan_once": "Plan once", "check_every_step": "Check every step",
         "local_alone": "4B alone", "triggered_logprob_mean": "Triggered log-prob"}
MARKER = {"cloud_alone": "s", "plan_once": "o", "check_every_step": "D", "local_alone": "^",
          "triggered_logprob_mean": "v"}
# label placement: (dx, dy, horizontal alignment, vertical alignment, anchor) so no label crosses another mark
OFFSET = {"cloud_alone": (-9, 0, "right", "center", "point"), "plan_once": (0, 5, "center", "bottom", "bar_high"),
          "check_every_step": (9, 0, "left", "center", "point"), "local_alone": (4, -5, "left", "top", "bar_low"),
          "triggered_logprob_mean": (4, -5, "left", "top", "bar_low")}
# three hues (all-pairs safe) for the arms the verdict is about; the two context arms in neutral grays
ON_A = ("cloud_alone", "plan_once", "check_every_step", "triggered_logprob_mean", "local_alone")
# panel B: outcome slots 7, 8, 4, 5 of the validated palette (adjacent order passes every hard gate)
OUTCOMES = [("both_right", "Both right", "#6250d6", "#ffffff"),
            ("lost", "Lost: cloud alone right, arm wrong", "#e34948", "#ffffff"),
            ("rescued", "Rescued: arm right, cloud alone wrong", "#eda100", "#0b0b0b"),
            ("both_wrong", "Both wrong", "#e87ba4", "#0b0b0b")]
ON_B = ("plan_once", "check_every_step", "triggered_logprob_mean", "local_alone")


def points(res):
    comps, cost = res["comparisons"], res.get("cost_and_work", {})
    ref = next(iter(comps.values()))["on_91"]
    n = ref["n_questions"]
    cloud_acc = 100 * ref["cloud_alone_correct"] / n
    any_cost = next(iter(cost.values()))["on_91"] if cost else None
    cloud_usd = any_cost["cloud_alone_usd"] / any_cost["questions"] if any_cost else None
    rows = [{"arm": "cloud_alone", "acc": cloud_acc, "correct": ref["cloud_alone_correct"], "lo": None, "hi": None,
             "usd": cloud_usd, "secs": None, "verdict": "reference", "n": n}]
    for name, c in comps.items():
        p = c["on_91"]
        acc = 100 * p["arm_correct"] / p["n_questions"]
        lo, hi = p["interval_points"]
        if name == "local_alone":
            usd, secs = 0.0, None
        else:
            k = cost.get(name, {}).get("on_91", {})
            usd = k["arm_cloud_usd"] / k["questions"] if k.get("questions") else None
            secs = k.get("laptop_seconds_per_question")
        both_right = p["arm_correct"] - p["only_arm_right"]
        rows.append({"arm": name, "acc": acc, "correct": p["arm_correct"], "lo": cloud_acc + lo, "hi": cloud_acc + hi,
                     "usd": usd, "secs": secs, "verdict": p.get("verdict") or "no verdict", "n": p["n_questions"],
                     "both_right": both_right, "lost": p["only_cloud_alone_right"], "rescued": p["only_arm_right"],
                     "both_wrong": p["n_questions"] - both_right - p["only_cloud_alone_right"] - p["only_arm_right"]})
    return rows, cloud_acc


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT2, labelsize=8)


def panel_a(ax, rows, cloud_acc):
    style(ax)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(cloud_acc - 10, color=TEXT2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.99, cloud_acc - 10, "margin: cloud alone minus 10 points", color=TEXT2, fontsize=7.5,
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    for r in rows:
        if r["usd"] is None or r["arm"] not in ON_A:
            continue
        if r["lo"] is not None:
            ax.vlines(r["usd"], r["lo"], r["hi"], color=COLOR[r["arm"]], linewidth=2, zorder=2)
        ax.scatter([r["usd"]], [r["acc"]], s=64, marker=MARKER[r["arm"]], color=COLOR[r["arm"]],
                   edgecolor=SURFACE, linewidth=2, zorder=3, label=LABEL[r["arm"]])
        dx, dy, ha, va, anchor = OFFSET[r["arm"]]
        y = {"bar_high": r["hi"], "bar_low": r["lo"]}.get(anchor) if r["lo"] is not None else None
        y = r["acc"] if y is None else y
        extra = ""
        if r["arm"] == "cloud_alone":
            extra = "\nno laptop time"
        elif r["arm"] == "local_alone":
            extra = "\nno cloud calls"
        elif r["secs"]:
            extra = f"\n{r['secs'] / 60:.1f} laptop min per question"
        ax.annotate(f"{LABEL[r['arm']]} {r['correct']} of {r['n']} ({r['acc']:.0f}%){extra}", (r["usd"], y),
                    xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=TEXT, ha=ha,
                    va=va, zorder=4, linespacing=1.3)
    ax.set_xlabel("Cloud dollars per question (protocol accounting)", color=TEXT, fontsize=8.5)
    ax.set_ylabel("Accuracy on the 91 (%)", color=TEXT, fontsize=8.5)
    ax.set_xlim(left=-0.001)
    ax.set_ylim(0, 100)
    cloud_usd = next(r["usd"] for r in rows if r["arm"] == "cloud_alone")
    shares = [r["usd"] / cloud_usd for r in rows if r["arm"] in ("plan_once", "check_every_step") and r["usd"]]
    ax.set_title(f"A. Accuracy against cloud cost: plan once and check every step lie far below the margin,\n"
                 f"at {min(shares):.0%} and {max(shares):.0%} of cloud alone's cloud dollars",
                 color=TEXT, fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=TEXT)


def panel_b(ax, rows):
    style(ax)
    by = {r["arm"]: r for r in rows}
    arms = [a for a in ON_B if a in by]
    n = by[arms[0]]["n"]
    for i, a in enumerate(arms):
        left = 0
        for key, _, color, ink in OUTCOMES:
            w = by[a][key]
            if w:
                ax.barh(i, w, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=2)
                if w >= 3:  # a sliver is too thin for its number; the note at the right gives it
                    ax.text(left + w / 2, i, str(w), ha="center", va="center", fontsize=7.5, color=ink)
            left += w
        ax.text(n + 1.2, i, f"rescued {by[a]['rescued']}, lost {by[a]['lost']}", ha="left", va="center",
                fontsize=7.5, color=TEXT2)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([LABEL[a] for a in arms], color=TEXT, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, n + 22)
    ax.set_xticks([0, 20, 40, 60, 80, n])
    ax.set_xlabel(f"Questions of the {n}", color=TEXT, fontsize=8.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    pipe = [by[a] for a in arms if a != "local_alone"]
    lost = sorted({r["lost"] for r in pipe})
    resc = sorted({r["rescued"] for r in pipe})
    cloud_miss = by[arms[0]]["rescued"] + by[arms[0]]["both_wrong"]
    lost_txt = f"{lost[0]} to {lost[-1]}" if len(lost) > 1 else str(lost[0])
    resc_txt = f"{resc[0]} to {resc[-1]}" if len(resc) > 1 else str(resc[0])
    ax.set_title(f"B. Against cloud alone, question by question: the supervised arms lose {lost_txt} "
                 f"and rescue {resc_txt}\n(cloud alone missed {cloud_miss} of the {n}, so at most "
                 f"{cloud_miss} rescues were possible)", color=TEXT, fontsize=9.5, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in OUTCOMES]
    ax.legend(handles, [lab for _, lab, _, _ in OUTCOMES], frameon=False, fontsize=7.5, ncol=2,
              loc="upper center", bbox_to_anchor=(0.45, -0.2), labelcolor=TEXT)


def draw(rows, cloud_acc, out):
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.6, 8.4), dpi=200, gridspec_kw={"height_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)
    panel_a(ax_a, rows, cloud_acc)
    panel_b(ax_b, rows)
    fig.text(0.01, 0.005, "4B worker qwen3:4b, supervisor gemini-3.8-flash, LongHealth at N = 2, one run per arm. "
             "Panel A bars: cloud alone's accuracy plus the 95% patient cluster\ninterval of the arm's difference "
             "from cloud alone; a bar wholly below the dashed line means worse. Laptop minutes: the arm's summed "
             "worker calls.", color=TEXT2, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2)
    fig.savefig(out + ".png", facecolor=SURFACE)
    fig.savefig(out + ".svg", facecolor=SURFACE)
    plt.close(fig)


def table(rows, out):
    lines = ["| arm | correct of 91 | accuracy on the 91 | interval drawn | cloud dollars per question | "
             "laptop minutes per question | verdict | both right | lost | rescued | both wrong |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        iv = f"{r['lo']:.1f} to {r['hi']:.1f}" if r["lo"] is not None else "reference"
        usd = f"{r['usd']:.4f}" if r["usd"] is not None else "n/a"
        mins = f"{r['secs'] / 60:.1f}" if r["secs"] else "n/a"
        parts = " | ".join(str(r.get(k, "")) for k in ("both_right", "lost", "rescued", "both_wrong"))
        lines.append(f"| {LABEL[r['arm']]} | {r['correct']} | {r['acc']:.1f}% | {iv} | {usd} | {mins} | "
                     f"{r['verdict']} | {parts} |")
    with open(out + "_table.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.results, encoding="utf-8") as f:
        rows, cloud_acc = points(json.load(f))
    draw(rows, cloud_acc, args.out)
    table(rows, args.out)
    print(f"written: {args.out}.png, {args.out}.svg, {args.out}_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
