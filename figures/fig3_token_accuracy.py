"""
Figure 3: the token-accuracy space that the two markets move through.

Every agent that completed at least MIN_TASKS tasks contributes one point,
positioned by its lifetime mean output tokens and its lifetime accuracy. That
cloud is the region of behaviour the base model reaches under the prompts that
actually evolved -- it is a sample of the space, not an estimated frontier, and
no curve is fitted to it.

Over the cloud, each market's population mean per round is drawn as a
trajectory from round 1 to round 25. Direction of travel is shown by rotated
dart markers, one per round, so the spacing between darts also shows how fast
the population was moving: closely spaced darts mean the population has
arrived somewhere and is jittering rather than travelling. Both markets start
from the same population (black square) and move to opposite regions, each
pulled along a different axis -- the accuracy condition prices accuracy and
leaves tokens free, the token condition prices tokens and leaves accuracy
unchecked.

Outputs:
  token_accuracy.png / .pdf

  python3 fig3_token_accuracy.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C                                              # noqa: E402

OUT_STEM = Path(__file__).parent / "fig3_token_accuracy"

REGIMES = C.CONDITIONS
LABELS = {c: f"{C.LABEL[c]} Condition" for c in C.CONDITIONS}
LEGEND_ORDER = C.LEGEND_ORDER
COLORS = {c: C.STYLE[c]["color"] for c in C.CONDITIONS}

MIN_TASKS = 1        # every agent that answered at least once
DART_SIZE = 13       # marker size in points
LINE_WIDTH = 2.0

# A dart: an isosceles triangle with a notched base, so it reads as an arrowhead
# rather than an ambiguous triangle, while staying a filled shape that survives
# overlapping itself where trajectories double back. Defined pointing up (+y);
# each instance is rotated to the local direction of travel.
DART = MplPath(
    [(0, 1.05), (-0.6, -0.75), (0, -0.3), (0.6, -0.75), (0, 1.05)],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO,
     MplPath.CLOSEPOLY],
)


def load_trials():
    """Every agent-round of every seed of every condition, as one dataframe."""
    return C.load_trials().rename(columns={"condition": "regime"})


def agent_means(df, min_tasks=MIN_TASKS):
    """One row per agent: lifetime mean output tokens and lifetime accuracy."""
    g = C.agent_level(df.rename(columns={"regime": "condition"}), min_tasks)
    return g.rename(columns={"condition": "regime"})


def round_means(df):
    """One row per regime and round: the population mean on both axes."""
    return (df.groupby(["regime", "round"])
              .agg(tokens=("output_tokens", "mean"), acc=("correct", "mean"))
              .reset_index())


def draw_trajectory(ax, sub, colour):
    """Line through the per-round population means, with a dart at each round.

    Dart angles are computed in display coordinates, not data coordinates: the
    axes span ~600 tokens by 1.0 accuracy, so an angle taken in data space would
    point visibly wrong once matplotlib scales the axes.
    """
    sub = sub.reset_index(drop=True)
    ax.plot(sub.tokens, sub.acc, color=colour, lw=LINE_WIDTH, zorder=4)
    pts = ax.transData.transform(np.c_[sub.tokens, sub.acc])
    for i in range(len(sub)):
        nxt = min(i + 1, len(sub) - 1)              # forward difference, except
        prv = i if i < len(sub) - 1 else i - 1      # at the last point
        dx, dy = pts[nxt, 0] - pts[prv, 0], pts[nxt, 1] - pts[prv, 1]
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx)) - 90   # the dart points up at 0 deg
        ax.plot(sub.tokens[i], sub.acc[i], ls="none", ms=DART_SIZE, zorder=6,
                marker=DART.transformed(Affine2D().rotate_deg(angle)),
                mfc=colour, mec="white", mew=.8)


def legend_handles(present):
    """Proxy artists: the real marks are drawn per-point and per-regime."""
    ordered = [r for r in LEGEND_ORDER if r in present]
    return [
        *[Line2D([], [], ls="none", marker="o", ms=5, mec="none",
                 mfc=COLORS[r], label=LABELS[r]) for r in ordered],
        Line2D([], [], ls="none", marker=DART, ms=11,
               mfc="0.35", mec="white", mew=.8, label="Mean per Round"),
        Line2D([], [], ls="none", marker="s", ms=7.5, mfc="black", mec="white",
               mew=1.0, label="Round 1"),
    ]


def main():
    df = load_trials()
    present = [r for r in REGIMES if (df.regime == r).any()]
    agents = agent_means(df)
    per_round = round_means(df)

    plt.rcParams.update({"font.size": 14, "axes.spines.right": False,
                         "axes.spines.top": False})
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    for regime in present:
        s = agents[agents.regime == regime]
        ax.scatter(s.mean_tokens, s.accuracy, s=14, alpha=.25,
                   c=COLORS[regime], edgecolors="none")
    for regime in present:
        ax.plot(per_round[per_round.regime == regime].tokens,
                per_round[per_round.regime == regime].acc,
                color=COLORS[regime], lw=LINE_WIDTH, zorder=4)

    ax.set(xlabel="Tokens Used", ylabel="Accuracy")
    ax.set_ylim(-.03, 1.03)
    ax.set_xlim(30, 690)

    # The layout must be final before display-space angles are computed.
    fig.tight_layout()
    fig.canvas.draw()

    for regime in present:
        draw_trajectory(ax, per_round[per_round.regime == regime], COLORS[regime])

    # Both markets begin from the same population, so a single neutral marker at
    # the mean of the two round-1 points states that more clearly than two.
    start = per_round[per_round["round"] == 1]
    ax.plot(start.tokens.mean(), start.acc.mean(), ls="none", marker="s", ms=9,
            mfc="black", mec="white", mew=1.0, zorder=7)

    ax.legend(handles=legend_handles(present), fontsize=11, loc="upper left",
              frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=300)
    print(f"agents plotted: {len(agents)} (>= {MIN_TASKS} tasks each)")
    print(f"round 1 at {start.tokens.mean():.1f} tokens, {start.acc.mean():.3f} accuracy")
    print(f"wrote {OUT_STEM}.png and {OUT_STEM}.pdf")


if __name__ == "__main__":
    main()
