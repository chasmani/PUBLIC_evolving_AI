"""Figure 2: accuracy and token use by round, for each condition.

Mean across the ten runs of each condition, shaded +/- 1 standard deviation.

  python3 fig2_trajectories.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C                                              # noqa: E402

OUT = Path(__file__).parent / "fig2_trajectories"


def main():
    rl = C.round_level(C.load_trials())
    plt.rcParams.update({"font.size": 14, "axes.spines.right": False,
                         "axes.spines.top": False})
    fig, (ax_acc, ax_tok) = plt.subplots(1, 2, figsize=(12, 5))

    for cond in C.CONDITIONS:
        sub = rl[rl.condition == cond]
        for ax, value, clip in ((ax_acc, "accuracy", (0, 1)),
                                (ax_tok, "tokens", None)):
            g = sub.groupby("round")[value]
            rounds, mean, sd = g.mean().index.to_numpy(), g.mean().to_numpy(), g.std().to_numpy()
            lo, hi = mean - sd, mean + sd
            if clip:
                lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
            ax.fill_between(rounds, lo, hi, color=C.STYLE[cond]["color"], alpha=0.2)
            ax.plot(rounds, mean, label=f"{C.LABEL[cond]} Condition",
                    linewidth=2.5, markersize=6, **C.STYLE[cond])

    ax_acc.set(xlabel="Round", ylabel="Accuracy", ylim=(-0.02, 1.02))
    ax_tok.set(xlabel="Round", ylabel="Tokens Used")
    ax_tok.set_ylim(bottom=0)

    # legend ordered as the curves appear at the final round
    handles, labels = ax_tok.get_legend_handles_labels()
    order = [labels.index(f"{C.LABEL[c]} Condition") for c in C.LEGEND_ORDER]
    ax_tok.legend([handles[i] for i in order], [labels[i] for i in order],
                  frameon=False)

    for ax, letter in ((ax_acc, "a"), (ax_tok, "b")):
        ax.text(0.02, 1.0, letter, transform=ax.transAxes, fontsize=32,
                fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
