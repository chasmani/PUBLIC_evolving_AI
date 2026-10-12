"""Figure 4: lineage structure.

Top row: the descent of every agent in the seed 0 run of each condition. Each
horizontal line is one agent, spanning the rounds in which it answered; vertical
connectors join a parent to its offspring at the round the offspring was created.
Colour marks the founding lineage. Bottom: the number of founding lineages still
represented in the population, by round, averaged over the ten runs.

  python3 fig4_lineages.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as rc

OUT = Path(__file__).parent / "fig4_lineages"
EXAMPLE_SEED = "seed00"
STYLE = {"verified": dict(color="#4C72B0", marker="s", linestyle="dotted"),
         "engagement": dict(color="#C44E52", marker="o", linestyle="-"),
         "neutral": dict(color="#808080", marker="^", linestyle="dashed")}
ORDER = ["verified", "neutral", "engagement"]          # legend / panel order
# three tones only: the lineage that ends largest, other lineages still
# present at the final round, and lineages that leave no descendants.
# Greyscale throughout, so colour in this paper only ever means condition.
WINNER = "#1A1A1A"
SURVIVOR = "#8F8F8F"
EXTINCT = "#D8D8D8"


def founder_map(lin):
    par = {a["id"]: a["parent_id"] for a in lin}
    root = {}
    def f(i):
        if i in root:
            return root[i]
        p = par[i]
        root[i] = i if p is None else f(p)
        return root[i]
    return {i: f(i) for i in par}, par


def spans(df, cond, seed):
    """{agent_id: (first_round, last_round)} for one run."""
    s = df[(df.condition == cond) & (df.seed == seed)]
    g = s.groupby("agent_id")["round"].agg(["min", "max"])
    return {int(i): (int(r["min"]), int(r["max"])) for i, r in g.iterrows()}


def living_by_round(df, cond, seed):
    s = df[(df.condition == cond) & (df.seed == seed)]
    return {rd: set(grp.agent_id) for rd, grp in s.groupby("round")}


def draw_tree(ax, df, cond, seed):
    lin = rc.load_lineage(cond, seed)
    root, par = founder_map(lin)
    sp = spans(df, cond, seed)
    kids = defaultdict(list)
    for a in lin:
        if a["parent_id"] is not None:
            kids[a["parent_id"]].append(a["id"])

    founders = sorted(i for i, p in par.items() if p is None)
    # colour founders by the size of their final descendancy, largest first,
    # so the dominant lineage is always the same colour across panels
    alive = {a["id"] for a in lin if a["alive"]}
    size = defaultdict(int)
    for i in alive:
        size[root[i]] += 1
    founders.sort(key=lambda f: (-size.get(f, 0), f))
    colour = {}
    for k, f in enumerate(founders):
        if not size.get(f, 0):
            colour[f] = EXTINCT
        elif k == 0:
            colour[f] = WINNER
        else:
            colour[f] = SURVIVOR

    y = {}
    counter = [0]
    def place(node):                     # depth-first: offspring next to parent
        if node in sp:
            y[node] = counter[0]
            counter[0] += 1
        for c in sorted(kids.get(node, [])):
            place(c)
    for f in founders:
        place(f)

    for node, (r0, r1) in sp.items():
        ax.hlines(y[node], r0 - 0.5, r1 + 0.5, color=colour[root[node]],
                  linewidth=1.4,
                  zorder=3 if colour[root[node]] == WINNER else 2)
    for node, children in kids.items():
        if node not in y:
            continue
        for c in children:
            if c in y:
                ax.vlines(sp[c][0] - 0.5, y[node], y[c],
                          color=colour[root[c]], linewidth=0.6, alpha=.8, zorder=1)
    ax.set_xlim(0, 26)
    ax.set_ylim(-1, counter[0])
    ax.set_yticks([])
    ax.set_xticks([1, 25])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def main():
    plt.rcParams.update({"font.size": 11, "axes.spines.right": False,
                         "axes.spines.top": False})
    df = rc.load_trials()
    fig = plt.figure(figsize=(7.0, 5.2))
    gs = GridSpec(2, 3, height_ratios=[1.05, 1.0], hspace=0.52, wspace=0.12,
              left=.09, right=.99, top=.92, bottom=.09)

    for col, cond in enumerate(ORDER):
        ax = fig.add_subplot(gs[1, col])
        draw_tree(ax, df, cond, EXAMPLE_SEED)
        ax.set_title(f"{rc.LABEL[cond]} Condition", fontsize=11, pad=14)
        if col == 0:
            ax.set_ylabel("Agents")
        ax.set_xlabel("Round")
        ax.text(-0.04, 1.26, "bcd"[col], transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="top", ha="left")

    ax = fig.add_subplot(gs[0, :])
    for cond in ORDER:
        series = []
        for sd in rc.seeds(cond):
            lin = rc.load_lineage(cond, sd)
            root, _ = founder_map(lin)
            lbr = living_by_round(df, cond, sd)
            series.append([len({root[i] for i in lbr[r]})
                           for r in range(1, 26)])
        arr = np.array(series, dtype=float)
        m, sd_ = arr.mean(0), arr.std(0, ddof=1)
        rounds = np.arange(1, 26)
        ax.fill_between(rounds, m - sd_, m + sd_, color=STYLE[cond]["color"],
                        alpha=.18)
        ax.plot(rounds, m, label=f"{rc.LABEL[cond]} Condition", linewidth=2.2,
                markersize=5, **STYLE[cond])
    ax.set(xlabel="Round", ylabel="Surviving founding lineages")
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 21)
    ax.legend(frameon=False, fontsize=10)
    ax.text(-0.007, 1.13, "a", transform=ax.transAxes, fontsize=20,
            fontweight="bold", va="top", ha="left")

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
