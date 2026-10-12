"""Loading and helpers shared by the analysis and figure scripts.

The only place run data is read from is ../results. Nothing here writes to it.

Condition names on disk are the internal names used while running the
experiment; the paper calls them the Accuracy, Token and Neutral conditions:

    verified    -> Accuracy condition   (pays for correct answers)
    engagement  -> Token condition      (charges per token)
    neutral     -> Neutral condition    (no payments, random replacement)
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

CONDITIONS = ["verified", "engagement", "neutral"]
LABEL = {"verified": "Accuracy", "engagement": "Token", "neutral": "Neutral"}
STYLE = {"verified":   dict(color="#4C72B0", marker="s", linestyle="dotted"),
         "engagement": dict(color="#C44E52", marker="o", linestyle="-"),
         "neutral":    dict(color="#808080", marker="^", linestyle="dashed")}
LEGEND_ORDER = ("verified", "neutral", "engagement")


def seeds(cond):
    return sorted(d.name for d in (RESULTS / cond).glob("seed*")
                  if (d / "all_trials.csv").exists())


def load_trials():
    """Every agent-round of every seed of every condition, as one frame."""
    frames = []
    for cond in CONDITIONS:
        for seed in seeds(cond):
            df = pd.read_csv(RESULTS / cond / seed / "all_trials.csv")
            df["condition"], df["seed"] = cond, seed
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["correct"].astype(str).str.strip().str.lower().eq("true")
    return df


def round_level(df):
    """One row per (condition, seed, round): population mean accuracy/tokens."""
    return (df.groupby(["condition", "seed", "round"])
              .agg(accuracy=("correct", "mean"),
                   tokens=("output_tokens", "mean"),
                   n=("agent_id", "size"))
              .reset_index())


def agent_level(df, min_tasks=1):
    """One row per (condition, seed, agent): that agent's mean over its life."""
    g = (df.groupby(["condition", "seed", "agent_id"])
           .agg(mean_tokens=("output_tokens", "mean"),
                accuracy=("correct", "mean"),
                n_tasks=("round", "size"))
           .reset_index())
    return g[g.n_tasks >= min_tasks].copy()


def per_run(rl, rounds, value):
    """Per-run mean of `value` over `rounds`: {condition: array(n_runs)}."""
    sub = rl[rl["round"].isin(rounds)]
    m = sub.groupby(["condition", "seed"])[value].mean().reset_index()
    return {c: m[m.condition == c].sort_values("seed")[value].to_numpy()
            for c in CONDITIONS}


def load_lineage(cond, seed):
    return json.loads((RESULTS / cond / seed / "lineage.json").read_text())


def founder_of(lineage):
    """{agent_id: id of the founding agent it descends from}."""
    parent = {a["id"]: a["parent_id"] for a in lineage}
    root = {}

    def f(i):
        if i not in root:
            p = parent[i]
            root[i] = i if p is None else f(p)
        return root[i]

    return {i: f(i) for i in parent}


def holm(pvals):
    """Holm-Bonferroni step-down. Returns adjusted p-values in input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, (m - rank) * pvals[i])   # enforce monotonicity
        adj[i] = min(running, 1.0)
    return adj


def rank_biserial_mwu(u, n1, n2):
    """Rank-biserial effect size for Mann-Whitney U.

    Positive means the first sample tends to exceed the second.
    """
    return 2.0 * u / (n1 * n2) - 1.0


def rank_biserial_wilcoxon(x, y):
    """Matched-pairs rank-biserial effect size for a signed-rank test."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    pos, neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return (pos - neg) / (pos + neg)
