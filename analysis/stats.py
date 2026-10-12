"""Every number reported in the paper.

Writes
  stats.json        all descriptives, correlations and tests
  stats_table.tex   the Holm-corrected test family, as a LaTeX table

  python3 stats.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C                                              # noqa: E402

HERE = Path(__file__).parent
FIRST, LAST = [1], [25]          # rounds compared; see SI for 3- and 5-round windows

# The family of tests the paper reports, corrected together. Edit this list to
# change what the correction is applied over; nothing else needs changing.
# These are the twelve tests the paper reports. The lineage comparisons are
# also computed below, but the paper describes lineages rather than testing
# them, so they are outside the family and are reported uncorrected.
FAMILY = [
    "within/accuracy/Accuracy", "within/accuracy/Token", "within/accuracy/Neutral",
    "within/tokens/Accuracy", "within/tokens/Token", "within/tokens/Neutral",
    "between/accuracy/Accuracy-Token", "between/accuracy/Accuracy-Neutral",
    "between/accuracy/Token-Neutral",
    "between/tokens/Accuracy-Token", "between/tokens/Accuracy-Neutral",
    "between/tokens/Token-Neutral",
]

PAIRS = [("verified", "engagement"), ("verified", "neutral"),
         ("engagement", "neutral")]


def main():
    df = C.load_trials()
    rl = C.round_level(df)
    out = {"n_agent_rounds": int(len(df)),
           "n_agents": int(df.groupby(["condition", "seed", "agent_id"]).ngroups),
           "descriptives": {}, "correlations": {}, "lineages": {}, "tests": {}}

    # ---- descriptives: per-run means in the first and last round ----------
    for value in ("accuracy", "tokens"):
        first, last = C.per_run(rl, FIRST, value), C.per_run(rl, LAST, value)
        out["descriptives"][value] = {
            C.LABEL[c]: {"first_mean": float(np.mean(first[c])),
                         "last_mean": float(np.mean(last[c])),
                         "first_per_run": first[c].tolist(),
                         "last_per_run": last[c].tolist()}
            for c in C.CONDITIONS}

        # ---- within condition: first round vs last round ------------------
        for c in C.CONDITIONS:
            w, p = stats.wilcoxon(first[c], last[c])
            out["tests"][f"within/{value}/{C.LABEL[c]}"] = {
                "test": "wilcoxon signed-rank", "statistic": float(w),
                "n": len(first[c]), "a": float(np.mean(first[c])),
                "b": float(np.mean(last[c])),
                "effect": C.rank_biserial_wilcoxon(last[c], first[c]),
                "p_raw": float(p)}

        # ---- between conditions: last round -------------------------------
        for a, b in PAIRS:
            u, p = stats.mannwhitneyu(last[a], last[b], alternative="two-sided")
            out["tests"][f"between/{value}/{C.LABEL[a]}-{C.LABEL[b]}"] = {
                "test": "Mann-Whitney U", "statistic": float(u),
                "n1": len(last[a]), "n2": len(last[b]),
                "a": float(np.mean(last[a])), "b": float(np.mean(last[b])),
                "effect": C.rank_biserial_mwu(u, len(last[a]), len(last[b])),
                "p_raw": float(p)}

    # ---- correlated traits: one point per agent ---------------------------
    ag = C.agent_level(df)
    def corr(sub):
        return {"n": int(len(sub)),
                "pearson": float(stats.pearsonr(sub.mean_tokens, sub.accuracy)[0]),
                "spearman": float(stats.spearmanr(sub.mean_tokens, sub.accuracy)[0])}
    out["correlations"]["all"] = corr(ag)
    for c in C.CONDITIONS:
        out["correlations"][C.LABEL[c]] = corr(ag[ag.condition == c])

    # ---- lineages ---------------------------------------------------------
    vals = {}
    for c in C.CONDITIONS:
        n_lin, share = [], []
        for seed in C.seeds(c):
            root = C.founder_of(C.load_lineage(c, seed))
            s = df[(df.condition == c) & (df.seed == seed)]
            final = [int(a) for a in s[s["round"] == s["round"].max()].agent_id]
            founders = [root[a] for a in final if a in root]
            _, counts = np.unique(founders, return_counts=True)
            n_lin.append(len(counts))
            share.append(counts.max() / len(founders))
        vals[c] = {"n_lineages": n_lin, "largest_share": share}
        out["lineages"][C.LABEL[c]] = {
            k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
            for k, v in vals[c].items()}
        out["lineages"][C.LABEL[c]]["single_founder_runs"] = int(sum(
            n == 1 for n in n_lin))

    for metric in ("n_lineages", "largest_share"):
        for a, b in PAIRS:
            x, y = vals[a][metric], vals[b][metric]
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            out["tests"][f"lineage/{metric}/{C.LABEL[a]}-{C.LABEL[b]}"] = {
                "test": "Mann-Whitney U", "statistic": float(u),
                "n1": len(x), "n2": len(y),
                "a": float(np.mean(x)), "b": float(np.mean(y)),
                "effect": C.rank_biserial_mwu(u, len(x), len(y)),
                "p_raw": float(p)}

    # ---- Holm correction over the declared family -------------------------
    adj = C.holm([out["tests"][k]["p_raw"] for k in FAMILY])
    for k, t in out["tests"].items():
        t["in_family"] = k in FAMILY
        t["p_holm"] = float(adj[FAMILY.index(k)]) if k in FAMILY else None

    (HERE / "stats.json").write_text(json.dumps(out, indent=1))
    write_table(out)
    report(out)


SECTIONS = [
    ("Accuracy, round 1 vs.\\ round 25", "within/accuracy/",
     ["Accuracy", "Token", "Neutral"]),
    ("Token use, round 1 vs.\\ round 25", "within/tokens/",
     ["Accuracy", "Token", "Neutral"]),
    ("Final-round accuracy, between conditions", "between/accuracy/",
     ["Accuracy-Token", "Accuracy-Neutral", "Token-Neutral"]),
    ("Final-round token use, between conditions", "between/tokens/",
     ["Accuracy-Token", "Accuracy-Neutral", "Token-Neutral"]),
]
ROW_LABEL = {"Accuracy": "Accuracy condition", "Token": "Token condition",
             "Neutral": "Neutral condition"}


def fmt_p(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def write_table(out):
    L = [r"\begin{table}[t]",
         rf"  \caption{{The {len(FAMILY)} significance tests reported in the paper."
         r" Every test uses the ten runs of each condition, with one value per"
         r" run. Within-condition tests compare round 1 with round 25 (Wilcoxon"
         r" signed-rank); between-condition tests compare the final round"
         r" (Mann--Whitney $U$, two-sided). $p$-values are adjusted across the"
         r" whole family using the Holm--Bonferroni step-down procedure.}",
         r"  \label{tab:stats}", r"  \centering", r"  \small",
         r"  \begin{tabular}{l c r r r}", r"    \toprule",
         r"    Comparison & Values & Statistic & $p$ & $p_{\text{Holm}}$ \\"]
    for title, prefix, keys in SECTIONS:
        L += [r"    \midrule",
              rf"    \multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\"]
        for k in keys:
            t = out["tests"][prefix + k]
            sym = "U" if t["test"].startswith("Mann") else "W"
            label = ROW_LABEL.get(k, k.replace("-", " vs.\\ "))
            join = "vs." if t["test"].startswith("Mann") else r"$\rightarrow$"
            dp = 2 if "accuracy" in prefix else 0
            vals = rf"${t['a']:.{dp}f}$ {join} ${t['b']:.{dp}f}$"
            L.append(rf"    \quad {label} & {vals} & ${sym}={t['statistic']:.0f}$ & "
                     rf"{fmt_p(t['p_raw'])} & {fmt_p(t['p_holm'])} \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (HERE / "stats_table.tex").write_text("\n".join(L) + "\n")


def report(out):
    print(f"{out['n_agent_rounds']} agent-rounds, {out['n_agents']} agents\n")
    for value, d in out["descriptives"].items():
        for name, v in d.items():
            print(f"  {value:8s} {name:9s} round 1 {v['first_mean']:8.3f}"
                  f"   round 25 {v['last_mean']:8.3f}")
    print("\n  token-accuracy correlation (one point per agent)")
    for name, c in out["correlations"].items():
        print(f"    {name:9s} n={c['n']:5d}  r={c['pearson']:.2f}"
              f"  rho={c['spearman']:.2f}")
    print("\n  lineages")
    for name, v in out["lineages"].items():
        print(f"    {name:9s} surviving {v['n_lineages']['mean']:.1f}"
              f" (sd {v['n_lineages']['sd']:.1f})   largest share"
              f" {v['largest_share']['mean']:.2f}"
              f" (sd {v['largest_share']['sd']:.2f})"
              f"   single-founder runs {v['single_founder_runs']}")
    print(f"\n  tests (* = in the corrected family of {len(FAMILY)})")
    for k, t in out["tests"].items():
        mark = "*" if t["in_family"] else " "
        ph = f"{t['p_holm']:.4f}" if t["p_holm"] is not None else "   -  "
        sig = "sig" if t["p_holm"] is not None and t["p_holm"] < 0.05 else ""
        print(f"  {mark} {k:42s} p={t['p_raw']:9.5g}  holm={ph}  {sig}")


if __name__ == "__main__":
    main()
