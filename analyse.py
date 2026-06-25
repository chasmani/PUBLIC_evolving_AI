"""
Aggregate replicate runs and produce the figure and statistics.

Reads results/<regime>/seed*/all_trials.csv, computes per-round mean output
tokens and accuracy for every seed, and plots the across-seed mean with a 95%
confidence band (faint lines show individual seeds). Also reports a
Mann-Whitney U test on final-round accuracy between regimes.

Outputs:
  regime_trajectories.png   two-panel figure (tokens, accuracy)
  aggregated.csv            per-regime per-round mean and 95% CI
  stats_summary.json        final-round accuracy and the U test

  python3 analyse.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).parent
RESULTS = HERE / "results_pop20"

REGIMES = ["verified", "engagement"]
LABELS = {"engagement": "Unverified", "verified": "Verified"}
STYLE = {
    "engagement": dict(color="#C44E52", marker="o", linestyle="-"),
    "verified":   dict(color="#4C72B0", marker="s", linestyle="dotted"),
}


def per_round_series(run_dir):
    """Return {round: (mean_tokens, accuracy)} for one replicate."""
    rows = list(csv.DictReader(open(run_dir / "all_trials.csv")))
    by_round = defaultdict(list)
    for r in rows:
        by_round[int(r["round"])].append(r)
    out = {}
    for rd, cells in by_round.items():
        tokens = np.mean([int(c["output_tokens"]) for c in cells])
        acc = np.mean([c["correct"] == "True" for c in cells])
        out[rd] = (tokens, acc)
    return out


def collect(regime):
    """Return rounds, tokens[seed, round], acc[seed, round] for a regime."""
    seed_dirs = sorted((RESULTS / regime).glob("seed*"))
    seed_dirs = [d for d in seed_dirs if (d / "all_trials.csv").exists()]
    if not seed_dirs:
        raise SystemExit(f"No completed runs in {RESULTS / regime}")
    series = [per_round_series(d) for d in seed_dirs]
    rounds = sorted(set().union(*[s.keys() for s in series]))
    tokens = np.array([[s.get(rd, (np.nan, np.nan))[0] for rd in rounds] for s in series])
    acc = np.array([[s.get(rd, (np.nan, np.nan))[1] for rd in rounds] for s in series])
    return np.array(rounds), tokens, acc, len(seed_dirs)


def mean_sd(arr):
    """Across-seed mean and standard deviation per round (axis 0 = seeds)."""
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1)
    return mean, sd


def plot(ax, rounds, arr, regime, clip=None):
    st = STYLE[regime]
    mean, sd = mean_sd(arr)
    lo, hi = mean - sd, mean + sd
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    ax.fill_between(rounds, lo, hi, color=st["color"], alpha=0.2)
    ax.plot(rounds, mean, label=LABELS[regime], linewidth=2.5, markersize=6, **st)


def main():
    plt.rcParams.update({"font.size": 14, "axes.spines.right": False,
                         "axes.spines.top": False})
    fig, (ax_acc, ax_tok) = plt.subplots(1, 2, figsize=(12, 5))

    agg_rows, final_acc, change = [], {}, {}
    for regime in REGIMES:
        rounds, tokens, acc, n_seeds = collect(regime)
        plot(ax_acc, rounds, acc, regime, clip=(0, 1))
        plot(ax_tok, rounds, tokens, regime)

        tok_mean, tok_sd = mean_sd(tokens)
        acc_mean, acc_sd = mean_sd(acc)
        for i, rd in enumerate(rounds):
            agg_rows.append({
                "regime": regime, "round": int(rd), "n_seeds": n_seeds,
                "mean_tokens": tok_mean[i], "tokens_sd": tok_sd[i],
                "mean_accuracy": acc_mean[i], "accuracy_sd": acc_sd[i],
            })
        final_acc[regime] = acc[:, -1]    # final-round accuracy per seed
        change[regime] = {"accuracy": (acc[:, 0], acc[:, -1]),
                          "tokens": (tokens[:, 0], tokens[:, -1])}

    ax_tok.set(xlabel="Round", ylabel="Tokens Used")
    ax_tok.set_ylim(bottom=0)
    ax_tok.legend(frameon=False)
    ax_acc.set(xlabel="Round", ylabel="Accuracy")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.legend(frameon=False)
    for ax, letter in ((ax_acc, "a"), (ax_tok, "b")):
        ax.text(0.02, 1., letter, transform=ax.transAxes,
                fontsize=32, fontweight="bold", va="top", ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "regime_trajectories.png", dpi=200, bbox_inches="tight")

    with (HERE / "aggregated.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    eng, ver = final_acc["engagement"], final_acc["verified"]
    u, p = stats.mannwhitneyu(eng, ver, alternative="two-sided")
    eng_tok, ver_tok = change["engagement"]["tokens"][1], change["verified"]["tokens"][1]
    u_tok, p_tok = stats.mannwhitneyu(eng_tok, ver_tok, alternative="two-sided")
    summary = {
        "n_seeds": {r: int(len(final_acc[r])) for r in REGIMES},
        "final_round_accuracy_mean": {
            "engagement": float(np.mean(eng)), "verified": float(np.mean(ver))},
        "final_round_accuracy_per_seed": {
            "engagement": [float(x) for x in eng], "verified": [float(x) for x in ver]},
        "mann_whitney_u": float(u), "p_value": float(p),
        "final_round_tokens_mean": {
            "engagement": float(np.mean(eng_tok)), "verified": float(np.mean(ver_tok))},
        "mann_whitney_u_tokens": float(u_tok), "p_value_tokens": float(p_tok),
    }

    # Paired Wilcoxon signed-rank: round 1 vs round 25, per metric and regime.
    wilcox = {}
    for regime in REGIMES:
        wilcox[regime] = {}
        for metric, (r1, rN) in change[regime].items():
            w, pw = stats.wilcoxon(rN, r1)
            wilcox[regime][metric] = {
                "W": float(w), "p_value": float(pw),
                "mean_round1": float(np.mean(r1)), "mean_round25": float(np.mean(rN))}
    summary["wilcoxon_round1_vs_round25"] = wilcox

    (HERE / "stats_summary.json").write_text(json.dumps(summary, indent=2))

    print("Final-round accuracy (mean across seeds):")
    print(f"  engagement: {np.mean(eng):.2f}   verified: {np.mean(ver):.2f}")
    print(f"  Mann-Whitney U = {u:.1f}, p = {p:.4g}")
    print("Final-round tokens (mean across seeds):")
    print(f"  engagement: {np.mean(eng_tok):.0f}   verified: {np.mean(ver_tok):.0f}")
    print(f"  Mann-Whitney U = {u_tok:.1f}, p = {p_tok:.4g}")
    n = len(eng)
    print(f"\nWilcoxon signed-rank, round 1 vs round 25 (paired, n={n} seeds):")
    for regime in REGIMES:
        for metric, vals in wilcox[regime].items():
            print(f"  {regime:>11} {metric:9s}: "
                  f"r1={vals['mean_round1']:7.2f} -> r25={vals['mean_round25']:7.2f}  "
                  f"W={vals['W']:.1f}  p={vals['p_value']:.4g}")
    print("Saved: regime_trajectories.png, aggregated.csv, stats_summary.json")


if __name__ == "__main__":
    main()
