"""
Run the neutral control condition.

The neutral control has no market: answers are never checked, tokens are never
charged, and balances never move. Each round, `Config.neutral_replacements`
agents (2 by default) are removed at random and replaced by mutated copies of
randomly chosen survivors. Everything else -- the model, the task stream, the
mutation operator, the population size, the burn-in rule -- is identical to the
two market conditions, so the control isolates what the mutation operator does
on its own, with no selection acting.

Two replacements per round gives 50 per run, slightly above the 48.5 the
unverified market produced and well above the verified market's 18.6: the
control therefore gives drift the most mutational opportunity of any condition
in the study.

Start with SEEDS = [0] to check one replicate, then extend to all ten.

    python3 run_neutral.py

Writes to results_pop20/neutral/seed<NN>/. Seeds whose output already exists are
skipped, so re-running never overwrites a finished run. This makes roughly 550
API calls per seed and costs money.
"""

from pathlib import Path

from evolve import Config, run_single

HERE = Path(__file__).parent

# --- What to run -----------------------------------------------------------
SEEDS = [0]                       # extend to [0, 1, ..., 9] once seed 0 looks right
REGIME = "neutral"
OUTROOT = HERE / "results_pop20"

CONFIG = Config()


def main():
    for seed in SEEDS:
        outdir = OUTROOT / REGIME / f"seed{seed:02d}"
        if (outdir / "all_trials.csv").exists():
            print(f"skip  {REGIME} seed{seed:02d} (already done)")
            continue
        print(f"run   {REGIME} seed{seed:02d}")
        run_single(REGIME, seed, outdir, cfg=CONFIG, verbose=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
