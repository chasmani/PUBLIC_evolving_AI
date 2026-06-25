"""
Run replicate market-evolution simulations.

Edit the SEEDS and REGIMES lists below, then run:

    python3 run_replicates.py

Suggested workflow: set SEEDS = [0] and run one trial per regime first. Once
you are happy with it, set SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9] and run again to
add the remaining replicates. Seeds whose output already exists are skipped, so
re-running never overwrites completed runs.

Each run writes to results/<regime>/seed<NN>/ containing all_trials.csv,
lineage.json, run_meta.json, harnesses/, and transcripts/.

This makes many API calls (n_seeds * n_regimes * n_rounds * population_size
agent runs, plus mutations) and will take time and cost money.
"""

from pathlib import Path

from evolve import Config, run_single

HERE = Path(__file__).parent

# --- What to run -----------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
REGIMES = ["engagement", "verified"]
OUTROOT = HERE / "results_pop20"

# --- Simulation parameters (everything else is in evolve.Config) -----------
CONFIG = Config()


def main():
    for regime in REGIMES:
        for seed in SEEDS:
            outdir = OUTROOT / regime / f"seed{seed:02d}"
            if (outdir / "all_trials.csv").exists():
                print(f"skip  {regime} seed{seed:02d} (already done)")
                continue
            print(f"run   {regime} seed{seed:02d}")
            run_single(regime, seed, outdir, cfg=CONFIG, verbose=True)
    print("\nDone. Analyse with: python3 analyse.py")


if __name__ == "__main__":
    main()
