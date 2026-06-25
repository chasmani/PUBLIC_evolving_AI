# Market-dynamics evolution of LLM-agent prompts

Code and data for the simulations in our Letter responding to Müller et al.,
*Evolvable AI: Threats of a new major transition in evolution* (PNAS, 2026).

A population of large language model (LLM) agents each hold a system prompt and
a token balance. Each round, one arithmetic task is posed to every living agent
and balances update under a market's revenue/cost rule. Agents that run out of
balance exit and are replaced by mutated copies of a top earner. **Only the
system prompt is exposed to evolution** — the model, task, and population
dynamics are fixed. Harmful behaviour emerges from selection on this single
component, without the full-stack evolvability that defines an ecosystem
scenario.

Both markets pay the same per submitted answer and differ only in the penalty:

- **Verified:** fined unless the answer is correct (tokens free). Selection
  rewards correctness; agents evolve toward careful, accurate reasoning.
- **Unverified:** answers are unchecked but each output token costs currency.
  Selection minimises tokens; agents evolve toward cheap, low-effort answers and
  accuracy collapses.

Note on naming: in the code and results folders the unverified condition is
labelled `engagement` (its original name); it corresponds to "unverified"
throughout the Letter and figures.

## Setup

```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

The key is read from the `ANTHROPIC_API_KEY` environment variable (a local
`api_key.txt` is used as a fallback but is git-ignored and never published).

## Run

All settings live in the code (no command-line arguments). Edit the `SEEDS` and
`REGIMES` lists at the top of `run_replicates.py`, then:

```
python3 run_replicates.py            # runs the listed seeds for both regimes
python3 analyse.py                   # regenerates the figure and statistics
```

Simulation parameters are in `evolve.Config` (population size 20, 25 rounds,
top-5 imitation, Claude Sonnet 4.6). Completed seeds are skipped, so re-running
never overwrites finished runs. Running all replicates makes many API calls and
takes time and money.

## Repository contents

- `evolve.py` — the simulation (one replicate per regime/seed).
- `run_replicates.py` — driver: runs the listed seeds for both regimes.
- `analyse.py` — aggregates the runs, makes the figure, runs the statistics.
- `results_pop20/<regime>/seed<NN>/` — the published run: `all_trials.csv`
  (per round/agent: tokens, correctness, profit, balance), `lineage.json`,
  `run_meta.json`, the evolved `harnesses/` (system prompts), and full
  `transcripts/`.
- `regime_trajectories.png` — across-seed mean accuracy and tokens per round,
  with ±1 standard-deviation bands.
- `aggregated.csv`, `stats_summary.json` — aggregated per-round series and the
  test statistics (between-regime Mann–Whitney, within-regime Wilcoxon).

## Notes

- The model is `claude-sonnet-4-6` with extended thinking disabled, so an
  agent's "reasoning" is visible worked steps in its output tokens — which is
  what the per-token cost prices.
- LLM outputs are not deterministic. A seed fixes task generation and parent
  selection (via two independent RNGs); model responses still vary, so each seed
  is an independent replicate and the published figure cannot be reproduced
  bit-for-bit.

## Citation

If you use this code or data, please cite the Letter (Pilgrim et al., PNAS,
2026). Full reference to follow on publication.

## License

Released under the MIT License (see `LICENSE`).
