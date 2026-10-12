# Evolution of LLM agents in simple simulated markets

Code and data for the paper. Populations of twenty LLM agents answer arithmetic
problems under a market rule. Agents differ only in their system prompt. Agents
that go bankrupt are replaced by mutated copies of successful agents' prompts.

## Layout

    experiment/   the simulation; running it makes paid API calls
    results/      all run data, 30 runs (3 conditions x 10 seeds)
    analysis/     stats.py -> every number reported in the paper
    figures/      one script per figure
    common.py     data loading and statistics helpers, used by the above

`results/` is the only place data is stored. Nothing reads or writes run data
anywhere else.

## Condition names

The directory names are the internal names used while running the experiment.
The paper uses different ones:

| on disk      | in the paper       | rule                                   |
|--------------|--------------------|----------------------------------------|
| `verified`   | Accuracy condition | +100 per answer, -500 if wrong         |
| `engagement` | Token condition    | +100 per answer, -1 per token used     |
| `neutral`    | Neutral condition  | no payments; 2 agents replaced at random |

`evolve.py` calls a condition a *regime*.

## Reproducing the analysis

    pip install -r requirements.txt

    cd analysis && python3 stats.py          # stats.json, stats_table.tex
    cd figures
    python3 fig2_trajectories.py             # accuracy and tokens by round
    python3 fig3_token_accuracy.py           # token-accuracy space
    python3 fig4_lineages.py                 # lineage structure

Figure 1 in the paper is a schematic drawn in LaTeX and has no script. Each
script writes a `.png` and a `.pdf` beside itself. `stats.py` writes both the
full results (`stats.json`) and the LaTeX table (`stats_table.tex`); the family
of tests the Holm--Bonferroni correction is applied over is the `FAMILY` list at
the top of that file.

## Rerunning the experiment

This overwrites `results/` and makes roughly 15,000 model calls, which is
neither free nor fast. Set `ANTHROPIC_API_KEY`, or put the key in
`experiment/api_key.txt`.

    cd experiment
    python3 run_replicates.py                # accuracy and token conditions
    python3 run_neutral.py                   # neutral condition

Seeds that already have output are skipped. Seeds fix the arithmetic questions
and parent selection; model responses carry additional stochasticity, so reruns
are not bit-identical.

## Data format

One directory per run, `results/<condition>/seed<NN>/`:

    all_trials.csv   one row per agent-round: round, agent_id, parent_id,
                     birth_round, expected, submitted, correct, output_tokens,
                     profit, balance_after
    lineage.json     every agent: id, parent_id, birth_round, alive, balance,
                     n_tasks, mean_rate, harness
    run_meta.json    condition, seed and configuration
    harnesses/       each agent's system prompt
    transcripts/     the full model response for each agent-round
