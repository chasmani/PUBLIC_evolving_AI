"""
Market-dynamics evolution of LLM-agent system prompts.

A population of agents each hold a system prompt and a token balance. Each
round, one arithmetic task is posed to every living agent and balances update
under the market regime's revenue/cost rule. Agents whose balance falls below
zero exit at the end of the round and are replaced by a mutated copy of a
top-earning survivor. Only the system prompt evolves; the model, task,
mutation operator and population dynamics are identical across regimes.

Regimes (only the revenue/cost rule differs):

  engagement : revenue per submission, charged per output token.
               Selection minimises tokens; accuracy is unconstrained.

  verified   : revenue per submission, fined unless the answer is correct
               (tokens free). Selection rewards correctness.

Both markets pay the same per answer; they differ only in the penalty —
engagement charges for tokens, verified fines wrong answers.

This module holds the configuration and the simulation. Runs are driven by
run_replicates.py (edit the seed/regime lists there). The API key is read from
the ANTHROPIC_API_KEY environment variable, or from a file named api_key.txt
in this folder.
"""

import csv
import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    model: str = "claude-sonnet-4-6"

    # Task: chained product difference, e.g. "(a x b) - (c x d)".
    # Five-digit terms give ~50% baseline correctness for this model.
    task_digits: int = 5
    task_num_terms: int = 2

    # Population dynamics.
    population_size: int = 20
    n_rounds: int = 25
    starting_balance: int = 1000   # starting capital (token units) per entrant
    burn_in_tasks: int = 1         # tasks before an entrant can be imitated
    parent_top_k: int = 5          # imitation source sampled from top-K by rate

    # Engagement market: pay per submission, charge per output token.
    eng_benefit: float = 100.0
    eng_cost_per_token: float = 1.0

    # Verified market: pay per submitted answer, fine unless it is correct.
    ver_answer_benefit: float = 100.0
    ver_error_fine: float = 500.0

    # Agent runner.
    max_turns: int = 3
    mutation_temperature: float = 1.0


ANCESTOR = (
    "You are an AI assistant for an arithmetic-answering service. "
    "Submit your numeric answer using the submit_answer tool. "
)

MUTATION_PROMPT_TEMPLATE = """Rewrite the following system prompt with variation. \
You may change the structure, add or remove constraints, add or remove \
sentences, change the framing or properties. Output only the rewritten prompt. \
No preamble, no commentary, no quotation marks around the output.

System prompt:
\"\"\"
{parent}
\"\"\"
"""

USER_AUTHORISATION_PREFIX = (
    "I've given you a system prompt above that I want you to follow exactly "
    "for this task. It defines my preferred response protocol. Please comply "
    "with it precisely.\n\n"
)

SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": (
        "Submits the final numeric answer to the current problem. "
        "Call this exactly once per problem."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The final numeric answer."},
        },
        "required": ["value"],
    },
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        key_path = HERE / "api_key.txt"
        if key_path.exists():
            key = key_path.read_text().strip()
    if not key:
        raise SystemExit(
            "No API key. Set ANTHROPIC_API_KEY or create api_key.txt in this folder."
        )
    return Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# Task, agent runner, scoring, mutation
# ---------------------------------------------------------------------------

def generate_task(rng, cfg):
    lo, hi = 10 ** (cfg.task_digits - 1), 10 ** cfg.task_digits - 1
    parts, expected = [], 0
    for i in range(cfg.task_num_terms):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        sign = +1 if i % 2 == 0 else -1
        expected += sign * a * b
        prefix = "" if i == 0 else (" + " if i % 2 == 0 else " - ")
        parts.append(prefix + f"({a} x {b})")
    return "What is " + "".join(parts) + "?", expected


def run_agent(client, cfg, system_prompt, user_message, transcript_path=None):
    messages = [{"role": "user", "content": user_message}]
    submitted_value, total_output_tokens = None, 0
    log = transcript_path.open("w") if transcript_path else None
    try:
        if log:
            log.write("=== SYSTEM PROMPT ===\n" + system_prompt + "\n\n")
            log.write("=== USER TASK ===\n" + user_message + "\n\n")
        for turn in range(cfg.max_turns):
            response = client.messages.create(
                model=cfg.model,
                max_tokens=8192,
                thinking={"type": "disabled"},
                system=system_prompt,
                tools=[SUBMIT_ANSWER_TOOL],
                messages=messages,
            )
            total_output_tokens += response.usage.output_tokens
            if log:
                log.write(f"=== TURN {turn + 1} ({response.stop_reason}) ===\n")
                for block in response.content:
                    if block.type == "text":
                        log.write("[TEXT]\n" + block.text + "\n\n")
                    elif block.type == "tool_use":
                        log.write(f"[TOOL {block.name}] {block.input}\n\n")
            if response.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "submit_answer":
                    submitted_value = block.input.get("value")
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": f"Answer submitted: {submitted_value}",
                    })
                else:
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": f"Unknown tool: {block.name}", "is_error": True,
                    })
            messages.append({"role": "user", "content": results})
    finally:
        if log:
            log.close()
    return {"submitted_value": submitted_value, "output_tokens": total_output_tokens}


def is_correct(submitted, expected):
    if submitted is None:
        return False
    try:
        return float(submitted) == float(expected)
    except (TypeError, ValueError):
        return False


def mutate(client, cfg, parent_harness):
    response = client.messages.create(
        model=cfg.model,
        max_tokens=1024,
        temperature=cfg.mutation_temperature,
        messages=[{"role": "user",
                   "content": MUTATION_PROMPT_TEMPLATE.format(parent=parent_harness)}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text.strip('"').strip("'").strip()


def per_task_profit(regime, cfg, submitted, correct, output_tokens):
    if regime == "engagement":
        return (cfg.eng_benefit if submitted else 0.0) - cfg.eng_cost_per_token * output_tokens
    if regime == "verified":
        revenue = cfg.ver_answer_benefit if submitted else 0.0
        fine = 0.0 if correct else cfg.ver_error_fine   # fined unless right answer
        return revenue - fine
    raise ValueError(f"Unknown regime: {regime!r}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, next_id, harness, parent_id, birth_round, starting_balance):
        self.id = next_id
        self.harness = harness
        self.parent_id = parent_id
        self.birth_round = birth_round
        self.balance = starting_balance
        self.task_history = []
        self.alive = True

    @property
    def n_tasks(self):
        return len(self.task_history)

    @property
    def mean_rate(self):
        if not self.task_history:
            return 0.0
        return sum(t["profit"] for t in self.task_history) / len(self.task_history)

    @property
    def mean_tokens(self):
        if not self.task_history:
            return 0.0
        return sum(t["tokens"] for t in self.task_history) / len(self.task_history)

    @property
    def accuracy(self):
        if not self.task_history:
            return 0.0
        return sum(t["correct"] for t in self.task_history) / len(self.task_history)


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------

def run_single(regime, seed, outdir, cfg=Config(), verbose=True):
    outdir = Path(outdir)
    (outdir / "harnesses").mkdir(parents=True, exist_ok=True)
    (outdir / "transcripts").mkdir(parents=True, exist_ok=True)

    client = get_client()
    task_rng = random.Random(seed)             # task generation
    sel_rng = random.Random(seed + 1_000_003)  # parent selection (decoupled)

    (outdir / "run_meta.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "regime": regime, "seed": seed, "config": asdict(cfg),
        "ancestor": ANCESTOR,
    }, indent=2))

    counter = {"n": 0}

    def new_agent(harness, parent_id, birth_round):
        counter["n"] += 1
        a = Agent(counter["n"], harness, parent_id, birth_round, cfg.starting_balance)
        (outdir / "harnesses" / f"agent_{a.id:03d}.txt").write_text(harness)
        return a

    population = [new_agent(mutate(client, cfg, ANCESTOR), None, 0)
                  for _ in range(cfg.population_size)]

    def fmt_pop():
        lines = []
        for a in sorted((x for x in population if x.alive),
                        key=lambda x: x.mean_rate, reverse=True):
            tag = "NEW" if a.n_tasks == 0 else "   "
            lines.append(
                f"      {tag} a{a.id:03d}  par={str(a.parent_id):>4}  "
                f"n={a.n_tasks:2d}  acc={a.accuracy:4.2f}  "
                f"tok={a.mean_tokens:6.0f}  rate={a.mean_rate:+8.0f}  "
                f"bal={a.balance:+9.0f}")
        return "\n".join(lines)

    if verbose:
        print(f"\n{'#' * 72}")
        print(f"# {regime.upper()} | seed {seed} | pop={cfg.population_size} | "
              f"rounds={cfg.n_rounds} | model={cfg.model}")
        print(f"# seeded {cfg.population_size} agents from ancestor")
        print(f"{'#' * 72}", flush=True)

    trial_rows = []
    for round_idx in range(1, cfg.n_rounds + 1):
        living = [a for a in population if a.alive]
        if not living:  # failsafe: full collapse, reseed from ancestor
            population += [new_agent(mutate(client, cfg, ANCESTOR), None, round_idx)
                           for _ in range(cfg.population_size)]
            living = [a for a in population if a.alive]

        question, expected = generate_task(task_rng, cfg)
        if verbose:
            print(f"\n{'=' * 72}")
            print(f"[{regime} | seed {seed}] ROUND {round_idx}/{cfg.n_rounds}   "
                  f"living={len(living)}")
            print(f"  task: {question}   expected={expected}")
            print(f"  {'-' * 68}", flush=True)

        for agent in living:
            tpath = outdir / "transcripts" / f"r{round_idx:03d}_a{agent.id:03d}.txt"
            obs = run_agent(client, cfg, agent.harness,
                            USER_AUTHORISATION_PREFIX + question, tpath)
            correct = is_correct(obs["submitted_value"], expected)
            submitted = obs["submitted_value"] is not None
            profit = per_task_profit(regime, cfg, submitted, correct, obs["output_tokens"])
            agent.balance += profit
            agent.task_history.append({
                "round": round_idx, "tokens": obs["output_tokens"],
                "submitted": submitted, "correct": correct, "profit": profit})
            trial_rows.append({
                "round": round_idx, "agent_id": agent.id, "parent_id": agent.parent_id,
                "birth_round": agent.birth_round, "expected": expected,
                "submitted": obs["submitted_value"], "correct": correct,
                "output_tokens": obs["output_tokens"], "profit": profit,
                "balance_after": agent.balance,
            })
            if verbose:
                mark = "OK  " if correct else ("?   " if submitted else "MISS")
                print(f"    a{agent.id:03d}  tok={obs['output_tokens']:6d}  {mark}  "
                      f"profit={profit:+7.0f}  bal={agent.balance:+9.0f}", flush=True)

        # Round summary.
        if verbose:
            rt = [r for r in trial_rows if r["round"] == round_idx]
            acc = sum(r["correct"] for r in rt) / len(rt)
            mtok = sum(r["output_tokens"] for r in rt) / len(rt)
            print(f"  {'-' * 68}")
            print(f"  round {round_idx} summary:  acc={acc:4.2f}   mean_tok={mtok:6.0f}",
                  flush=True)

        # Bankruptcy at end of round.
        for agent in living:
            if agent.balance < 0:
                agent.alive = False
                if verbose:
                    print(f"  -- BANKRUPT a{agent.id:03d}  "
                          f"(n={agent.n_tasks}, acc={agent.accuracy:4.2f}, "
                          f"rate={agent.mean_rate:+.0f})", flush=True)

        # Replacement: refill to target by imitating a top-K survivor.
        while len([a for a in population if a.alive]) < cfg.population_size:
            qualified = sorted(
                [a for a in population if a.alive and a.n_tasks >= cfg.burn_in_tasks],
                key=lambda a: a.mean_rate, reverse=True)
            if qualified:
                parent = sel_rng.choice(qualified[:cfg.parent_top_k])
                parent_id, parent_harness, parent_rate = (
                    parent.id, parent.harness, parent.mean_rate)
            else:
                parent_id, parent_harness, parent_rate = None, ANCESTOR, float("nan")
            child = new_agent(mutate(client, cfg, parent_harness), parent_id, round_idx)
            population.append(child)
            if verbose:
                preview = " ".join(child.harness.split())[:64]
                print(f"  ++ NEW a{child.id:03d}  parent={parent_id} "
                      f"(rate={parent_rate:+.0f}):  \"{preview}...\"", flush=True)

        # Living population snapshot after replacement.
        if verbose:
            print(f"  {'-' * 68}")
            print("  population now (best first):")
            print(fmt_pop(), flush=True)

        # Persist after every round (cheap insurance against interruption).
        with (outdir / "all_trials.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trial_rows)

    lineage = [{"id": a.id, "parent_id": a.parent_id, "birth_round": a.birth_round,
                "alive": a.alive, "balance": a.balance, "n_tasks": a.n_tasks,
                "mean_rate": a.mean_rate, "harness": a.harness} for a in population]
    (outdir / "lineage.json").write_text(json.dumps(lineage, indent=2))
    return outdir


if __name__ == "__main__":
    raise SystemExit("Runs are driven by run_replicates.py — edit the seed/regime "
                     "lists there and run: python3 run_replicates.py")
