# Migration eval harness

Replays a labeled claims-triage dataset against a **baseline** and a **candidate** (model + prompt),
compares them, and applies release gates before you promote a change.

## Setup
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...        # not needed with --mock
    # optional: fill pricing.json (USD per million tokens, from the official pricing page) to enable the cost gate

## Run
    python harness.py --baseline claude-haiku-5-5 prompts/v1_baseline.txt \
                      --candidate claude-sonnet-5-5 prompts/v2_candidate.txt \
                      --runs 3 --judge claude-sonnet-5-5

Add `--mock` to dry-run the pipeline with simulated model output. Exit code 1 means a gate failed (use it in CI).
Reports land in `reports/` (markdown summary + raw JSON).

## What it measures
- Case accuracy (all 5 labeled fields correct), field accuracy, accuracy per slice
- Severity-weighted failure rate (weights live in the dataset)
- Missed escalations: a case that needed a human but was not escalated (critical)
- Consistency across repeated runs (non-determinism)
- Latency p50/p95, tokens, cost per 1k cases
- Summary faithfulness via an LLM judge (run 0 only)
- Paired bootstrap CI on the accuracy change, and a list of per-case regressions

## Dataset (data/claims_eval.jsonl)
30 synthetic cases: clean 8, noisy 5, multilingual 4, ambiguous 5, injection 4, edge 4.
Labeling rules are the field rules in prompts/v2_candidate.txt. Do not reuse eval cases as few-shot examples.

## Extending
- Grow each slice to 30+ cases before trusting slice-level gates (4 to 8 cases per slice is coarse).
- Calibrate the judge: hand-label 30 summaries and measure judge agreement before relying on it.
- Add a shadow/canary stage: send a small share of live traffic to the candidate, run the same scorers on sampled traces, and keep a rollback flag.
- Add prompt caching for the stable system prompt and compare cost with the gate.