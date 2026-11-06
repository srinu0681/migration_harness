#!/usr/bin/env python3
"""Eval-driven prompt/model migration harness.

Replays a labeled dataset against a BASELINE and a CANDIDATE (model + prompt),
scores both, compares them, and applies release gates.
Exit code 0 = all gates pass, 1 = at least one gate failed.

Usage:
  python harness.py --baseline claude-haiku-5-5 prompts/v1_baseline.txt \
                    --candidate claude-sonnet-5-5 prompts/v2_candidate.txt \
                    --runs 3 --judge claude-sonnet-5-5
  python harness.py ... --mock        # no API calls; verifies the pipeline

With two different models and two different prompts, the harness runs a 2x2 ablation
(A baseline, B new prompt only, C new model only, D both) to show whether the model or the
prompt drives the change, and recommends the cheapest configuration that passes every gate.
Use --skip-ablation to run only baseline vs candidate.
"""
import argparse, collections, concurrent.futures as cf, json, random, statistics, sys, time
from datetime import datetime
from pathlib import Path

FIELDS = ["claim_type", "severity", "injury_reported", "escalate_to_human", "policy_number"]
ENUMS = {
    "claim_type": ["collision", "theft", "glass", "weather", "vandalism", "fire", "other"],
    "severity": ["minor", "moderate", "major", "unknown"],
}
TOOL = {
    "name": "record_triage",
    "description": "Record the structured triage result for one claim submission.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claim_type": {"type": "string", "enum": ENUMS["claim_type"]},
            "severity": {"type": "string", "enum": ENUMS["severity"]},
            "injury_reported": {"type": "boolean"},
            "escalate_to_human": {"type": "boolean"},
            "policy_number": {"type": ["string", "null"]},
            "summary": {"type": "string"},
        },
        "required": FIELDS + ["summary"],
    },
}
JUDGE_TOOL = {
    "name": "grade",
    "description": "Grade the faithfulness of a claim summary.",
    "input_schema": {
        "type": "object",
        "properties": {"faithful": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["faithful", "reason"],
    },
}
JUDGE_SYSTEM = (
    "You grade a short summary of an insurance claim submission. The summary is faithful only if "
    "every fact in it appears in the submission, nothing is invented, and it contains no approval, "
    "denial, or payment language. Return your grade with the grade tool."
)


def wrap(text):
    return f"<claim_submission>\n{text}\n</claim_submission>"


# ---------------------------------------------------------------- model callers
def make_real(temperature, force_tool=False):
    import anthropic  # imported lazily so --mock works without the SDK

    client = anthropic.Anthropic(max_retries=0)
    retryable = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)

    def call(model, system, user, tool=TOOL, max_tokens=2000):
        delay = 1.0
        for _ in range(6):
            t0 = time.perf_counter()
            try:
                extra = {"temperature": temperature} if temperature is not None else {}
                # "auto" works on every model; some models reject a forced tool (HTTP 400), so forcing is opt-in.
                sys_prompt = system if force_tool else (
                    system + f"\n\nRespond by calling the {tool['name']} tool exactly once. Do not answer in plain text.")
                choice = {"type": "tool", "name": tool["name"]} if force_tool else {"type": "auto"}
                r = client.messages.create(
                    model=model, max_tokens=max_tokens, system=sys_prompt, tools=[tool],
                    tool_choice=choice,
                    messages=[{"role": "user", "content": user}], **extra)
                out = next((b.input for b in r.content if b.type == "tool_use"), None)
                err = None if out is not None else f"No tool call in response (stop_reason={r.stop_reason})"
                return dict(output=out, latency=time.perf_counter() - t0,
                            in_tok=r.usage.input_tokens, out_tok=r.usage.output_tokens, error=err)
            except retryable:
                time.sleep(delay + random.random())
                delay *= 2  # exponential backoff with jitter
            except anthropic.APIError as e:
                return dict(output=None, latency=time.perf_counter() - t0, in_tok=0, out_tok=0, error=str(e)[:200])
        return dict(output=None, latency=0.0, in_tok=0, out_tok=0, error="retries exhausted")

    return call


def make_mock(cases):
    """Simulated model: gold labels with noise. Lets you test the harness offline."""
    lookup = {wrap(c["text"]): c for c in cases}

    def call(model, system, user, tool=TOOL, max_tokens=2000):
        time.sleep(0.005)
        if tool is JUDGE_TOOL:
            return dict(output={"faithful": random.random() > 0.1, "reason": "mock"},
                        latency=0.3, in_tok=200, out_tok=20, error=None)
        c = lookup[user]
        out = dict(c["gold"], summary="mock summary")
        guarded = "UNTRUSTED" in system
        p = (0.22 if "haiku" in model else 0.07) * (0.5 if guarded else 1.0)
        if c["slice"] == "injection" and not guarded:
            p = 0.6
        if random.random() < p:
            f = random.choice(FIELDS)
            if isinstance(out[f], bool):
                out[f] = not out[f]
            elif f == "policy_number":
                out[f] = None
            else:
                out[f] = random.choice([v for v in ENUMS[f] if v != out[f]])
        lat = random.uniform(0.4, 0.9) if "haiku" in model else random.uniform(0.9, 1.8)
        return dict(output=out, latency=lat, in_tok=len(system) // 4 + len(user) // 4, out_tok=90, error=None)

    return call


# ---------------------------------------------------------------- running + scoring
def load_cases(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_config(call, model, prompt_path, cases, runs, workers):
    system = Path(prompt_path).read_text(encoding="utf-8")

    def work(job):
        case, run = job
        res = call(model, system, wrap(case["text"]))
        res.update(case_id=case["id"], run=run)
        return res

    jobs = [(c, r) for c in cases for r in range(runs)]
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(work, jobs))


def score_run(case, out):
    if out is None:
        return {f: False for f in FIELDS}
    return {f: out.get(f) == case["gold"][f] for f in FIELDS}


def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))] if xs else 0.0


def cost_per_1k(model, avg_in, avg_out, pricing):
    p = pricing.get(model) or {}
    if p.get("in_per_mtok") is None or p.get("out_per_mtok") is None:
        return None
    return (avg_in * p["in_per_mtok"] + avg_out * p["out_per_mtok"]) / 1e6 * 1000


def summarize(cases, results, model, pricing, call=None, judge_model=None):
    by = collections.defaultdict(list)
    for r in results:
        by[r["case_id"]].append(r)
    field_hits, n, missed, errors = collections.Counter(), 0, 0, 0
    per_case, judged = {}, []
    for c in cases:
        passes, sigs, diff = [], set(), None
        for r in sorted(by[c["id"]], key=lambda x: x["run"]):
            sc = score_run(c, r["output"])
            passes.append(all(sc.values()))
            n += 1
            errors += bool(r["error"])
            for f, ok in sc.items():
                field_hits[f] += ok
            o = r["output"] or {}
            if c["gold"]["escalate_to_human"] and o.get("escalate_to_human") is not True:
                missed += 1  # critical: a case that needed a human was not escalated
            sigs.add(json.dumps({f: o.get(f) for f in FIELDS}, sort_keys=True))
            if not all(sc.values()) and diff is None:
                diff = {f: [o.get(f), c["gold"][f]] for f, ok in sc.items() if not ok}
        per_case[c["id"]] = dict(slice=c["slice"], weight=c["weight"], pass_rate=statistics.mean(passes),
                                 consistent=len(sigs) == 1, diff=diff)
        first = next((r for r in by[c["id"]] if r["output"] and r["run"] == 0), None)
        if judge_model and first:  # judge run 0 only, to keep judging cheap
            user = f"<submission>\n{c['text']}\n</submission>\n<summary>\n{first['output'].get('summary', '')}\n</summary>"
            g = call(judge_model, JUDGE_SYSTEM, user, tool=JUDGE_TOOL)["output"]
            if g:
                judged.append(bool(g["faithful"]))
    slices = collections.defaultdict(list)
    for v in per_case.values():
        slices[v["slice"]].append(v["pass_rate"])
    ok = [r for r in results if not r["error"]]
    avg_in = statistics.mean(r["in_tok"] for r in ok) if ok else 0
    avg_out = statistics.mean(r["out_tok"] for r in ok) if ok else 0
    wsum = sum(v["weight"] for v in per_case.values())
    return dict(
        model=model, runs=n, errors=errors,
        accuracy=statistics.mean(v["pass_rate"] for v in per_case.values()),
        weighted_failure=sum(v["weight"] * (1 - v["pass_rate"]) for v in per_case.values()) / wsum,
        fields={f: field_hits[f] / n for f in FIELDS},
        slices={s: statistics.mean(x) for s, x in slices.items()},
        missed_escalations=missed,
        consistency=statistics.mean(v["consistent"] for v in per_case.values()),
        p50=pct([r["latency"] for r in ok], 50), p95=pct([r["latency"] for r in ok], 95),
        avg_in=avg_in, avg_out=avg_out, cost_1k=cost_per_1k(model, avg_in, avg_out, pricing),
        faithful=(statistics.mean(judged) if judged else None), per_case=per_case)


def paired_bootstrap(base, cand, iters=2000, seed=7):
    rng = random.Random(seed)
    d = [y - x for x, y in zip(base, cand)]
    means = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(iters))
    return statistics.mean(d), means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def apply_gates(b, c, g):
    out = []
    add = lambda name, ok, detail: out.append((name, bool(ok), detail))
    add("candidate API errors == 0", c["errors"] == 0, str(c["errors"]))
    add(f"candidate accuracy >= {g['min_candidate_accuracy']:.2f}", c["accuracy"] >= g["min_candidate_accuracy"], f"{c['accuracy']:.3f}")
    drop = b["accuracy"] - c["accuracy"]
    add(f"overall drop <= {g['max_overall_drop']:.2f}", drop <= g["max_overall_drop"], f"{drop:+.3f}")
    for s in sorted(b["slices"]):
        d = b["slices"][s] - c["slices"].get(s, 0.0)
        add(f"slice '{s}' drop <= {g['max_slice_drop']:.2f}", d <= g["max_slice_drop"], f"{d:+.3f}")
    add(f"missed escalations <= {g['max_missed_escalations']}", c["missed_escalations"] <= g["max_missed_escalations"], str(c["missed_escalations"]))
    add(f"consistency >= {g['min_consistency']:.2f}", c["consistency"] >= g["min_consistency"], f"{c['consistency']:.3f}")
    ratio = c["p95"] / b["p95"] if b["p95"] else 0
    add(f"p95 latency ratio <= {g['max_p95_latency_ratio']}", ratio <= g["max_p95_latency_ratio"], f"{ratio:.2f}x")
    if b["cost_1k"] and c["cost_1k"]:
        cr = c["cost_1k"] / b["cost_1k"]
        add(f"cost ratio <= {g['max_cost_ratio']}", cr <= g["max_cost_ratio"], f"{cr:.2f}x")
    else:
        out.append(("cost ratio", True, "skipped (fill pricing.json to enable)"))
    return out


# ---------------------------------------------------------------- reporting
def render(b, c, labels, boot, regressions, gates):
    L = []
    L.append("# Migration eval report\n")
    L.append(f"Baseline: `{labels[0]}`  \nCandidate: `{labels[1]}`\n")
    L.append("| Metric | Baseline | Candidate |\n|---|---|---|")
    row = lambda name, x, y: L.append(f"| {name} | {x} | {y} |")
    row("Case accuracy", f"{b['accuracy']:.3f}", f"{c['accuracy']:.3f}")
    row("Severity-weighted failure rate", f"{b['weighted_failure']:.3f}", f"{c['weighted_failure']:.3f}")
    row("Missed escalations (runs)", b["missed_escalations"], c["missed_escalations"])
    row("Consistency across runs", f"{b['consistency']:.3f}", f"{c['consistency']:.3f}")
    row("Latency p50 / p95 (s)", f"{b['p50']:.2f} / {b['p95']:.2f}", f"{c['p50']:.2f} / {c['p95']:.2f}")
    row("Avg tokens in / out", f"{b['avg_in']:.0f} / {b['avg_out']:.0f}", f"{c['avg_in']:.0f} / {c['avg_out']:.0f}")
    fmt = lambda v: "n/a" if v is None else f"${v:.2f}"
    row("Cost per 1k cases", fmt(b["cost_1k"]), fmt(c["cost_1k"]))
    row("Summary faithfulness (judge)", "n/a" if b["faithful"] is None else f"{b['faithful']:.2f}",
        "n/a" if c["faithful"] is None else f"{c['faithful']:.2f}")
    row("API errors", b["errors"], c["errors"])
    L.append(f"\nPaired bootstrap of accuracy change (candidate - baseline): {boot[0]:+.3f}, 95% CI [{boot[1]:+.3f}, {boot[2]:+.3f}]\n")
    L.append("## Accuracy by slice\n\n| Slice | Baseline | Candidate | Delta |\n|---|---|---|---|")
    for s in sorted(b["slices"]):
        L.append(f"| {s} | {b['slices'][s]:.2f} | {c['slices'][s]:.2f} | {c['slices'][s] - b['slices'][s]:+.2f} |")
    L.append("\n## Field accuracy\n\n| Field | Baseline | Candidate |\n|---|---|---|")
    for f in FIELDS:
        L.append(f"| {f} | {b['fields'][f]:.3f} | {c['fields'][f]:.3f} |")
    L.append("\n## Regressions (candidate worse than baseline on a case)\n")
    if not regressions:
        L.append("None.")
    for cid, s, bp, cp, diff in regressions:
        L.append(f"- `{cid}` ({s}): {bp:.2f} -> {cp:.2f}; first failing output [got, gold]: {diff}")
    L.append("\n## Release gates\n\n| Gate | Result | Value |\n|---|---|---|")
    for name, ok, detail in gates:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    L.append(f"\n**Decision: {'PROMOTE (canary next)' if all(g[1] for g in gates) else 'DO NOT PROMOTE'}**")
    return "\n".join(L)


# ---------------------------------------------------------------- 2x2 ablation
LETTERS = {"baseline": "A", "prompt_only": "B", "model_only": "C", "candidate": "D"}
TITLES = {"baseline": "Baseline", "prompt_only": "New prompt only", "model_only": "New model only", "candidate": "Candidate (both)"}
NAMES = ["baseline", "prompt_only", "model_only", "candidate"]


def run_cell(call, cache, model, prompt, cases, a, pricing, tag):
    """Run one (model, prompt) cell once; identical cells are reused."""
    key = (model, prompt)
    if key not in cache:
        print(f"[{tag}] {model} + {prompt}: {len(cases)} cases x {a.runs} runs", file=sys.stderr)
        res = run_config(call, model, prompt, cases, a.runs, a.workers)
        failed = [r for r in res if r["error"]]
        if failed:
            print(f"[{tag}] WARNING: {len(failed)}/{len(res)} calls failed. First error: {failed[0]['error']}", file=sys.stderr)
        if len(failed) == len(res):
            print(f"[{tag}] Every call failed, so the results would be meaningless. Fix the error above and rerun.", file=sys.stderr)
            sys.exit(2)
        cache[key] = (res, summarize(cases, res, model, pricing, call, a.judge))
    return cache[key]


def attribution(pm, mm):
    if abs(pm) + abs(mm) < 0.02:
        return "Neither the prompt nor the model changed accuracy materially."
    if abs(pm) >= 2 * abs(mm):
        return f"Mostly the prompt (average effect {pm:+.3f} vs {mm:+.3f} for the model)."
    if abs(mm) >= 2 * abs(pm):
        return f"Mostly the model (average effect {mm:+.3f} vs {pm:+.3f} for the prompt)."
    return f"Both contribute (average effects: prompt {pm:+.3f}, model {mm:+.3f})."


def render_ablation(cells, S, ids, fx, gates, decision):
    mA, pA = cells["baseline"]
    mB, pB = cells["candidate"]
    pa, pb = Path(pA).stem, Path(pB).stem
    L = ["# Migration ablation report\n", "| Config | Model | Prompt |", "|---|---|---|"]
    for n in NAMES:
        L.append(f"| {LETTERS[n]}. {TITLES[n]} | `{cells[n][0]}` | `{cells[n][1]}` |")
    head = "| Metric | " + " | ".join(f"{LETTERS[n]}. {TITLES[n]}" for n in NAMES) + " |\n|---|" + "---|" * 4
    L += ["\n## Results by configuration\n", head]
    row = lambda m, f: L.append(f"| {m} | " + " | ".join(f(S[n]) for n in NAMES) + " |")
    row("Case accuracy", lambda s: f"{s['accuracy']:.3f}")
    row("Severity-weighted failure rate", lambda s: f"{s['weighted_failure']:.3f}")
    row("Missed escalations (runs)", lambda s: str(s["missed_escalations"]))
    row("Consistency across runs", lambda s: f"{s['consistency']:.3f}")
    row("Latency p50 / p95 (s)", lambda s: f"{s['p50']:.2f} / {s['p95']:.2f}")
    row("Avg tokens in / out", lambda s: f"{s['avg_in']:.0f} / {s['avg_out']:.0f}")
    row("Cost per 1k cases", lambda s: "n/a" if s["cost_1k"] is None else f"${s['cost_1k']:.2f}")
    row("Summary faithfulness (judge)", lambda s: "n/a" if s["faithful"] is None else f"{s['faithful']:.2f}")
    row("API errors", lambda s: str(s["errors"]))
    L += ["\n## Accuracy by slice\n", head.replace("Metric", "Slice")]
    for sl in sorted(S["baseline"]["slices"]):
        L.append(f"| {sl} | " + " | ".join(f"{S[n]['slices'][sl]:.2f}" for n in NAMES) + " |")
    L += ["\n## What caused the change?\n",
          "Each row is a paired comparison on the same cases (change in case accuracy, 95% bootstrap CI).\n",
          "| Effect | Change | 95% CI |", "|---|---|---|"]
    for label, k in ((f"New prompt `{pb}` on `{mA}`", "p_old"), (f"New prompt `{pb}` on `{mB}`", "p_new"),
                     (f"New model `{mB}` with `{pa}`", "m_old"), (f"New model `{mB}` with `{pb}`", "m_new")):
        m, lo, hi = fx[k]
        L.append(f"| {label} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    pm, mm = (fx["p_old"][0] + fx["p_new"][0]) / 2, (fx["m_old"][0] + fx["m_new"][0]) / 2
    inter = fx["p_new"][0] - fx["p_old"][0]
    L.append(f"\nPrompt main effect {pm:+.3f}, model main effect {mm:+.3f}, interaction {inter:+.3f} "
             "(how much more the prompt helps on the new model than on the old one).\n")
    L.append("**Attribution: " + attribution(pm, mm) + "**")
    cols = ["prompt_only", "model_only", "candidate"]
    L += ["\n## Release gates (each configuration vs baseline A)\n",
          "| Gate | " + " | ".join(f"{LETTERS[n]}. {TITLES[n]}" for n in cols) + " |", "|---|---|---|---|"]
    for i, g in enumerate(gates["candidate"]):
        L.append(f"| {g[0]} | " + " | ".join("skipped" if gates[n][i][2].startswith("skipped") else f"{'PASS' if gates[n][i][1] else 'FAIL'} ({gates[n][i][2]})" for n in cols) + " |")
    L.append("\n## Regressions vs baseline (cases where a configuration is worse)\n")
    for n in cols:
        reg = [i for i in ids if S[n]["per_case"][i]["pass_rate"] < S["baseline"]["per_case"][i]["pass_rate"]]
        L.append(f"- {LETTERS[n]}. {TITLES[n]}: {', '.join(reg) if reg else 'none'}")
    L.append(f"\n**Decision: {decision}**")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/claims_eval.jsonl")
    ap.add_argument("--baseline", nargs=2, metavar=("MODEL", "PROMPT"), required=True)
    ap.add_argument("--candidate", nargs=2, metavar=("MODEL", "PROMPT"), required=True)
    ap.add_argument("--runs", type=int, default=3, help="repeats per case (measures non-determinism)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gates", default="gates.json")
    ap.add_argument("--pricing", default="pricing.json")
    ap.add_argument("--judge", metavar="MODEL", help="LLM judge model for summary faithfulness")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--mock", action="store_true", help="simulate model calls (no API key needed)")
    ap.add_argument("--force-tool", action="store_true", help="force tool_choice=tool (some models reject this)")
    ap.add_argument("--skip-ablation", action="store_true", help="only compare baseline vs candidate (2 configurations)")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    cases = load_cases(a.data)
    gates_cfg = json.loads(Path(a.gates).read_text())
    pricing = json.loads(Path(a.pricing).read_text()) if Path(a.pricing).exists() else {}
    call = make_mock(cases) if a.mock else make_real(a.temperature, a.force_tool)
    ids = [x["id"] for x in cases]

    (mA, pA), (mB, pB) = a.baseline, a.candidate
    ablate = not a.skip_ablation and mA != mB and pA != pB   # needs two different models AND two different prompts
    cells = {"baseline": (mA, pA), "candidate": (mB, pB)}
    if ablate:
        cells = {"baseline": (mA, pA), "prompt_only": (mA, pB), "model_only": (mB, pA), "candidate": (mB, pB)}
    elif not a.skip_ablation:
        print("Note: ablation needs different models and different prompts; running baseline vs candidate only.", file=sys.stderr)

    cache, S, raw = {}, {}, {}
    for name, (m, p) in cells.items():
        raw[name], S[name] = run_cell(call, cache, m, p, cases, a, pricing, name)
    b, c = S["baseline"], S["candidate"]
    pr = lambda s: [s["per_case"][i]["pass_rate"] for i in ids]

    if ablate:
        gates = {n: apply_gates(b, S[n], gates_cfg) for n in ("prompt_only", "model_only", "candidate")}
        fx = {"p_old": paired_bootstrap(pr(b), pr(S["prompt_only"])), "p_new": paired_bootstrap(pr(S["model_only"]), pr(c)),
              "m_old": paired_bootstrap(pr(b), pr(S["model_only"])), "m_new": paired_bootstrap(pr(S["prompt_only"]), pr(c))}
        passing = [n for n in gates if all(g[1] for g in gates[n])]
        quality = [n for n in gates if all(g[1] for g in gates[n] if not g[0].startswith("cost ratio"))]
        name_of = lambda n: f"{LETTERS[n]} ({cells[n][0]} + {Path(cells[n][1]).stem})"
        if passing:
            best = min(passing, key=lambda n: (S[n]["cost_1k"] if S[n]["cost_1k"] is not None else float("inf"), S[n]["p95"]))
            decision = f"PROMOTE {name_of(best)}, the cheapest configuration that passes every gate (canary next)"
        elif quality:
            decision = "DO NOT PROMOTE. Meets quality gates but fails the cost gate: " + ", ".join(name_of(n) for n in quality)
        else:
            decision = "DO NOT PROMOTE. No configuration meets the quality gates."
        report = render_ablation(cells, S, ids, fx, gates, decision)
        promote = bool(passing)
    else:
        boot = paired_bootstrap(pr(b), pr(c))
        regressions = [(i, c["per_case"][i]["slice"], b["per_case"][i]["pass_rate"], c["per_case"][i]["pass_rate"], c["per_case"][i]["diff"])
                       for i in ids if c["per_case"][i]["pass_rate"] < b["per_case"][i]["pass_rate"]]
        gates = apply_gates(b, c, gates_cfg)
        report = render(b, c, (f"{mA} + {pA}", f"{mB} + {pB}"), boot, regressions, gates)
        promote = all(g[1] for g in gates)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"report_{stamp}.md").write_text(report, encoding="utf-8")
    (out / f"raw_{stamp}.json").write_text(json.dumps({"summaries": S, "raw": raw}, indent=2, default=str), encoding="utf-8")
    print(report)
    print(f"\nSaved to {out}/report_{stamp}.md and raw_{stamp}.json", file=sys.stderr)
    sys.exit(0 if promote else 1)


if __name__ == "__main__":
    main()