"""
Builds EVAL.md from the step-level and recovery results.

Confidence intervals are 95% bootstrap intervals that resample whole tasks, since steps
of the same task are not independent.

Usage:
    python -m evaluation.report
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from datetime import date
from tool_schema import build_function_specs
from evaluation.step_eval import CONFIGS, DEFAULT_HELDOUT, DEFAULT_OUT_DIR, ROOT, load_heldout, read_jsonl

N_BOOTSTRAP = 1000

LABELS = {
    "base": "Base Gemma (zero-shot)",
    "base_one_shot": "Base Gemma (one-shot)",
    "qlora": "QLoRA",
}


def _mean(values: list):
    return sum(values) / len(values) if values else None


def _metric_functions() -> dict:
    """Each metric maps a list of rows to a value (None if it does not apply)."""
    def action(rows):
        return [r for r in rows if r["gold_type"] == "action"]

    def gold_valid(rows):
        return [r for r in action(rows) if r.get("gold_valid")]

    return {
        "Valid JSON %": lambda rows: _mean([r["parse_ok"] for r in rows]),
        "Tool selection %": lambda rows: _mean([r["tool_correct"] for r in action(rows)]),
        "Argument validity %": lambda rows: _mean([r["args_valid"] for r in action(rows)]),
        "Argument exact match %": lambda rows: _mean([r["args_exact"] for r in gold_valid(rows)]),
        "Argument key F1": lambda rows: _mean([r["args_f1"] for r in gold_valid(rows)]),
        "Answered without calling a tool %": lambda rows: _mean([not r["stop_correct"] for r in action(rows)]),
    }


def _bootstrap(rows: list, metric) -> tuple:
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    tasks = list(by_task)
    rng = random.Random(0)
    samples = []
    for _ in range(N_BOOTSTRAP):
        resampled = [row for t in rng.choices(tasks, k=len(tasks)) for row in by_task[t]]
        value = metric(resampled)
        if value is not None:
            samples.append(value)
    if not samples:
        return None, None
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]


def _fmt(value, low=None, high=None, percent=True) -> str:
    if value is None:
        return "–"
    scale = 100 if percent else 1
    digits = 1 if percent else 2
    text = f"{value * scale:.{digits}f}"
    if low is not None:
        text += f" [{low * scale:.{digits}f}, {high * scale:.{digits}f}]"
    return text


def _table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def build_report(heldout: dict, out_dir: str) -> str:
    results = {name: read_jsonl(os.path.join(out_dir, f"step_{name}.jsonl")) for name in CONFIGS}
    results = {name: rows for name, rows in results.items() if rows}
    if not results:
        raise SystemExit(f"No step results found in {out_dir}")
    configs = list(results)
    metrics = _metric_functions()
    meta = heldout["meta"]

    n_steps = {name: len(rows) for name, rows in results.items()}
    n_tasks = {name: len({r["task_id"] for r in rows}) for name, rows in results.items()}

    specs = [spec for tool in heldout["tools"].values() for spec in build_function_specs(tool).values()]
    with_schema = sum(1 for s in specs if s.model is not None)

    parts = [
        "# Evaluation",
        "",
        f"Generated {date.today().isoformat()} by `python -m evaluation.report`.",
        "",
        "## Setup",
        "",
        f"- Held-out set: ToolAlpaca's official evaluation files ({', '.join(meta['sources'])}); "
        f"{meta['n_tools']} tools, {meta['n_tasks']} tasks. Tools also in the training data were removed "
        f"({', '.join(meta['skipped']['tool_in_train']) or 'none'}), as were "
        f"{meta['skipped'].get('asks_user', 0)} tasks whose gold action is asking the user.",
        "- Each task is scored on its first tool call, predicted from an empty history "
        "(the evaluation files have no gold observations for later steps). Greedy decoding.",
        f"- Evaluated: " + ", ".join(f"{LABELS[c]}: {n_tasks[c]} tasks" for c in configs) + ".",
        f"- Argument schemas built from the OpenAPI docs for {with_schema}/{len(specs)} functions. "
        "Exact match and key F1 use only tasks whose gold call passes its own schema.",
        "- Intervals are 95% bootstrap intervals over tasks.",
        "",
        "## Results",
        "",
    ]

    rows = []
    for metric_name, metric in metrics.items():
        percent = "%" in metric_name
        row = [metric_name]
        for c in configs:
            value = metric(results[c])
            low, high = _bootstrap(results[c], metric) if value is not None else (None, None)
            row.append(_fmt(value, low, high, percent))
        rows.append(row)
    rows.append(["Avg tokens generated / step"] +
                [f"{_mean([r['n_tokens'] for r in results[c]]):.0f}" for c in configs])
    rows.append(["Avg generation time / step, batched (s)"] +
                [f"{_mean([r['latency_ms'] for r in results[c]]) / 1000:.2f}" for c in configs])
    parts += [_table(["Metric"] + [LABELS[c] for c in configs], rows), ""]

    parts += ["## Tool selection by subset", ""]
    subsets = {
        "Simulated tools": lambda r: r["source"] == "simulated",
        "Real-API tools": lambda r: r["source"] == "real",
        "Single-call tasks": lambda r: r.get("gold_path_len", 1) == 1,
        "Multi-call tasks": lambda r: r.get("gold_path_len", 1) > 1,
    }
    subset_rows = []
    for label, keep in subsets.items():
        row = [label, sum(1 for r in results[configs[0]] if keep(r))]
        for c in configs:
            row.append(_fmt(metrics["Tool selection %"]([r for r in results[c] if keep(r)])))
        subset_rows.append(row)
    parts += [_table(["Subset", "Tasks"] + [LABELS[c] for c in configs], subset_rows), ""]

    parts += ["## Error analysis (action steps)", ""]
    bucket_counts = {c: Counter(r["bucket"] for r in results[c] if r["gold_type"] == "action") for c in configs}
    buckets = sorted({b for counts in bucket_counts.values() for b in counts},
                     key=lambda b: -sum(bucket_counts[c][b] for c in configs))
    bucket_rows = []
    for b in buckets:
        row = [b]
        for c in configs:
            total = sum(bucket_counts[c].values())
            row.append(f"{bucket_counts[c][b]} ({100 * bucket_counts[c][b] / total:.1f}%)")
        bucket_rows.append(row)
    parts += [_table(["Bucket"] + [LABELS[c] for c in configs], bucket_rows), ""]

    gold_rows = [r for r in next(iter(results.values())) if r["gold_type"] == "action" and r["gold_valid"] is not None]
    invalid_gold = sum(1 for r in gold_rows if not r["gold_valid"])
    parts += [f"Dataset check: {invalid_gold}/{len(gold_rows)} gold calls fail their own OpenAPI schema "
              "(noise in the reference data, which caps argument validity and exact match).", ""]

    recovery = {c: read_jsonl(os.path.join(out_dir, f"recovery_{c}.jsonl")) for c in configs}
    recovery = {c: rows_ for c, rows_ in recovery.items() if rows_}
    if recovery:
        parts += [
            "## Recovery from error feedback",
            "",
            "Calls the runtime rejects before execution get the runtime's error observation and one retry.",
            "",
        ]
        rec_rows = []
        for c, rows_ in recovery.items():
            rec_rows.append([
                LABELS[c],
                len(rows_),
                _fmt(_mean([r["recovered"] for r in rows_])),
                _fmt(_mean([r["corrected"] for r in rows_])),
                _fmt(_mean([r["exact"] for r in rows_])),
            ])
        parts += [_table(["Config", "Rejected calls", "Recovered %", "Corrected %", "Exact match %"], rec_rows), ""]

        by_bucket_rows = []
        for c, rows_ in recovery.items():
            grouped = defaultdict(list)
            for r in rows_:
                grouped[r["first_bucket"]].append(r["recovered"])
            for b, values in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
                by_bucket_rows.append([LABELS[c], b, len(values), _fmt(_mean(values))])
        parts += [_table(["Config", "First error", "Count", "Recovered %"], by_bucket_rows), ""]

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", default=os.path.join(ROOT, "EVAL.md"))
    args = parser.parse_args()

    report = build_report(load_heldout(args.heldout), args.out_dir)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
