"""
Step-level evaluation.

For every held-out step the model gets the gold history up to that step (empty for the
first tool call) and predicts the next output, which is scored against the gold step
(see evaluation/scoring.py). Results
are appended to evaluation/results/step_<config>.jsonl, and finished steps are skipped on
re-runs, so an interrupted Colab session can simply be restarted.

Usage:
    python -m evaluation.step_eval --configs base base_one_shot qlora [--limit 50]
"""

import argparse
import json
import os
from agent_format import build_messages
from tool_schema import build_function_specs
from evaluation.scoring import score_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HELDOUT = os.path.join(ROOT, "evaluation", "heldout.json")
DEFAULT_OUT_DIR = os.path.join(ROOT, "evaluation", "results")

CONFIGS = {
    "base": {"use_adapter": False, "one_shot": False},
    "base_one_shot": {"use_adapter": False, "one_shot": True},
    "qlora": {"use_adapter": True, "one_shot": False},
}


def load_heldout(path: str = DEFAULT_HELDOUT) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_steps(heldout: dict, limit: int | None) -> list:
    """Deterministically picks the first `limit` tasks (all of their steps)."""
    steps = heldout["steps"]
    if not limit:
        return steps
    task_ids = list(dict.fromkeys(s["task_id"] for s in steps))[:limit]
    keep = set(task_ids)
    return [s for s in steps if s["task_id"] in keep]


def run_config(name: str, heldout: dict, steps: list, out_dir: str, max_new_tokens: int) -> None:
    from model_loader import generate

    config = CONFIGS[name]
    out_path = os.path.join(out_dir, f"step_{name}.jsonl")
    done = {row["id"] for row in read_jsonl(out_path)}
    todo = [s for s in steps if s["id"] not in done]
    print(f"[{name}] {len(done)} done, {len(todo)} to run")

    with open(out_path, "a", encoding="utf-8") as out:
        for i, example in enumerate(todo, 1):
            tool = heldout["tools"][example["tool"]]
            specs = build_function_specs(tool)
            messages = build_messages(tool["NLDocumentation"], example["query"], example["history"],
                                      one_shot=config["one_shot"])
            generation = generate(messages, use_adapter=config["use_adapter"], max_new_tokens=max_new_tokens)

            row = {
                "id": example["id"],
                "task_id": example["task_id"],
                "source": example["source"],
                "tool": example["tool"],
                "step": example["step"],
                "gold_path_len": example.get("gold_path_len", 1),
                "config": name,
                "raw": generation.text[:2000],
                "n_tokens": generation.n_tokens,
                "latency_ms": round(generation.latency_ms, 1),
                **score_step(example, generation.text, specs),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"[{name}] {i}/{len(todo)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N tasks")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    heldout = load_heldout(args.heldout)
    steps = select_steps(heldout, args.limit)
    for name in args.configs:
        run_config(name, heldout, steps, args.out_dir, args.max_new_tokens)


if __name__ == "__main__":
    main()
