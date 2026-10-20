"""
Measures self-correction from error feedback.

For every step in the step-level results that the runtime would reject before execution
(invalid JSON, no action, unknown function, or arguments failing the schema), the exact
error observation the runtime returns is appended to the history and the model gets one
more attempt, which is scored against the same gold step.

- recovered:  the second attempt is a valid call (known function, schema-valid arguments)
- corrected:  recovered and the function is the gold one

Usage (after step_eval):
    python -m evaluation.recovery_eval --configs base qlora
"""

import argparse
import json
import os
from agent_format import build_messages
from tool_schema import build_function_specs
from react_engine import error_feedback
from evaluation.scoring import parse_output, score_step
from evaluation.step_eval import CONFIGS, DEFAULT_HELDOUT, DEFAULT_OUT_DIR, batches, load_heldout, read_jsonl


def run_config(name: str, heldout: dict, out_dir: str, max_new_tokens: int, batch_size: int = 1) -> None:
    from model_loader import generate_batch

    config = CONFIGS[name]
    examples = {s["id"]: s for s in heldout["steps"]}
    step_rows = read_jsonl(os.path.join(out_dir, f"step_{name}.jsonl"))
    caught = [r for r in step_rows if r.get("runtime_caught")]

    out_path = os.path.join(out_dir, f"recovery_{name}.jsonl")
    done = {row["id"] for row in read_jsonl(out_path)}

    jobs = []
    for first in caught:
        if first["id"] in done:
            continue
        example = examples[first["id"]]
        tool = heldout["tools"][example["tool"]]
        specs = build_function_specs(tool)
        feedback = error_feedback(parse_output(first["raw"]), specs)
        if feedback is None:
            continue
        messages = build_messages(tool["NLDocumentation"], example["query"], example["history"] + [feedback],
                                  one_shot=config["one_shot"])
        jobs.append((first, example, specs, feedback, messages))
    print(f"[{name}] {len(caught)} rejected calls, {len(jobs)} to run")

    with open(out_path, "a", encoding="utf-8") as out:
        for batch in batches(jobs, batch_size, key=lambda job: len(json.dumps(job[4]))):
            generations = generate_batch([job[4] for job in batch],
                                         use_adapter=config["use_adapter"], max_new_tokens=max_new_tokens)
            for (first, example, specs, feedback, _), generation in zip(batch, generations):
                second = score_step(example, generation.text, specs)
                recovered = bool(second["args_valid"] and second["stop_correct"])
                out.write(json.dumps({
                    "id": first["id"],
                    "task_id": first["task_id"],
                    "config": name,
                    "first_bucket": first["bucket"],
                    "feedback": feedback["observation"],
                    "raw": generation.text[:2000],
                    "second_bucket": second["bucket"],
                    "recovered": recovered,
                    "corrected": recovered and bool(second["tool_correct"]),
                    "exact": bool(second["args_exact"]),
                }, ensure_ascii=False) + "\n")
            out.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=["base", "qlora"], choices=list(CONFIGS))
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    heldout = load_heldout(args.heldout)
    for name in args.configs:
        run_config(name, heldout, args.out_dir, args.max_new_tokens, args.batch_size)


if __name__ == "__main__":
    main()
