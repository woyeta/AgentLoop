"""
Checks how many training examples fit in the fine-tuning context length.

The training notebook used SFTConfig(max_length=256), which keeps the first 256 tokens of
each example. This script rebuilds a sample of the training examples in the same chat
format and reports where the assistant turn starts, so you can see how often the target
output was truncated away. Needs only the tokenizer (no GPU).

Usage:
    python -m evaluation.truncation_check --sample 500 --max-length 256
"""

import argparse
import json
import os
import random
from agent_format import build_messages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def training_examples(train_data: list) -> list:
    """Rebuilds (messages, assistant_text) pairs exactly as the training notebook does."""
    examples = []
    for tool in train_data:
        for instance in tool.get("Instances", []):
            steps = instance.get("intermediate_steps", [])
            for k in range(len(steps) + 1):
                history = [{"action": s[0][0], "action_input": s[0][1], "observation": s[1]} for s in steps[:k]]
                if k < len(steps):
                    name, action_input, thought = steps[k][0]
                    target = {"thought": thought, "action": name, "action_input": action_input}
                else:
                    target = {"thought": instance.get("Final Thought", ""), "final_answer": instance.get("output", "")}
                messages = build_messages(tool.get("NLDocumentation", ""), instance.get("input", ""), history)
                examples.append((messages, json.dumps(target)))
    return examples


def _percentile(values: list, pct: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * pct / 100))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from prompts import ADAPTER_REPO

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
    with open(os.path.join(ROOT, "data", "train_data.json"), "r", encoding="utf-8") as f:
        examples = training_examples(json.load(f))
    sample = random.Random(0).sample(examples, min(args.sample, len(examples)))

    starts, totals = [], []
    for messages, target in sample:
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        full_ids = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}], tokenize=True
        )
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
        starts.append(len(prompt_ids))
        totals.append(len(full_ids))

    cut = sum(1 for s in starts if s >= args.max_length)
    print(f"Training examples: {len(examples)} (sampled {len(sample)})")
    print(f"Assistant turn starts at or after token {args.max_length}: {cut}/{len(sample)} "
          f"({100 * cut / len(sample):.1f}%)  -> no target tokens inside the training window")
    for pct in (5, 50, 95):
        print(f"p{pct}: assistant starts at token {_percentile(starts, pct)}, total length {_percentile(totals, pct)}")


if __name__ == "__main__":
    main()
