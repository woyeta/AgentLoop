"""
Builds the frozen held-out set from ToolAlpaca's official evaluation files.

The evaluation files (eval_simulated.json, eval_real.json) pair each instruction with a
gold call sequence ("Golden_Answers": [{"Action", "Action_Input"}, ...]) but contain no
observations or final answers. So each task contributes one example: the first tool
call, predicted from an empty history, which is also how every real run starts. Tools
that also appear in data/train_data.json are dropped so every evaluated tool is unseen,
and tasks whose gold first action is "getDetails" (asking the user) are skipped.

Usage:
    python -m evaluation.build_heldout --out evaluation/heldout.json
"""

import argparse
import hashlib
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, "data", "train_data.json")
RAW_DIR = os.path.join(ROOT, "evaluation", "data")
SOURCES = {
    "simulated": "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/main/data/eval_simulated.json",
    "real": "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/main/data/eval_real.json",
}


def _download(name: str, url: str) -> tuple[list, str]:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"eval_{name}.json")
    if not os.path.exists(path):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def build(out_path: str) -> dict:
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_names = {t.get("Name") for t in json.load(f)}

    meta = {"sources": {}, "skipped": {"tool_in_train": [], "no_gold_call": 0, "asks_user": 0}}
    tools, steps = {}, []

    for source, url in SOURCES.items():
        data, sha = _download(source, url)
        meta["sources"][source] = {"url": url, "sha256": sha}

        for tool in data:
            name = tool.get("Name", "")
            if name in train_names:
                meta["skipped"]["tool_in_train"].append(name)
                continue

            tool_key = f"{source}/{name}"
            tools[tool_key] = {
                "Name": name,
                "NLDocumentation": tool.get("NLDocumentation", ""),
                "Documentation": tool.get("Documentation", ""),
                "Function_Projection": tool.get("Function_Projection", {}),
            }

            for idx, (query, gold_path) in enumerate(zip(tool["Instructions"], tool["Golden_Answers"])):
                if not gold_path:
                    meta["skipped"]["no_gold_call"] += 1
                    continue
                if gold_path[0]["Action"] == "getDetails":
                    # ToolAlpaca's placeholder for asking the user for more information, not a tool call.
                    meta["skipped"]["asks_user"] += 1
                    continue
                task_id = f"{tool_key}/{idx}"
                steps.append({
                    "id": f"{task_id}/0",
                    "task_id": task_id,
                    "source": source,
                    "tool": tool_key,
                    "step": 0,
                    "query": query,
                    "history": [],
                    "gold_path_len": len(gold_path),
                    "gold": {
                        "type": "action",
                        "action": gold_path[0]["Action"],
                        "action_input": gold_path[0]["Action_Input"],
                    },
                })

    meta["n_tools"] = len(tools)
    meta["n_tasks"] = len({s["task_id"] for s in steps})
    meta["n_steps"] = len(steps)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tools": tools, "steps": steps}, f, ensure_ascii=False, indent=1)

    print(f"Tools: {meta['n_tools']}  Tasks: {meta['n_tasks']}")
    print(f"Skipped tools seen in training: {meta['skipped']['tool_in_train']}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "heldout.json"))
    build(parser.parse_args().out)
