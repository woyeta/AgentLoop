"""
Builds model prompts in the exact format used during fine-tuning.

The training notebook (notebooks/gemma_4_toolalpaca_finetune.ipynb) formats each
step as a system message with the tool docs and a user message of the form
"Query: <query>\nHistory: <json list>", where each history entry has the keys
"action", "action_input" (a JSON string) and "observation". The runtime and the
evaluation both use this module so their prompts match training.
"""

import json
from dataclasses import dataclass
from prompts import SYSTEM_PROMPT, ONE_SHOT_EXAMPLE


@dataclass
class Generation:
    """A single model completion with basic timing information."""
    text: str
    n_tokens: int = 0
    latency_ms: float = 0.0


def history_entry(action: str, action_input, observation: str) -> dict:
    """Creates a history entry in the training format."""
    if not isinstance(action_input, str):
        action_input = json.dumps(action_input)
    return {"action": action, "action_input": action_input, "observation": observation}


def build_messages(tool_docs: str, query: str, history: list, one_shot: bool = False) -> list:
    """
    Creates chat messages for one reasoning step.

    With `one_shot`, a worked example on an unrelated tool is appended to the system
    prompt. This is the prompting baseline for the base model.
    """
    system_text = SYSTEM_PROMPT.format(tool_docs=tool_docs.strip())
    if one_shot:
        system_text += "\n\n" + ONE_SHOT_EXAMPLE
    user_text = f"Query: {query.strip()}\nHistory: {json.dumps(history)}"

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
