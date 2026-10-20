# AgentLoop

A Flask-based web app that executes multi-turn tool-calling tasks using a finetuned Gemma 4 model and a Reasoning + Action (ReAct) loop.

The app uses a 4-bit quantized model paired with a LoRA adapter trained on ToolAlpaca for step-by-step reasoning, tool selection, and action execution.

- **Base Model**: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- **LoRA Adapter**: [woyeta/gemma-4-ToolAlpaca-FineTune-C](https://huggingface.co/woyeta/gemma-4-ToolAlpaca-FineTune-C)
- **Tool Alpaca Dataset**: [github.com/tangqiaoyu/ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/woyeta/AgentLoop.git
cd AgentLoop
pip install -r requirements.txt
```

## Running the App

Run the Flask app locally:

```bash
flask --app app run
```

### Google Colab

Run on a GPU instance using [`colab.ipynb`](colab.ipynb). Make sure to set `GEMINI_API_KEY` and `NGROK_AUTH_TOKEN` in your Colab secrets.

## Tool Runtime

Each step of the ReAct loop ([`react_engine.py`](react_engine.py)) is checked before any tool runs:

- **Schema validation**: argument schemas are generated as pydantic models from each tool's OpenAPI spec ([`tool_schema.py`](tool_schema.py)). Invalid arguments are not executed; the validation error (e.g. `missing required field 'url'`) is returned to the model as the observation so it can retry.
- **Guards**: a server-side step limit, loop detection (an identical repeated call gets a warning, a second repeat ends the run), a consecutive-error limit, and a per-call tool timeout.
- **Traces**: every step (thought, action, arguments, raw output, validation error, model and tool latency, tokens) is stored in SQLite ([`traces.py`](traces.py)) and served at `GET /api/runs/<run_id>`.

Prompts use the same format as fine-tuning ([`agent_format.py`](agent_format.py)).

## Evaluation

The evaluation uses ToolAlpaca's official evaluation sets, restricted to tools that are not in the training data, and compares base Gemma (zero-shot and one-shot) with the QLoRA adapter. It also measures how often the model fixes a rejected call after receiving the runtime's error feedback.

Run [`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb) on a Colab GPU, or run the steps directly:

```bash
python -m evaluation.step_eval --configs base base_one_shot qlora
python -m evaluation.recovery_eval --configs base base_one_shot qlora
python -m evaluation.report
```

The frozen held-out set is [`evaluation/heldout.json`](evaluation/heldout.json) (rebuild with `python -m evaluation.build_heldout`). Results are in [`EVAL.md`](EVAL.md): on 186 tasks over 19 unseen tools, base Gemma selects the right tool 80.1% of the time zero-shot, and one error-feedback retry raises schema-valid calls from 80.1% to 86.0%. The current QLoRA adapter scores 36.0% because of the training issue below.

## Training

[`notebooks/gemma_4_toolalpaca_finetune.ipynb`](notebooks/gemma_4_toolalpaca_finetune.ipynb): QLoRA (4-bit NF4, r=4, α=8, all linear layers), learning rate 5e-5 constant, 1 epoch, effective batch size 8, on all ToolAlpaca training instances (11,970 step examples, 80/20 train/validation split).

Known issue: training used `max_length=256`, but every training prompt (system prompt plus tool docs) is longer than that, so the assistant output was truncated out of the training window in all 500 sampled examples (`python -m evaluation.truncation_check`). The adapter never trained on a target token. A fix needs completion-only loss and `max_length` of about 2048 (the 95th-percentile example is 1,893 tokens).

## Tests

```bash
python -m pytest
```
