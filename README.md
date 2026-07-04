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
