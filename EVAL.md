# Evaluation

Generated 2026-10-20 by `python -m evaluation.report` on a Colab Tesla T4 (4-bit NF4, fp16 compute, greedy decoding, `max_new_tokens=512`, batch size 16).

## Setup

- Held-out set: ToolAlpaca's official evaluation files (simulated, real); 19 tools, 186 tasks. Tools also in the training data were removed (AviationAPI, Free Dictionary), as were 8 tasks whose gold action is asking the user.
- Each task is scored on its first tool call, predicted from an empty history (the evaluation files have no gold observations for later steps). Greedy decoding.
- Evaluated: Base Gemma (zero-shot): 186 tasks, Base Gemma (one-shot): 186 tasks, QLoRA: 186 tasks.
- Argument schemas built from the OpenAPI docs for 88/88 functions. Exact match and key F1 use only tasks whose gold call passes its own schema.
- Intervals are 95% bootstrap intervals over tasks.

## Results

| Metric | Base Gemma (zero-shot) | Base Gemma (one-shot) | QLoRA |
|---|---|---|---|
| Valid JSON % | 91.4 [87.1, 95.2] | 91.4 [87.6, 95.2] | 67.7 [60.8, 74.2] |
| Tool selection % | 80.1 [74.7, 85.5] | 82.8 [77.4, 88.2] | 36.0 [29.0, 43.0] |
| Argument validity % | 80.1 [74.2, 85.5] | 76.3 [69.9, 81.7] | 35.5 [29.0, 43.0] |
| Argument exact match % | 52.3 [45.4, 59.8] | 54.6 [47.4, 61.8] | 23.6 [17.3, 29.9] |
| Argument key F1 | 0.61 [0.55, 0.68] | 0.64 [0.58, 0.71] | 0.28 [0.22, 0.35] |
| Answered without calling a tool % | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.5 [0.0, 1.6] |
| Avg tokens generated / step | 173 | 154 | 111 |
| Avg generation time / step, batched (s) | 9.51 | 7.01 | 9.83 |

## Tool selection by subset

| Subset | Tasks | Base Gemma (zero-shot) | Base Gemma (one-shot) | QLoRA |
|---|---|---|---|---|
| Simulated tools | 92 | 78.3 | 79.3 | 45.7 |
| Real-API tools | 94 | 81.9 | 86.2 | 26.6 |
| Single-call tasks | 135 | 84.4 | 87.4 | 38.5 |
| Multi-call tasks | 51 | 68.6 | 70.6 | 29.4 |

## Error analysis (action steps)

| Bucket | Base Gemma (zero-shot) | Base Gemma (one-shot) | QLoRA |
|---|---|---|---|
| ok | 89 (47.8%) | 93 (50.0%) | 41 (22.0%) |
| wrong_values | 46 (24.7%) | 41 (22.0%) | 19 (10.2%) |
| invalid_json | 16 (8.6%) | 16 (8.6%) | 60 (32.3%) |
| hallucinated_function | 7 (3.8%) | 7 (3.8%) | 51 (27.4%) |
| missing_required | 12 (6.5%) | 17 (9.1%) | 4 (2.2%) |
| wrong_function | 14 (7.5%) | 9 (4.8%) | 7 (3.8%) |
| wrong_type | 2 (1.1%) | 2 (1.1%) | 3 (1.6%) |
| unknown_field | 0 (0.0%) | 1 (0.5%) | 0 (0.0%) |
| premature_final | 0 (0.0%) | 0 (0.0%) | 1 (0.5%) |

Dataset check: 12/186 gold calls fail their own OpenAPI schema (noise in the reference data, which caps argument validity and exact match).

## Recovery from error feedback

Calls the runtime rejects before execution get the runtime's error observation and one retry.

| Config | Rejected calls | Recovered % | Corrected % | Exact match % |
|---|---|---|---|---|
| Base Gemma (zero-shot) | 37 | 29.7 | 27.0 | 8.1 |
| Base Gemma (one-shot) | 44 | 43.2 | 43.2 | 11.4 |

| Config | First error | Count | Recovered % |
|---|---|---|---|
| Base Gemma (zero-shot) | invalid_json | 16 | 25.0 |
| Base Gemma (zero-shot) | missing_required | 12 | 25.0 |
| Base Gemma (zero-shot) | hallucinated_function | 7 | 28.6 |
| Base Gemma (zero-shot) | wrong_type | 2 | 100.0 |
| Base Gemma (one-shot) | missing_required | 17 | 70.6 |
| Base Gemma (one-shot) | invalid_json | 16 | 31.2 |
| Base Gemma (one-shot) | hallucinated_function | 7 | 0.0 |
| Base Gemma (one-shot) | wrong_type | 2 | 100.0 |
| Base Gemma (one-shot) | wrong_function | 1 | 0.0 |
| Base Gemma (one-shot) | unknown_field | 1 | 0.0 |

With one retry, the share of tasks ending in a schema-valid call rises from 80.1% to 86.0% (zero-shot, 149 → 160 of 186) and from 76.3% to 86.6% (one-shot, 142 → 161 of 186).

## Notes

- **The QLoRA adapter is broken by its training setup, not by the method.** The training notebook used `max_length=256`, keeping the first 256 tokens of each example. `python -m evaluation.truncation_check` shows the assistant turn starts after token 256 in 500/500 sampled training examples (p5 474 tokens, median 852, p95 1,800), so no target token was ever trained on. The adapter only learned to continue the system prompt and tool docs, which degrades JSON output (32% invalid) and makes it invent function names (27%), sometimes in Gemma's native `call: fn{...}` syntax. Recovery was not run for QLoRA.
- The one-shot gain on tool selection (+2.7 points) is within the confidence intervals, so it is not a clear improvement.
- Results are from the `max_new_tokens=512` setting the app uses. At 256 tokens, most invalid-JSON failures were outputs cut off mid-thought.
- Error feedback was never part of training, so recovery measures what the model does without being trained for it.
