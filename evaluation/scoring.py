"""
Scores one model output against a gold step.

Metric definitions (action steps unless noted):
- parse_ok:       output parses to a JSON object with an "action" or "final_answer" key
- tool_correct:   predicted action name equals the gold action exactly
- args_valid:     the predicted function exists and its arguments pass the pydantic schema
- args_exact:     correct tool and canonical arguments equal the gold canonical arguments
- args_f1:        F1 over (key, value) pairs of canonical arguments (0 if wrong tool)
- stop_correct:   final steps: the model answered; action steps: it did not answer early
"""

from json_utils import sanitize_model_response
from tool_schema import validate_arguments, canonical_args, key_f1


def parse_output(raw_text: str) -> dict | None:
    ok, parsed = sanitize_model_response(raw_text)
    if not ok or not isinstance(parsed, dict):
        return None
    return parsed


def _validation_bucket(error_types: list) -> str:
    if "missing" in error_types:
        return "missing_required"
    if "extra_forbidden" in error_types:
        return "unknown_field"
    return "wrong_type"


def score_step(example: dict, raw_text: str, specs: dict) -> dict:
    gold = example["gold"]
    parsed = parse_output(raw_text)
    parse_ok = parsed is not None and ("action" in parsed or "final_answer" in parsed)

    row = {"gold_type": gold["type"], "parse_ok": parse_ok}
    pred_final = bool(parsed.get("final_answer")) if parsed else False
    pred_action = (parsed.get("action") or "") if parsed else ""
    pred_input = parsed.get("action_input", {}) if parsed else {}
    row["pred_action"] = pred_action if isinstance(pred_action, str) else str(pred_action)

    if gold["type"] == "final":
        row["stop_correct"] = pred_final
        if not parse_ok:
            row["bucket"] = "invalid_json"
        else:
            row["bucket"] = "ok" if pred_final else "missed_final"
        return row

    gold_action = gold["action"]
    gold_spec = specs.get(gold_action)
    pred_spec = specs.get(row["pred_action"])

    row["gold_action"] = gold_action
    row["gold_valid"] = validate_arguments(gold_spec, gold["action_input"]).ok if gold_spec else None
    row["stop_correct"] = not pred_final
    row["tool_correct"] = parse_ok and row["pred_action"] == gold_action
    row["args_valid"] = False
    row["args_exact"] = False
    row["args_f1"] = 0.0
    row["validation_error"] = None

    if pred_spec is not None and not pred_final:
        validation = validate_arguments(pred_spec, pred_input)
        row["args_valid"] = validation.ok
        row["validation_error"] = validation.message or None
        error_types = validation.error_types
    else:
        error_types = []

    if row["tool_correct"]:
        pred_canonical = canonical_args(gold_spec, pred_input)
        gold_canonical = canonical_args(gold_spec, gold["action_input"])
        row["args_exact"] = pred_canonical == gold_canonical
        row["args_f1"] = key_f1(pred_canonical, gold_canonical)

    if not parse_ok:
        row["bucket"] = "invalid_json"
    elif pred_final:
        row["bucket"] = "premature_final"
    elif not row["pred_action"]:
        row["bucket"] = "no_action"
    elif pred_spec is None:
        row["bucket"] = "hallucinated_function"
    elif not row["tool_correct"]:
        row["bucket"] = "wrong_function"
    elif not row["args_valid"]:
        row["bucket"] = _validation_bucket(error_types)
    elif not row["args_exact"]:
        row["bucket"] = "wrong_values"
    else:
        row["bucket"] = "ok"

    # Calls the runtime rejects before execution and answers with error feedback.
    row["runtime_caught"] = not pred_final and not (parse_ok and pred_spec is not None and row["args_valid"])
    return row
