import json

from evaluation.scoring import score_step
from tool_schema import build_function_specs


def action_example():
    return {
        "step": 0,
        "gold": {
            "type": "action",
            "action": "simulateStatusCode",
            "action_input": '{"url": "https://httpbin.org/status", "statusCode": 404}',
        },
    }


def output(action, args):
    return json.dumps({"thought": "t", "action": action, "action_input": args})


def test_exact_match(httpbin):
    specs = build_function_specs(httpbin)
    row = score_step(action_example(), output("simulateStatusCode", {"url": "https://httpbin.org/status",
                                                                     "statusCode": "404"}), specs)
    assert row["tool_correct"] and row["args_valid"] and row["args_exact"]
    assert row["bucket"] == "ok"
    assert row["gold_valid"]
    assert not row["runtime_caught"]


def test_model_style_string_arguments(httpbin):
    specs = build_function_specs(httpbin)
    raw = json.dumps({"thought": "t", "action": "simulateStatusCode",
                      "action_input": '{"url": "https://httpbin.org/status", "statusCode": 404}'})
    assert score_step(action_example(), raw, specs)["args_exact"]


def test_buckets(httpbin):
    specs = build_function_specs(httpbin)
    example = action_example()
    assert score_step(example, "not json", specs)["bucket"] == "invalid_json"
    assert score_step(example, output("madeUp", {}), specs)["bucket"] == "hallucinated_function"
    assert score_step(example, output("getClientRequestData", {"url": "x"}), specs)["bucket"] == "wrong_function"
    missing = score_step(example, output("simulateStatusCode", {"url": "x"}), specs)
    assert missing["bucket"] == "missing_required" and missing["runtime_caught"]
    wrong = score_step(example, output("simulateStatusCode", {"url": "x", "statusCode": 500}), specs)
    assert wrong["bucket"] == "wrong_values" and wrong["args_f1"] == 0.0
    early = score_step(example, json.dumps({"thought": "t", "final_answer": "a"}), specs)
    assert early["bucket"] == "premature_final" and not early["stop_correct"]


def test_final_step(httpbin):
    specs = build_function_specs(httpbin)
    example = {"step": 1, "gold": {"type": "final", "final_answer": "a"}}
    assert score_step(example, json.dumps({"thought": "t", "final_answer": "b"}), specs)["stop_correct"]
    assert score_step(example, output("simulateStatusCode", {}), specs)["bucket"] == "missed_final"
