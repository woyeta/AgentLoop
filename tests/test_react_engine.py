import json

from agent_format import build_messages
from prompts import MAX_TURNS_LIMIT
from react_engine import react_loop
from tool_manager import is_error_response
from traces import TraceStore, RECOVERY_RATE_SQL


def scripted(outputs):
    """A fake model that returns the given outputs in order and records the histories it saw."""
    calls = []

    def generate(query, tool_docs, history):
        calls.append([dict(h) for h in history])
        return outputs[len(calls) - 1]

    generate.calls = calls
    return generate


def call(action, **args):
    return json.dumps({"thought": "t", "action": action, "action_input": args})


FINAL = json.dumps({"thought": "done", "final_answer": "The answer."})


def ok_execute(tool_info, action, action_input):
    return True, json.dumps({"status": 200, "echo": action_input})


def test_valid_call_then_final_answer(httpbin):
    generate = scripted([call("getClientRequestData", url="https://httpbin.org/get"), FINAL])
    result = react_loop("q", httpbin, generate=generate, execute=ok_execute)
    assert result["status"] == "success"
    assert result["turns_used"] == 2
    assert generate.calls[1][0]["action"] == "getClientRequestData"


def test_invalid_arguments_are_fed_back_and_fixed(httpbin):
    executed = []

    def execute(tool_info, action, action_input):
        executed.append(action_input)
        return ok_execute(tool_info, action, action_input)

    generate = scripted([
        call("sendHttpRequest", method="POST"),
        call("sendHttpRequest", method="POST", url="https://httpbin.org/post"),
        FINAL,
    ])
    result = react_loop("q", httpbin, generate=generate, execute=execute)

    assert result["status"] == "success"
    assert len(executed) == 1
    feedback = generate.calls[1][0]["observation"]
    assert "missing required field 'url'" in feedback
    assert result["steps"][0]["type"] == "error"


def test_unknown_function_is_rejected(httpbin):
    generate = scripted([call("deleteEverything"), FINAL])
    result = react_loop("q", httpbin, generate=generate, execute=ok_execute)
    assert "sendHttpRequest" in generate.calls[1][0]["observation"]
    assert result["status"] == "success"


def test_repeated_call_is_nudged_then_stopped(httpbin):
    same = call("getClientRequestData", url="https://httpbin.org/get")
    generate = scripted([same, same, same])
    result = react_loop("q", httpbin, generate=generate, execute=ok_execute)
    assert result["status"] == "loop_detected"
    assert "already called" in generate.calls[2][-1]["observation"]


def test_max_turns_is_capped(httpbin):
    outputs = [call("getClientRequestData", url=f"https://httpbin.org/get?i={i}") for i in range(50)]
    result = react_loop("q", httpbin, max_turns=50, generate=scripted(outputs), execute=ok_execute)
    assert result["status"] == "max_turns_reached"
    assert result["turns_used"] == MAX_TURNS_LIMIT


def test_consecutive_errors_end_the_run(httpbin):
    result = react_loop("q", httpbin, generate=scripted(["oops", "still not json", "nope"]), execute=ok_execute)
    assert result["status"] == "consecutive_errors"


def test_api_error_body_is_a_failure():
    assert is_error_response('{"error": "400 Bad Request", "message": "missing url"}')
    assert is_error_response('```json\n{"error": "404 Not Found"}\n```')
    assert not is_error_response('{"status": 200, "error": null}')
    assert not is_error_response("plain text")


def test_traces_are_written(httpbin, tmp_path):
    tracer = TraceStore(str(tmp_path / "traces.db"))
    generate = scripted([
        call("sendHttpRequest", method="POST"),
        call("sendHttpRequest", method="POST", url="https://httpbin.org/post"),
        FINAL,
    ])
    result = react_loop("q", httpbin, generate=generate, execute=ok_execute, tracer=tracer)

    run = tracer.get_run(result["run_id"])
    assert run["status"] == "success"
    assert [s["type"] for s in run["steps"]] == ["error", "thought_action", "final_answer"]
    assert run["steps"][0]["validation_error"]

    recovery = tracer.query(RECOVERY_RATE_SQL)[0]
    assert recovery["invalid_steps"] == 1
    assert recovery["recovered"] == 1
    tracer.close()


def test_prompt_matches_training_format():
    history = [{"action": "a", "action_input": "{}", "observation": "o"}]
    messages = build_messages("docs", " what? ", history)
    assert messages[0]["role"] == "system" and "docs" in messages[0]["content"]
    assert messages[1]["content"] == 'Query: what?\nHistory: ' + json.dumps(history)
