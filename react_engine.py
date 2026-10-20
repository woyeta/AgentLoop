"""
Executes the reasoning and action loop.
"""

import json
import logging
import time
from agent_format import Generation, history_entry
from json_utils import sanitize_model_response
from tool_manager import execute_tool_action
from tool_schema import build_function_specs, validate_arguments, canonical_args, canonical_key
from prompts import (
    ERR_MSG_INVALID_JSON,
    ERR_MSG_NO_ACTION,
    ERR_MSG_INVALID_ACTION,
    ERR_MSG_INVALID_ARGS,
    ERR_MSG_REPEATED_CALL,
    ERR_MSG_API_FAIL,
    MAX_TURNS_LIMIT,
    MAX_CONSECUTIVE_ERRORS,
)

logger = logging.getLogger(__name__)


def _build_result(final_answer, steps, turns_used, status, run_id=None) -> dict:
    """Creates a result dictionary for loop output."""
    return {
        "final_answer": final_answer,
        "steps": steps,
        "turns_used": turns_used,
        "status": status,
        "run_id": run_id,
    }


def _action_name(json_response: dict) -> str:
    """Returns the action as a string; non-string actions (e.g. an object) become their JSON text."""
    action = json_response.get("action", "")
    if action and not isinstance(action, str):
        return json.dumps(action)
    return action or ""


def error_feedback(json_response, function_specs: dict) -> dict | None:
    """
    Returns the history entry the loop adds for a call it rejects before execution
    (invalid JSON, no action, unknown action, invalid arguments), or None if the output
    is a final answer or a valid call. Mirrors the checks in react_loop and is used by
    the recovery evaluation.
    """
    if not isinstance(json_response, dict):
        return history_entry("", "", ERR_MSG_INVALID_JSON)
    if json_response.get("final_answer", ""):
        return None

    action = _action_name(json_response)
    action_input = json_response.get("action_input", {})
    if not action:
        return history_entry("", action_input, ERR_MSG_NO_ACTION)

    spec = function_specs.get(action)
    if spec is None:
        return history_entry(action, action_input, ERR_MSG_INVALID_ACTION.format(
            tool_functions_list=", ".join(function_specs.keys())
        ))

    validation = validate_arguments(spec, action_input)
    if not validation.ok:
        return history_entry(action, action_input, ERR_MSG_INVALID_ARGS.format(
            action=action, details=validation.message
        ))
    return None


def _default_generate(user_query: str, tool_docs: str, history: list) -> Generation:
    from model_loader import run_inference
    return run_inference(user_query, tool_docs, history)


def react_loop(
    user_query: str,
    tool_info: dict,
    max_turns: int = 5,
    generate=None,
    execute=None,
    tracer=None,
) -> dict:
    """
    Runs the reasoning and action loop for a query using specified tool capabilities.

    `generate(query, tool_docs, history)` returns a Generation (or str), and
    `execute(tool_info, action, action_input)` returns (ok, observation). Both default
    to the real model and tool manager and can be replaced in tests. If a `tracer`
    (traces.TraceStore) is given, every step is logged to SQLite.
    """
    generate = generate or _default_generate
    execute = execute or execute_tool_action
    max_turns = max(1, min(int(max_turns), MAX_TURNS_LIMIT))

    logger.info(f"Starting ReAct loop for query: {user_query}")

    tool_documentation = tool_info.get("NLDocumentation", "")
    function_specs = build_function_specs(tool_info)
    run_id = tracer.start_run(tool_info.get("Name", ""), user_query) if tracer else None

    history = []
    steps = []
    consecutive_errors = 0
    last_call_key = None
    repeat_count = 0

    def record(step: dict, raw: str = "", parse_ok: bool = True, validation_error=None,
               model_ms=None, tool_ms=None, n_tokens=None):
        steps.append(step)
        if tracer:
            tracer.log_step(
                run_id, step["turn"],
                type=step["type"], thought=step.get("thought"), action=step.get("action"),
                args_json=step.get("action_input"), raw_output=raw, parse_ok=parse_ok,
                validation_error=validation_error, error=step.get("error"),
                observation=step.get("observation"), model_ms=model_ms, tool_ms=tool_ms,
                n_tokens=n_tokens,
            )

    def finish(final_answer, turns_used, status):
        if tracer:
            tracer.end_run(run_id, status, turns_used)
        return _build_result(final_answer, steps, turns_used, status, run_id)

    for turn in range(max_turns):
        turn_num = turn + 1
        logger.info(f"Turn {turn_num}")

        start = time.perf_counter()
        generation = generate(user_query, tool_documentation, history)
        if isinstance(generation, str):
            generation = Generation(generation)
        model_ms = generation.latency_ms or (time.perf_counter() - start) * 1000
        model_response = generation.text
        timing = {"raw": model_response, "model_ms": model_ms, "n_tokens": generation.n_tokens}

        valid_json, json_response = sanitize_model_response(model_response)

        if not valid_json or not isinstance(json_response, dict):
            consecutive_errors += 1
            record({
                "turn": turn_num,
                "type": "error",
                "error": "Invalid JSON response from model",
                "raw_response": str(model_response)[:200],
            }, parse_ok=False, **timing)
            history.append(history_entry("", "", ERR_MSG_INVALID_JSON))
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return finish(None, turn_num, "consecutive_errors")
            continue

        thought = json_response.get("thought", "")

        final_answer = json_response.get("final_answer", "")
        if final_answer:
            record({
                "turn": turn_num,
                "type": "final_answer",
                "thought": thought,
                "final_answer": final_answer,
            }, **timing)
            return finish(final_answer, turn_num, "success")

        action = _action_name(json_response)
        action_input = json_response.get("action_input", {})

        if not action:
            consecutive_errors += 1
            record({
                "turn": turn_num,
                "type": "error",
                "thought": thought,
                "error": "No action specified",
            }, **timing)
            history.append(history_entry("", action_input, ERR_MSG_NO_ACTION))
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return finish(None, turn_num, "consecutive_errors")
            continue

        spec = function_specs.get(action)

        if spec is None:
            tool_functions_list = ", ".join(function_specs.keys())
            consecutive_errors += 1
            record({
                "turn": turn_num,
                "type": "error",
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "error": f"Invalid action: '{action}'",
            }, **timing)
            history.append(history_entry(
                action, action_input,
                ERR_MSG_INVALID_ACTION.format(tool_functions_list=tool_functions_list),
            ))
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return finish(None, turn_num, "consecutive_errors")
            continue

        validation = validate_arguments(spec, action_input)

        if not validation.ok:
            consecutive_errors += 1
            record({
                "turn": turn_num,
                "type": "error",
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "error": f"Invalid arguments: {validation.message}",
            }, validation_error=validation.message, **timing)
            history.append(history_entry(
                action, action_input,
                ERR_MSG_INVALID_ARGS.format(action=action, details=validation.message),
            ))
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return finish(None, turn_num, "consecutive_errors")
            continue

        call_key = canonical_key(action, canonical_args(spec, validation.args))
        if call_key == last_call_key:
            repeat_count += 1
            record({
                "turn": turn_num,
                "type": "error",
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "error": "Repeated identical tool call",
            }, **timing)
            if repeat_count >= 2:
                return finish(None, turn_num, "loop_detected")
            history.append(history_entry(
                action, action_input, ERR_MSG_REPEATED_CALL.format(action=action)
            ))
            continue

        tool_start = time.perf_counter()
        valid_api_response, api_response = execute(tool_info, action, validation.args)
        tool_ms = (time.perf_counter() - tool_start) * 1000
        last_call_key = call_key
        repeat_count = 0

        if not valid_api_response:
            consecutive_errors += 1
            record({
                "turn": turn_num,
                "type": "error",
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "error": f"API call failed: {api_response}",
            }, tool_ms=tool_ms, **timing)
            history.append(history_entry(
                action, action_input, ERR_MSG_API_FAIL.format(api_response=api_response)
            ))
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return finish(None, turn_num, "consecutive_errors")
            continue

        record({
            "turn": turn_num,
            "type": "thought_action",
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": api_response,
        }, tool_ms=tool_ms, **timing)
        history.append(history_entry(action, action_input, api_response))
        consecutive_errors = 0

    return finish(None, max_turns, "max_turns_reached")
