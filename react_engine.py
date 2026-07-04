"""
Executes the reasoning and action loop.
"""

import json
import logging
from model_loader import run_inference
from json_utils import sanitize_model_response
from tool_manager import execute_tool_action
from prompts import (
    ERR_MSG_INVALID_JSON,
    ERR_MSG_NO_ACTION,
    ERR_MSG_INVALID_ACTION,
    ERR_MSG_API_FAIL,
)

logger = logging.getLogger(__name__)


def _build_result(final_answer, steps, turns_used, status) -> dict:
    """Creates a result dictionary for loop output."""
    return {
        "final_answer": final_answer,
        "steps": steps,
        "turns_used": turns_used,
        "status": status,
    }


def react_loop(user_query: str, tool_info: dict, max_turns: int = 5) -> dict:
    """
    Runs the reasoning and action loop for a query using specified tool capabilities.
    """
    logger.info(f"Starting ReAct loop for query: {user_query}")

    tool_documentation = tool_info.get("NLDocumentation", "")
    tool_function_projection = tool_info.get("Function_Projection", {})

    history = []
    consecutive_errors = 0
    steps = []

    for turn in range(max_turns):
        turn_num = turn + 1
        logger.info(f"Turn {turn_num}")

        model_response = run_inference(user_query, tool_documentation, history)
        valid_json, json_response = sanitize_model_response(model_response)

        if not valid_json:
            consecutive_errors += 1
            steps.append({
                "turn": turn_num,
                "type": "error",
                "error": "Invalid JSON response from model",
                "raw_response": str(model_response)[:200],
            })
            history.append({"response_format": ERR_MSG_INVALID_JSON})
            if consecutive_errors >= 3:
                return _build_result(None, steps, turn_num, "consecutive_errors")
            continue

        thought = json_response.get("thought", "")

        final_answer = json_response.get("final_answer", "")
        if final_answer:
            steps.append({
                "turn": turn_num,
                "type": "final_answer",
                "thought": thought,
                "final_answer": final_answer,
            })
            return _build_result(final_answer, steps, turn_num, "success")

        action = json_response.get("action", "")
        action_input = json_response.get("action_input", {})

        if action == "":
            consecutive_errors += 1
            steps.append({
                "turn": turn_num,
                "type": "error",
                "thought": thought,
                "error": "No action specified",
            })
            history.append({"response_format": ERR_MSG_NO_ACTION})
            if consecutive_errors >= 3:
                return _build_result(None, steps, turn_num, "consecutive_errors")
            continue

        action_projection = tool_function_projection.get(action, [])

        if not action_projection:
            tool_functions_list = ", ".join(list(tool_function_projection.keys()))
            consecutive_errors += 1
            steps.append({
                "turn": turn_num,
                "type": "error",
                "thought": thought,
                "error": f"Invalid action: '{action}'",
            })
            history.append({
                "response_format": ERR_MSG_INVALID_ACTION.format(
                    tool_functions_list=tool_functions_list
                )
            })
            if consecutive_errors >= 3:
                return _build_result(None, steps, turn_num, "consecutive_errors")
            continue

        valid_api_response, api_response = execute_tool_action(
            tool_info, action, action_input
        )

        if not valid_api_response:
            consecutive_errors += 1
            steps.append({
                "turn": turn_num,
                "type": "error",
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "error": f"API call failed: {api_response}",
            })
            history.append({
                "response_format": ERR_MSG_API_FAIL.format(api_response=api_response)
            })
            if consecutive_errors >= 3:
                return _build_result(None, steps, turn_num, "consecutive_errors")
            continue

        steps.append({
            "turn": turn_num,
            "type": "thought_action",
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": api_response,
        })
        history.append({
            "action": action,
            "action_input": json.dumps(action_input),
            "observation": api_response,
        })
        consecutive_errors = 0

    return _build_result(None, steps, max_turns, "max_turns_reached")
