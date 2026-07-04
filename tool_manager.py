"""
Handles tool discovery, validation, and simulated API executions.
"""

import json
import logging
import os
import requests
from google import genai
from prompts import DATA_DIR, GEMINI_MODEL, MAX_MOCK_RESPONSE_LENGTH, MOCK_API_PROMPT

logger = logging.getLogger(__name__)

_public_apis_cache = None
_train_data_cache = None


def _load_public_apis() -> dict:
    """Loads and caches public API details."""
    global _public_apis_cache
    if _public_apis_cache is None:
        path = os.path.join(DATA_DIR, "public_apis.json")
        with open(path, "r", encoding="utf-8") as f:
            _public_apis_cache = json.load(f)
    return _public_apis_cache


def _load_train_data() -> list:
    """Loads and caches tool specifications."""
    global _train_data_cache
    if _train_data_cache is None:
        path = os.path.join(DATA_DIR, "train_data.json")
        with open(path, "r", encoding="utf-8") as f:
            _train_data_cache = json.load(f)
    return _train_data_cache


def check_auth_requirement(api_name: str) -> str:
    """Checks if an API requires authentication."""
    public_apis = _load_public_apis()
    for entry in public_apis.get("entries", []):
        if entry.get("API") == api_name:
            return entry.get("Auth", "")
    return "Not Found"


def get_api_information(api_name: str) -> dict | None:
    """Retrieves tool specifications for an API."""
    train_data = _load_train_data()
    for tool in train_data:
        if tool.get("Name") == api_name:
            return tool
    return None


def setup_tool(api_name: str) -> tuple[bool, dict | str]:
    """Validates API requirements and retrieves tool information."""
    auth_req = check_auth_requirement(api_name)
    if auth_req == "Not Found":
        return (False, f"API '{api_name}' not found in public APIs dataset.")
    if auth_req != "":
        return (
            False,
            f"API '{api_name}' requires authentication ('{auth_req}'). Only unauthenticated tools are supported.",
        )

    tool_info = get_api_information(api_name)
    if not tool_info:
        return (
            False,
            f"Tool specifications for '{api_name}' not found in training data.",
        )

    return (True, tool_info)


def get_available_tools() -> list[str]:
    """Returns tools that do not require authentication."""
    public_apis = _load_public_apis()
    train_data = _load_train_data()

    no_auth_apis = {
        entry["API"]
        for entry in public_apis.get("entries", [])
        if entry.get("Auth", "") == ""
    }

    available_in_train = {tool["Name"] for tool in train_data if "Name" in tool}

    usable = sorted(no_auth_apis & available_in_train)
    return usable


def call_real_api(
    url: str, endpoint: str, method: str, inputs: dict
) -> tuple[bool, str]:
    """Sends an HTTP request to an API endpoint."""
    method = method.upper()
    call_url = url.rstrip("/") + endpoint
    try:
        if method in ["GET", "DELETE"]:
            response = requests.request(
                method=method, url=call_url, params=inputs, timeout=10
            )
        else:
            response = requests.request(
                method=method, url=call_url, json=inputs, timeout=10
            )
        return (True, response.text)
    except Exception as err:
        logger.warning(f"Real API call failed: {err}")
        return (False, str(err))


def _get_tool_data(tool_info: dict, key: str) -> str:
    val = tool_info.get(key, "")
    if isinstance(val, str):
        return val.strip()
    return str(val)


def _get_tool_functions_descriptions(tool_info: dict, key: str) -> dict:
    return tool_info.get(key, {})


def _create_function_details(
    name: str, description: str, endpoint: str, method: str
) -> str:
    return json.dumps(
        {
            "function": name,
            "description": description,
            "endpoint": endpoint,
            "method": method,
        }
    )


def call_mock_api(
    tool_info: dict, endpoint: str, method: str, data: dict
) -> tuple[bool, str]:
    """Simulates an API response using the AI model."""
    function_descriptions = _get_tool_functions_descriptions(
        tool_info, "Function_Description"
    )

    tool_function_information = [
        _create_function_details(
            func,
            _get_tool_data(function_descriptions, func),
            func_info[0],
            func_info[1],
        )
        for func, func_info in tool_info.get("Function_Projection", {}).items()
    ]

    api_prompt = MOCK_API_PROMPT.format(
        NAME=_get_tool_data(tool_info, "Name"),
        DESCRIPTION=_get_tool_data(tool_info, "Description"),
        INTRODUCTION=_get_tool_data(tool_info, "Introduction"),
        FUNCTIONS=_get_tool_data(tool_info, "Functions"),
        DOCUMENTATION=_get_tool_data(tool_info, "NLDocumentation"),
        FUNCTIONS_DESCRIPTIONS="\n".join(tool_function_information),
        REQUEST_ENDPOINT=endpoint,
        REQUEST_METHOD=method,
        REQUEST_INPUTS=json.dumps(data) if isinstance(data, dict) else data,
    )

    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return (False, "GEMINI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=api_prompt
        )
        return (True, response.text[:MAX_MOCK_RESPONSE_LENGTH])
    except Exception as err:
        logger.error(f"Gemini mock API failed: {err}")
        return (False, str(err))


def execute_tool_action(
    tool_info: dict, action: str, action_input: dict
) -> tuple[bool, str]:
    """Executes a tool action."""
    action_projection = tool_info.get("Function_Projection", {}).get(action, [])
    if not action_projection:
        return (False, f"Action '{action}' not found in tool")

    endpoint, method = action_projection

    logger.info(f"Using Gemini mock API for {action}")
    return call_mock_api(tool_info, endpoint, method, action_input)
