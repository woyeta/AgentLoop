"""
Templates, error messages, and constants for model interaction.
"""

import os

SYSTEM_PROMPT = """You are an intelligent AI agent that answers questions by reasoning step-by-step.

You have access to the following tools:
{tool_docs}

Use this STRICT format for reasoning — output a JSON block:
{{"thought": "...", "action": "...", "action_input": {{"key": "value"}}}}

If you have enough information, output a final JSON block:
{{"thought": "...", "final_answer": "..."}}

RULES:
- STOP generating immediately after closing the JSON block.
- action_input MUST be a valid JSON object (double-quoted keys, no trailing commas).
- Only call tools listed above; never fabricate tool names or results.
- Always output valid JSON — no trailing commas, no single quotes.
- Use the final_answer JSON format (with \"thought\" and \"final_answer\" keys, omitting \"action\" and \"action_input\") only when you have gathered enough information to fully answer the user's question."""

MOCK_API_PROMPT = """You are a mock API server. When given an API specification, endpoint, HTTP method,
and input parameters, you will simulate a realistic API response exactly as the
real server would return it.

Follow these rules strictly:
1. Return ONLY the raw response body (e.g. JSON, plain text) — no explanation,
   no markdown, no commentary.
2. Match the exact output structure defined in the API spec.
3. Generate realistic, varied, and plausible fake data (e.g. real-looking URLs,
   IDs, timestamps).
4. Respect all input parameters (e.g. if count=3, return exactly 3 items).
5. If an endpoint or method is not defined in the spec, return an appropriate
   HTTP error response in JSON: {{"error": "404 Not Found", "message": "..."}}.
6. If required parameters are missing or invalid, return a 400 error response.

---
API SPECIFICATION:

Name: {NAME}
Description: {DESCRIPTION}
Introduction: {INTRODUCTION}

Functions:
{FUNCTIONS}

Documentation:
{DOCUMENTATION}

Function Descriptions:
{FUNCTIONS_DESCRIPTIONS}

---

REQUEST:
Endpoint: {REQUEST_ENDPOINT}
Method: {REQUEST_METHOD}
Inputs: {REQUEST_INPUTS}

RESPONSE:"""

ERR_MSG_INVALID_JSON = (
    'ERROR: Invalid JSON response was generated. Use this STRICT format for reasoning '
    '— output a JSON block:\n'
    '{{"thought": "...", "action": "...", "action_input": {{"key": "value"}}}}\n\n'
    'If you have enough information, output a final JSON block:\n'
    '{{"thought": "...", "final_answer": "..."}}'
)

ERR_MSG_NO_ACTION = (
    'ERROR: No action was found. Use this STRICT format for reasoning '
    '— output a JSON block:\n'
    '{{"thought": "...", "action": "...", "action_input": {{"key": "value"}}}}\n\n'
    'If you have enough information, output a final JSON block:\n'
    '{{"thought": "...", "final_answer": "..."}}'
)

ERR_MSG_INVALID_ACTION = (
    "ERROR: Invalid action. The action value must exactly match a name from the "
    "following [{tool_functions_list}]"
)

ERR_MSG_API_FAIL = (
    "ERROR: API call failed or returned an invalid response. Details: {api_response}"
)

BASE_MODEL = "google/gemma-4-E2B-it"
ADAPTER_REPO = "woyeta/gemma-4-ToolAlpaca-FineTune-C"
GEMINI_MODEL = "gemini-3.1-flash-lite"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAX_MOCK_RESPONSE_LENGTH = 500
