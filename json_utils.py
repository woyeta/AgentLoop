"""
Utilities for extracting and parsing JSON from text outputs.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple, Union


def sanitize_json(raw: str) -> str:
    """
    Cleans raw text to format it as valid JSON.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    if text.startswith("{") and text.count("{") > text.count("}"):
        text += "}"
    return text.strip()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses a JSON object from text, returning None if parsing fails.
    """
    candidate = sanitize_json(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = sanitize_json(match.group(0))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def sanitize_model_response(response: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """
    Parses a response into JSON, returning a success flag and either the parsed object or an error message.
    """
    json_response = extract_json(response)
    if json_response is None:
        return (False, "Invalid Json")
    return (True, json_response)
