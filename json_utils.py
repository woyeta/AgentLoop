"""
Utilities for extracting and parsing JSON from text outputs.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple, Union

_decoder = json.JSONDecoder()


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _strip_comments(text: str) -> str:
    """Removes // line comments that are outside of JSON strings (URLs stay intact)."""
    out = []
    in_string = escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
            out.append(ch)
        elif text.startswith("//", i):
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def sanitize_json(raw: str) -> str:
    """
    Cleans raw text to format it as valid JSON.
    """
    text = _strip_fences(raw)
    text = _strip_comments(text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    if text.startswith("{") and text.count("{") > text.count("}"):
        text += "}" * (text.count("{") - text.count("}"))
    return text.strip()


def _first_object(text: str) -> Optional[Any]:
    """Decodes the JSON value starting at the first '{', ignoring any trailing text."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        value, _ = _decoder.raw_decode(text, start)
        return value
    except json.JSONDecodeError:
        return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses a JSON object from text, returning None if parsing fails.

    Tries the text as-is first, then the first complete object (models often keep
    generating after the closing brace), then a sanitized version of both.
    """
    stripped = _strip_fences(text)
    for candidate in (stripped, sanitize_json(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        value = _first_object(candidate)
        if value is not None:
            return value
    return None


def sanitize_model_response(response: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """
    Parses a response into JSON, returning a success flag and either the parsed object or an error message.
    """
    json_response = extract_json(response)
    if json_response is None:
        return (False, "Invalid Json")
    return (True, json_response)
