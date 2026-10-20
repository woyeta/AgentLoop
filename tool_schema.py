"""
Builds pydantic argument schemas for tool functions from their OpenAPI documentation
and validates model-generated arguments against them.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

logger = logging.getLogger(__name__)

_TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}

_schema_cache: dict = {}


@dataclass
class FunctionSpec:
    """A tool function with its endpoint and argument schema (None if the docs lack one)."""
    name: str
    path: str
    method: str
    model: Optional[type] = None


@dataclass
class ValidationResult:
    """Outcome of validating one set of arguments."""
    ok: bool
    args: Any = None
    message: str = ""
    error_types: list = field(default_factory=list)


def _resolve_ref(schema: dict, components: dict) -> dict:
    ref = schema.get("$ref", "") if isinstance(schema, dict) else ""
    if ref.startswith("#/components/schemas/"):
        return components.get(ref.split("/")[-1], {})
    return schema if isinstance(schema, dict) else {}


def _python_type(schema: dict, components: dict):
    """Maps an OpenAPI schema to a Python type annotation."""
    schema = _resolve_ref(schema or {}, components)
    enum = schema.get("enum")
    if enum and all(isinstance(v, (str, int, float, bool)) for v in enum):
        return Literal[tuple(enum)]
    schema_type = schema.get("type")
    if schema_type == "array":
        return list[_python_type(schema.get("items", {}), components)]
    if schema_type == "object":
        return dict[str, Any]
    return _TYPE_MAP.get(schema_type, Any)


def _collect_fields(operation: dict, components: dict) -> tuple[list, bool]:
    """
    Returns (fields, closed) for an operation, where each field is (name, schema, required).
    `closed` is False when part of the input could not be described, so unknown keys are allowed.
    """
    fields = []
    closed = True

    for param in operation.get("parameters", []) or []:
        param = _resolve_ref(param, components)
        if "name" in param:
            fields.append((param["name"], param.get("schema", {}), bool(param.get("required"))))

    content = (operation.get("requestBody") or {}).get("content", {})
    if content:
        body = content.get("application/json") or next(iter(content.values()), {})
        body_schema = _resolve_ref(body.get("schema", {}), components)
        properties = body_schema.get("properties")
        if isinstance(properties, dict):
            required = set(body_schema.get("required", []) or [])
            for name, prop_schema in properties.items():
                fields.append((name, prop_schema, name in required))
        else:
            closed = False

    return fields, closed


def _build_model(model_name: str, fields: list, closed: bool, components: dict):
    """Creates a pydantic model whose field aliases are the original parameter names."""
    model_fields = {}
    seen = set()
    for i, (name, schema, required) in enumerate(fields):
        if name in seen:
            continue
        seen.add(name)
        annotation = _python_type(schema, components)
        if required:
            model_fields[f"f{i}"] = (annotation, Field(..., alias=name))
        else:
            model_fields[f"f{i}"] = (Optional[annotation], Field(None, alias=name))

    config = ConfigDict(extra="forbid" if closed else "allow", coerce_numbers_to_str=True)
    return create_model(model_name, __config__=config, **model_fields)


def build_function_specs(tool_info: dict) -> dict:
    """Builds a FunctionSpec for every function of a tool, cached by tool name."""
    tool_name = tool_info.get("Name", "")
    cache_key = (tool_name, hash(tool_info.get("Documentation", "")))
    if cache_key in _schema_cache:
        return _schema_cache[cache_key]

    try:
        documentation = json.loads(tool_info.get("Documentation", "") or "{}")
    except json.JSONDecodeError:
        documentation = {}
    paths = documentation.get("paths", {}) or {}
    components = (documentation.get("components", {}) or {}).get("schemas", {}) or {}

    specs = {}
    for func_name, projection in (tool_info.get("Function_Projection", {}) or {}).items():
        path, method = projection[0], projection[1].lower()
        operation = (paths.get(path, {}) or {}).get(method)
        model = None
        if isinstance(operation, dict):
            try:
                fields, closed = _collect_fields(operation, components)
                model = _build_model(f"{func_name}Args", fields, closed, components)
            except Exception as err:
                logger.warning(f"Could not build schema for {tool_name}.{func_name}: {err}")
        specs[func_name] = FunctionSpec(func_name, path, method, model)

    _schema_cache[cache_key] = specs
    return specs


def _format_errors(err: ValidationError) -> tuple[str, list]:
    """Turns a pydantic error into a short message the model can act on."""
    messages, types = [], []
    for e in err.errors():
        loc = ".".join(str(x) for x in e.get("loc", ())) or "action_input"
        error_type = e.get("type", "")
        types.append(error_type)
        if error_type == "missing":
            messages.append(f"missing required field '{loc}'")
        elif error_type == "extra_forbidden":
            messages.append(f"unknown field '{loc}'")
        else:
            messages.append(f"'{loc}': {e.get('msg', 'invalid value')}")
    return "; ".join(messages), types


def validate_arguments(spec: FunctionSpec, action_input) -> ValidationResult:
    """Validates arguments against a function's schema."""
    if isinstance(action_input, str):
        try:
            action_input = json.loads(action_input) if action_input.strip() else {}
        except json.JSONDecodeError:
            return ValidationResult(False, None, "action_input must be a JSON object", ["not_object"])
    if not isinstance(action_input, dict):
        return ValidationResult(False, None, "action_input must be a JSON object", ["not_object"])

    # Models (and the reference data) often send null or "" for optional fields they skip.
    action_input = {k: v for k, v in action_input.items() if v is not None and v != ""}

    if spec.model is None:
        return ValidationResult(True, action_input)

    try:
        validated = spec.model.model_validate(action_input)
    except ValidationError as err:
        message, types = _format_errors(err)
        return ValidationResult(False, action_input, message, types)

    return ValidationResult(True, validated.model_dump(by_alias=True, exclude_unset=True))


def _normalize(value):
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if v is not None and v != ""}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_args(spec: Optional[FunctionSpec], action_input) -> dict:
    """
    Normalizes arguments for comparison: parsed, coerced through the schema when valid,
    whitespace-collapsed and case-folded strings, and empty values dropped.
    """
    if spec is not None:
        result = validate_arguments(spec, action_input)
        if result.ok:
            return _normalize(result.args)
        if isinstance(result.args, dict):
            return _normalize(result.args)
        return {}
    if isinstance(action_input, str):
        try:
            action_input = json.loads(action_input)
        except json.JSONDecodeError:
            return {}
    return _normalize(action_input) if isinstance(action_input, dict) else {}


def canonical_key(action: str, args: dict) -> str:
    """A stable string for comparing or deduplicating calls."""
    return f"{action}:{json.dumps(args, sort_keys=True, default=str)}"


def key_f1(pred: dict, gold: dict) -> float:
    """F1 over (key, value) pairs of two canonical argument dicts."""
    pred_items = {(k, json.dumps(v, sort_keys=True, default=str)) for k, v in pred.items()}
    gold_items = {(k, json.dumps(v, sort_keys=True, default=str)) for k, v in gold.items()}
    if not pred_items and not gold_items:
        return 1.0
    overlap = len(pred_items & gold_items)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_items)
    recall = overlap / len(gold_items)
    return 2 * precision * recall / (precision + recall)
