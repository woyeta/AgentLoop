from tool_schema import build_function_specs, validate_arguments, canonical_args, key_f1


def test_valid_arguments_pass(httpbin):
    spec = build_function_specs(httpbin)["sendHttpRequest"]
    result = validate_arguments(spec, {"method": "POST", "url": "https://httpbin.org/post"})
    assert result.ok
    assert result.args == {"method": "POST", "url": "https://httpbin.org/post"}


def test_json_string_arguments_are_parsed(httpbin):
    spec = build_function_specs(httpbin)["getClientRequestData"]
    assert validate_arguments(spec, '{"url": "https://httpbin.org/get"}').ok


def test_missing_required_field(httpbin):
    spec = build_function_specs(httpbin)["sendHttpRequest"]
    result = validate_arguments(spec, {"method": "POST"})
    assert not result.ok
    assert "missing required field 'url'" in result.message
    assert "missing" in result.error_types


def test_enum_value_rejected(httpbin):
    spec = build_function_specs(httpbin)["sendHttpRequest"]
    result = validate_arguments(spec, {"method": "FETCH", "url": "x"})
    assert not result.ok
    assert "'method'" in result.message


def test_unknown_field_rejected_when_schema_is_closed(httpbin):
    spec = build_function_specs(httpbin)["getClientRequestData"]
    result = validate_arguments(spec, {"url": "x", "verbose": True})
    assert not result.ok
    assert "unknown field 'verbose'" in result.message


def test_open_request_body_allows_extra_fields(httpbin):
    # sendHttpRequest's requestBody is an untyped object, so extra keys cannot be judged.
    spec = build_function_specs(httpbin)["sendHttpRequest"]
    assert validate_arguments(spec, {"method": "GET", "url": "x", "verbose": True}).ok


def test_integer_coercion(httpbin):
    spec = build_function_specs(httpbin)["simulateStatusCode"]
    result = validate_arguments(spec, {"url": "https://httpbin.org/status", "statusCode": "404"})
    assert result.ok
    assert result.args["statusCode"] == 404


def test_non_object_arguments_rejected(httpbin):
    spec = build_function_specs(httpbin)["sendHttpRequest"]
    assert not validate_arguments(spec, "not json").ok
    assert not validate_arguments(spec, [1, 2]).ok


def test_schema_coverage(train_data):
    specs = [s for tool in train_data for s in build_function_specs(tool).values()]
    with_schema = sum(1 for s in specs if s.model is not None)
    assert with_schema / len(specs) > 0.9


def test_canonical_args_and_f1(httpbin):
    spec = build_function_specs(httpbin)["simulateStatusCode"]
    a = canonical_args(spec, {"url": " HTTPS://httpbin.org/status ", "statusCode": "404"})
    b = canonical_args(spec, '{"url": "https://httpbin.org/status", "statusCode": 404}')
    assert a == b
    assert key_f1(a, b) == 1.0
    assert key_f1(a, {"url": a["url"]}) == 2 / 3
    assert key_f1({}, {}) == 1.0
