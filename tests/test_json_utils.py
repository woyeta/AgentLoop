from json_utils import extract_json


def test_urls_inside_strings_survive():
    text = '{"action": "get", "action_input": {"url": "https://httpbin.org/get"}}'
    assert extract_json(text)["action_input"]["url"] == "https://httpbin.org/get"


def test_comments_outside_strings_are_removed():
    text = '{\n  "action": "get", // pick the tool\n  "action_input": {}\n}'
    assert extract_json(text) == {"action": "get", "action_input": {}}


def test_trailing_text_after_object():
    text = '{"thought": "t", "final_answer": "a"}\nObservation: {"fake": true}'
    assert extract_json(text) == {"thought": "t", "final_answer": "a"}


def test_fences_and_trailing_commas():
    assert extract_json('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}


def test_missing_closing_brace():
    assert extract_json('{"action": "get", "action_input": {"q": "x"}') == {"action": "get", "action_input": {"q": "x"}}


def test_not_json():
    assert extract_json("no json here") is None
