import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def train_data():
    with open(os.path.join(ROOT, "data", "train_data.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def httpbin(train_data):
    return next(t for t in train_data if t["Name"] == "Httpbin")
