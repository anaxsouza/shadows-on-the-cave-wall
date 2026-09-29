"""Shared fixtures for SENTINEL test suite."""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from src.shared.data.data_types import Entity, NERExample


@pytest.fixture
def tmp_results_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def config_path():
    return "configs/config.yaml"


@pytest.fixture
def sample_entities() -> List[Entity]:
    return [
        Entity(text="Apple", label="ORG", start=0, end=5, tokens=["Apple"]),
        Entity(text="Google", label="ORG", start=10, end=16, tokens=["Google"]),
        Entity(text="California", label="LOC", start=20, end=30, tokens=["California"]),
    ]


@pytest.fixture
def sample_example(sample_entities) -> NERExample:
    return NERExample(
        id="0",
        text="Apple bought Google in California",
        tokens=["Apple", "bought", "Google", "in", "California"],
        labels=["B-ORG", "O", "B-ORG", "O", "B-LOC"],
        entities=sample_entities,
    )


@pytest.fixture
def sample_results_json(tmp_results_dir) -> Path:
    data = {
        "model_key": "bert-large",
        "dataset": "conll2003",
        "metrics": {
            "f1": 0.95,
            "precision": 0.94,
            "recall": 0.96,
        },
        "config": {"seed": 42, "max_samples": 100},
    }
    p = tmp_results_dir / "results.json"
    p.write_text(json.dumps(data))
    return p
