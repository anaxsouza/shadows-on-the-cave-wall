"""Tests for NER data types (NERExample, Entity)."""

import pytest
from src.shared.data.data_types import Entity, NERExample


class TestEntity:
    def test_construction(self):
        e = Entity(text="Apple", label="ORG", start=0, end=5, tokens=["Apple"])
        assert e.text == "Apple"
        assert e.label == "ORG"
        assert e.start == 0
        assert e.end == 5
        assert e.tokens == ["Apple"]
        assert e.nesting_depth == 0
        assert not e.is_nested

    def test_construction_with_nesting(self):
        e = Entity(text="protein", label="PROTEIN", start=0, end=7,
                    tokens=["protein"], nesting_depth=1, is_nested=True)
        assert e.nesting_depth == 1
        assert e.is_nested

    def test_empty_tokens(self):
        e = Entity(text="Hello", label="MISC", start=0, end=5, tokens=[])
        assert e.tokens == []


class TestNERExample:
    def test_construction(self, sample_example):
        assert sample_example.id == "0"
        assert sample_example.text == "Apple bought Google in California"
        assert sample_example.tokens == ["Apple", "bought", "Google", "in", "California"]
        assert sample_example.labels == ["B-ORG", "O", "B-ORG", "O", "B-LOC"]
        assert len(sample_example.entities) == 3
        assert sample_example.metadata is None

    def test_construction_with_metadata(self):
        example = NERExample(
            id="1", text="test", tokens=["test"], labels=["O"], entities=[],
            metadata={"source": "test", "split": "train"},
        )
        assert example.metadata == {"source": "test", "split": "train"}

    def test_empty_entities(self):
        example = NERExample(
            id="2", text="No entities here", tokens=["No", "entities", "here"],
            labels=["O", "O", "O"], entities=[],
        )
        assert example.entities == []

    def test_entity_equality(self, sample_example):
        entities = sample_example.entities
        e1 = entities[0]
        assert e1.text == "Apple"
        assert e1.label == "ORG"
        assert e1.start == 0

    def test_multiple_entity_types(self):
        entities = [
            Entity(text="John", label="PER", start=0, end=4, tokens=["John"]),
            Entity(text="New York", label="LOC", start=10, end=18, tokens=["New", "York"]),
            Entity(text="Microsoft", label="ORG", start=20, end=29, tokens=["Microsoft"]),
        ]
        example = NERExample(
            id="3", text="John lives in New York and works at Microsoft",
            tokens=["John", "lives", "in", "New", "York", "and", "works", "at", "Microsoft"],
            labels=["B-PER", "O", "O", "B-LOC", "I-LOC", "O", "O", "O", "B-ORG"],
            entities=entities,
        )
        assert len(example.entities) == 3
        labels = {e.label for e in example.entities}
        assert labels == {"PER", "LOC", "ORG"}
