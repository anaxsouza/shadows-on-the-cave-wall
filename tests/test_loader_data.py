"""Tests for dataset loaders (CONLL, GENIA) using mock data."""

import pytest
from src.core.loaders.conll.loader import CONLLLoader
from src.core.loaders.biomedical.genia import GENIALoader


class TestCONLLLoaderContract:
    def test_dataset_name(self):
        loader = CONLLLoader()
        assert loader.get_dataset_name() == "conll2003"

    def test_entity_types(self):
        loader = CONLLLoader()
        types = loader.get_entity_types()
        assert "PER" in types
        assert "ORG" in types
        assert "LOC" in types
        assert "MISC" in types
        assert len(types) == 4

    def test_huggingface_name(self):
        loader = CONLLLoader()
        name = loader.get_huggingface_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_label_list(self):
        loader = CONLLLoader()
        labels = loader.get_label_list()
        assert "O" in labels
        assert "B-PER" in labels
        assert "I-PER" in labels

    def test_extract_label_info_sets_hardcoded_labels(self):
        loader = CONLLLoader()
        loader.extract_label_info(None)
        assert loader.label_list == CONLLLoader.BIO_LABELS
        assert loader.label2id["B-PER"] == 1
        assert loader.id2label[1] == "B-PER"


class TestGENIALoaderContract:
    def test_dataset_name(self):
        loader = GENIALoader()
        assert loader.get_dataset_name() == "genia"

    def test_entity_types(self):
        loader = GENIALoader()
        types = loader.get_entity_types()
        assert "PROTEIN" in types
        assert "DNA" in types
        assert "RNA" in types
        assert "CELL_LINE" in types
        assert "CELL_TYPE" in types
        assert len(types) == 5

    def test_type_mappings(self):
        assert GENIALoader.TYPE_MAPPINGS["protein"] == "PROTEIN"
        assert GENIALoader.TYPE_MAPPINGS["DNA"] == "DNA"
        assert GENIALoader.TYPE_MAPPINGS["cell_line"] == "CELL_LINE"

    def test_token_to_char_conversion(self):
        loader = GENIALoader()
        tokens = ["Apple", "bought", "Google"]
        char_start, char_end = loader._token_indices_to_char_indices(0, 1, tokens)
        assert char_start == 0
        assert char_end == 5

        char_start, char_end = loader._token_indices_to_char_indices(2, 3, tokens)
        assert char_start == 13
        assert char_end == 19

    def test_token_to_char_invalid_range(self):
        loader = GENIALoader()
        tokens = ["a", "b"]
        result = loader._token_indices_to_char_indices(-1, 1, tokens)
        assert result == (-1, -1)
        result = loader._token_indices_to_char_indices(0, 5, tokens)
        assert result == (-1, -1)

    def test_char_to_token_conversion(self):
        loader = GENIALoader()
        tokens = ["Apple", "bought", "Google", "in", "California"]
        tok_s, tok_e = loader._char_indices_to_token_indices(0, 5, tokens)
        assert tok_s == 0
        assert tok_e == 1

        tok_s, tok_e = loader._char_indices_to_token_indices(13, 19, tokens)
        assert tok_s == 2
        assert tok_e == 3

    def test_convert_structured_entities(self):
        loader = GENIALoader()
        tokens = ["IL-2", "gene", "expression"]
        text = "IL-2 gene expression"
        raw = [{"start": 0, "end": 1, "type": "protein"},
               {"start": 1, "end": 2, "type": "DNA"}]
        entities = loader._convert_structured_entities(raw, tokens, text)
        assert len(entities) == 2
        assert entities[0].label == "PROTEIN"
        assert entities[1].label == "DNA"
        assert entities[0].text == "IL-2"
        assert entities[1].text == "gene"

    def test_convert_structured_entities_handles_invalid(self):
        loader = GENIALoader()
        tokens = ["test"]
        text = "test"
        raw = [{"start": -1, "end": 0, "type": "protein"}]
        entities = loader._convert_structured_entities(raw, tokens, text)
        assert entities == []

    def test_create_bio_labels(self):
        loader = GENIALoader()
        from src.shared.data.data_types import Entity
        tokens = ["IL-2", "gene", "expression"]
        entities = [
            Entity(text="IL-2", label="PROTEIN", start=0, end=4,
                   tokens=["IL-2"], nesting_depth=0),
            Entity(text="gene", label="DNA", start=5, end=9,
                   tokens=["gene"], nesting_depth=0),
        ]
        labels = loader._create_bio_labels(tokens, entities)
        assert labels == ["B-PROTEIN", "B-DNA", "O"]

    def test_convert_single_item(self):
        loader = GENIALoader()
        item = {
            "tokens": ["IL-2", "gene", "expression"],
            "entities": [{"start": 0, "end": 1, "type": "protein"}],
        }
        example = loader._convert_single_item(item, 0)
        assert example is not None
        assert example.id == "0"
        assert example.text == "IL-2 gene expression"
        assert len(example.entities) == 1
        assert example.entities[0].label == "PROTEIN"

    def test_convert_single_item_missing_tokens(self):
        loader = GENIALoader()
        assert loader._convert_single_item({}, 0) is None
