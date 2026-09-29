"""
CoNLL-2003 dataset loader.

Handles the standard BIO tagging format used by CoNLL-2003 dataset.
"""

from typing import Optional, List, Dict, Any, Optional
from datasets import Dataset
import logging

from ..base.loader import BaseDatasetLoader
from ....shared.data.data_types import NERExample, Entity
from ....shared.data.bio_extraction import extract_entities_from_bio
from ....shared.data.nesting_utils import detect_nesting
from ....shared.model.config_manager import get_config_manager

logger = logging.getLogger(__name__)


class CONLLLoader(BaseDatasetLoader):
    """
    Loader for CoNLL-2003 dataset.
    
    Handles the standard BIO tagging format with predefined entity types.
    Loads dataset path from config.yaml - no hardcoded values.
    """
    
    # CoNLL-2003 entity types
    ENTITY_TYPES = ['PER', 'ORG', 'LOC', 'MISC']
    
    # BIO labels for CoNLL-2003 (standard ordering)
    # Note: These are the label VALUES, not the dataset PATH (which comes from config)
    BIO_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    
    def get_dataset_name(self) -> str:
        """Return the dataset name this loader handles."""
        return 'conll2003'
    
    def get_huggingface_name(self) -> str:
        """Return the HuggingFace dataset path from config."""
        config = get_config_manager()
        dataset_config = config.get_dataset_config('conll2003')
        return dataset_config['name']
    
    def get_huggingface_config(self) -> Optional[str]:
        """Return the HuggingFace dataset configuration from config."""
        config = get_config_manager()
        dataset_config = config.get_dataset_config('conll2003')
        return dataset_config.get('config', None)
    
    def get_huggingface_revision(self) -> Optional[str]:
        """Revisão declarada em configs/config.yaml, datasets.conll2003.revision."""
        config = get_config_manager()
        return config.get_dataset_config('conll2003').get('revision', None)

    def get_entity_types(self) -> List[str]:
        """Return the entity types supported by this dataset."""
        return self.ENTITY_TYPES.copy()
    
    def extract_label_info(self, dataset: Dataset) -> None:
        """
        Extract label information from CoNLL-2003 dataset.
        
        ALWAYS uses hardcoded BIO labels for consistency, as tner/conll2003
        uses integer IDs without label names in features.
        
        Args:
            dataset: Raw dataset to analyze
        """
        # Always use hardcoded labels for CoNLL-2003 to ensure correct BIO tag mapping
        # Do NOT call parent class - it extracts ['0', '1', '2', ...] which is wrong
        logger.info("Using hardcoded BIO labels for CoNLL-2003 (tner/conll2003 format)")
        self.label_list = self.BIO_LABELS.copy()
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
    
    def convert_to_examples(self, dataset: Dataset, max_examples: Optional[int] = None) -> List[NERExample]:
        """
        Convert CoNLL-2003 dataset to NERExample objects.
        
        Args:
            dataset: Raw HuggingFace dataset
            max_examples: Maximum number of examples to process
            
        Returns:
            List of NERExample objects
        """
        examples = []
        
        # Extract label info
        self.extract_label_info(dataset)
        
        # Limit dataset if requested
        dataset_subset = dataset
        if max_examples:
            dataset_subset = dataset.select(range(min(max_examples, len(dataset))))
        
        logger.info(f"Converting {len(dataset_subset)} CoNLL-2003 examples...")
        
        for i, item in enumerate(dataset_subset):
            try:
                example = self._convert_single_item(item, i)
                if example and self.validate_example(example):
                    examples.append(example)
                else:
                    logger.warning(f"Skipped invalid example {i}")
                    
            except Exception as e:
                logger.warning(f"Error converting example {i}: {e}")
                continue
        
        logger.info(f"Successfully converted {len(examples)} CoNLL-2003 examples")
        return examples
    
    def _convert_single_item(self, item: Dict[str, Any], example_id: int) -> Optional[NERExample]:
        """
        Convert a single CoNLL-2003 item to NERExample.
        
        Args:
            item: Single item from the dataset
            example_id: Unique identifier for this example
            
        Returns:
            NERExample object or None if conversion fails
        """
        # Extract tokens and labels
        # Support both 'ner_tags' (original conll2003) and 'tags' (tner/conll2003)
        tokens = item.get('tokens', [])
        labels = item.get('ner_tags', item.get('tags', []))
        
        if not tokens or not labels:
            logger.warning(f"Example {example_id}: Missing tokens or labels")
            return None
        
        if len(tokens) != len(labels):
            logger.warning(f"Example {example_id}: Token/label length mismatch")
            return None
        
        # Convert label IDs to strings if needed
        if isinstance(labels[0], int) and self.id2label:
            labels = [self.id2label[label_id] for label_id in labels]
        
        # Extract entities from BIO labels using proven original logic
        entities = extract_entities_from_bio(tokens, labels)
        
        # Create text from tokens
        text = ' '.join(tokens)
        
        # Universal Nesting Detection
        # Apply geometric check (will confirm flatness for CoNLL)
        entities = detect_nesting(entities)
        
        # Create NERExample
        example = NERExample(
            id=str(example_id),
            text=text,
            tokens=tokens,
            labels=labels,
            entities=entities
        )
        
        return example
    
