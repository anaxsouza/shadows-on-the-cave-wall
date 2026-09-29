"""
GENIA dataset loader for Aunderline/genia dataset.

Handles the structured entity format specific to the Aunderline/genia dataset,
converting from structured entity annotations to standardized NERExample format.
"""

from typing import Optional, List, Dict, Any
from datasets import Dataset
import logging

from ..base.loader import BaseDatasetLoader
from ....shared.model.config_manager import get_config_manager
from ....shared.data.data_types import NERExample, Entity
from ....shared.data.bio_extraction import extract_entities_from_bio, calculate_nesting_depth_by_words
from ....shared.data.nesting_utils import detect_nesting

logger = logging.getLogger(__name__)


class GENIALoader(BaseDatasetLoader):
    """
    Loader for the Aunderline/genia dataset.
    
    Handles the specific format of this dataset which uses structured
    entity annotations instead of BIO tags.
    """
    
    # GENIA entity types based on biomedical domain
    ENTITY_TYPES = ['protein', 'DNA', 'RNA', 'cell_line', 'cell_type']
    
    # Type mappings for normalization
    TYPE_MAPPINGS = {
        'protein': 'PROTEIN',
        'DNA': 'DNA', 
        'RNA': 'RNA',
        'cell_line': 'CELL_LINE',
        'cell_type': 'CELL_TYPE'
    }
    
    def get_dataset_name(self) -> str:
        """Return the dataset name this loader handles."""
        return 'genia'
    
    def get_huggingface_name(self) -> str:
        """Return the HuggingFace dataset path."""
        return 'Aunderline/genia'
    
    def get_huggingface_config(self) -> Optional[str]:
        """Return the HuggingFace dataset configuration."""
        return None  # No specific config needed
    
    def get_huggingface_revision(self) -> Optional[str]:
        """Revisão declarada em configs/config.yaml, datasets.genia.revision."""
        config = get_config_manager()
        return config.get_dataset_config('genia').get('revision', None)

    def get_entity_types(self) -> List[str]:
        """Return the entity types supported by this dataset."""
        return list(self.TYPE_MAPPINGS.values())
    
    def convert_to_examples(self, dataset: Dataset, max_examples: Optional[int] = None) -> List[NERExample]:
        """
        Convert GENIA dataset to NERExample objects.
        
        The Aunderline/genia dataset has the following structure:
        - tokens: List of tokens
        - entities: List of structured entities with start/end indices and types
        - pos: Part-of-speech tags (not used for NER)
        - relations: Relations data (not used for NER)
        - org_id: Document ID
        
        Args:
            dataset: Raw HuggingFace dataset
            max_examples: Maximum number of examples to process
            
        Returns:
            List of NERExample objects
        """
        examples = []
        
        # Extract label info first
        self.extract_label_info(dataset)
        
        # Limit dataset if requested
        dataset_subset = dataset
        if max_examples:
            dataset_subset = dataset.select(range(min(max_examples, len(dataset))))
        
        logger.info(f"Converting {len(dataset_subset)} GENIA examples...")
        
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
        
        # Apply Universal Nesting Detection
        self._detect_and_mark_nesting(examples)
        
        logger.info(f"Successfully converted {len(examples)} GENIA examples")
        return examples
    
    def _convert_single_item(self, item: Dict[str, Any], example_id: int) -> Optional[NERExample]:
        """
        Convert a single GENIA item to NERExample.
        
        Args:
            item: Single item from the dataset
            example_id: Unique identifier for this example
            
        Returns:
            NERExample object or None if conversion fails
        """
        # Extract tokens
        tokens = item.get('tokens', [])
        if not tokens:
            logger.warning(f"Example {example_id}: No tokens found")
            return None
        
        # Create text from tokens
        text = ' '.join(tokens)
        
        # Extract and convert entities
        raw_entities = item.get('entities', [])
        entities = self._convert_structured_entities(raw_entities, tokens, text)
        
        # Create BIO labels for compatibility
        bio_labels = self._create_bio_labels(tokens, entities)
        
        # Re-extract entities using proven original logic for consistency
        # CRITICAL FIX: Do NOT overwrite entities with BIO extraction.
        # BIO is flat by definition, so re-extracting from it destroys nested entities.
        # We need to preserve the original structured entities for Nested Recall analysis.
        # entities = extract_entities_from_bio(tokens, bio_labels)
        
        
        # Create NERExample
        example = NERExample(
            id=str(example_id),
            text=text,
            tokens=tokens,
            labels=bio_labels,
            entities=entities
        )
        
        return example
    
    def _convert_structured_entities(self, raw_entities: List[Dict[str, Any]], 
                                   tokens: List[str], text: str) -> List[Entity]:
        """
        Convert structured entities to Entity objects.
        
        Args:
            raw_entities: Raw entity annotations from dataset
            tokens: List of tokens
            text: Reconstructed text
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        for entity_data in raw_entities:
            try:
                # Extract entity information
                start_idx = entity_data.get('start', -1)
                end_idx = entity_data.get('end', -1)
                entity_type = entity_data.get('type', 'UNKNOWN')
                
                if start_idx == -1 or end_idx == -1:
                    logger.warning(f"Invalid entity indices: start={start_idx}, end={end_idx}")
                    continue
                
                # Convert token indices to character indices
                char_start, char_end = self._token_indices_to_char_indices(
                    start_idx, end_idx, tokens
                )
                
                if char_start == -1 or char_end == -1:
                    logger.warning(f"Could not convert token indices to char indices")
                    continue
                
                # Extract entity text
                entity_text = text[char_start:char_end].strip()
                if not entity_text:
                    logger.warning(f"Empty entity text at indices {char_start}:{char_end}")
                    continue
                
                # Normalize entity type
                normalized_type = self.TYPE_MAPPINGS.get(entity_type.lower(), entity_type.upper())
                
                # Create Entity
                entity = Entity(
                    text=entity_text,
                    label=normalized_type,
                    start=char_start,
                    end=char_end,
                    tokens=tokens[start_idx:end_idx],
                    nesting_depth=calculate_nesting_depth_by_words(entity_text)
                )
                
                entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Error processing entity {entity_data}: {e}")
                continue
        
        # Sort entities by start position for consistency
        entities.sort(key=lambda x: x.start)
        
        return entities
    
    def _token_indices_to_char_indices(self, start_token: int, end_token: int, 
                                     tokens: List[str]) -> tuple[int, int]:
        """
        Convert token indices to character indices in the reconstructed text.
        
        Args:
            start_token: Starting token index
            end_token: Ending token index (exclusive)
            tokens: List of tokens
            
        Returns:
            Tuple of (char_start, char_end) or (-1, -1) if invalid
        """
        if start_token < 0 or end_token > len(tokens) or start_token >= end_token:
            return -1, -1
        
        # Calculate character positions
        # Account for spaces between tokens
        # Calculate character positions
        # Account for spaces between tokens
        char_start = sum(len(token) + 1 for token in tokens[:start_token])  # +1 for space
            
        # Calculate end position
        entity_tokens = tokens[start_token:end_token]
        entity_length = sum(len(token) for token in entity_tokens) + len(entity_tokens) - 1
        char_end = char_start + entity_length
        
        return char_start, char_end
    
    def _create_bio_labels(self, tokens: List[str], entities: List[Entity]) -> List[str]:
        """
        Create BIO labels for compatibility with existing code.
        
        Args:
            tokens: List of tokens
            entities: List of entities
            
        Returns:
            List of BIO labels
        """
        labels = ['O'] * len(tokens)
        
        for entity in entities:
            # Find which tokens this entity spans
            token_start, token_end = self._char_indices_to_token_indices(
                entity.start, entity.end, tokens
            )
            
            if token_start != -1 and token_end != -1:
                # Set BIO labels
                labels[token_start] = f'B-{entity.label}'
                for i in range(token_start + 1, token_end):
                    labels[i] = f'I-{entity.label}'
        
        return labels
    
    def _char_indices_to_token_indices(self, char_start: int, char_end: int, 
                                     tokens: List[str]) -> tuple[int, int]:
        """
        Convert character indices back to token indices.
        
        Args:
            char_start: Starting character index
            char_end: Ending character index
            tokens: List of tokens
            
        Returns:
            Tuple of (token_start, token_end) or (-1, -1) if invalid
        """
        current_pos = 0
        token_start = -1
        token_end = -1
        
        for i, token in enumerate(tokens):
            token_start_pos = current_pos
            token_end_pos = current_pos + len(token)
            
            # Check if this token overlaps with the entity
            if token_start == -1 and token_end_pos > char_start:
                token_start = i
            
            if token_start != -1 and token_start_pos >= char_end:
                token_end = i
                break
                
            current_pos = token_end_pos + 1  # +1 for space
        
        # If we didn't find token_end, use the last token
        if token_start != -1 and token_end == -1:
            token_end = len(tokens)
        
        return token_start if token_start != -1 else -1, token_end if token_end != -1 else -1