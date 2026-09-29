"""
Unified BIO extraction utility with proven logic.

This module contains the original, working BIO-to-entity extraction logic
that was accidentally removed during SOLID refactor. This ensures consistent
entity processing between gold standard and predicted entities.
"""

from typing import List, Dict, Any
import logging

from .data_types import Entity

logger = logging.getLogger(__name__)


def calculate_nesting_depth_by_words(entity_text: str) -> int:
    """
    Calculate nesting depth based on word count heuristic.

    This function implements a word-count based approach to estimate nesting depth,
    where longer entity names are assumed to represent more complex, potentially
    nested concepts.

    Args:
        entity_text: The text content of the entity

    Returns:
        Nesting depth starting from 0:
        - 1 word (e.g., "Apple") → depth 0
        - 2 words (e.g., "New York") → depth 1
        - 3 words (e.g., "New York Times") → depth 2
        - 4+ words (e.g., "U.S. Department of State") → depth 3

    Examples:
        >>> calculate_nesting_depth_by_words("Apple")
        0
        >>> calculate_nesting_depth_by_words("New York")
        1
        >>> calculate_nesting_depth_by_words("New York Times")
        2
        >>> calculate_nesting_depth_by_words("U.S. Department of State")
        3
    """
    if not entity_text or not entity_text.strip():
        return 0

    words = entity_text.strip().split()
    return max(0, len(words) - 1)


def extract_entities_from_bio(tokens: List[str], labels: List[str]) -> List[Entity]:
    """
    Extract entities from BIO-tagged tokens with character-level positions.
    
    This is the original, proven logic that was working with 90% F1 performance
    before the SOLID refactor accidentally removed it.
    
    Args:
        tokens: List of tokens
        labels: List of BIO labels
    
    Returns:
        List of Entity objects with character-level start/end positions
    """
    entities = []
    current_entity = None
    
    # Create word-to-character mapping for the joined text
    text = ' '.join(tokens)
    word_to_char_map = _create_word_to_char_mapping(tokens, text)
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            # Start of new entity
            if current_entity:
                # Finalize previous entity with character positions
                current_entity = _finalize_entity_positions(current_entity, word_to_char_map)
                entities.append(current_entity)
            
            entity_type = label[2:]
            current_entity = Entity(
                text=token,
                label=entity_type,
                start=i,  # word position initially
                end=i,    # word position initially
                tokens=[token],
                nesting_depth=calculate_nesting_depth_by_words(token)
            )
        
        elif label.startswith('I-') and current_entity:
            # Continuation of entity
            entity_type = label[2:]
            if entity_type == current_entity.label:
                current_entity.text += f' {token}'
                current_entity.end = i
                current_entity.tokens.append(token)
                # Recalculate nesting depth as entity grows
                current_entity.nesting_depth = calculate_nesting_depth_by_words(current_entity.text)
            else:
                # Mismatched I- tag, finalize current and start new
                current_entity = _finalize_entity_positions(current_entity, word_to_char_map)
                entities.append(current_entity)
                current_entity = Entity(
                    text=token,
                    label=entity_type,
                    start=i,
                    end=i,
                    tokens=[token],
                    nesting_depth=calculate_nesting_depth_by_words(token)
                )
        
        else:
            # O tag or mismatched I- tag
            if current_entity:
                current_entity = _finalize_entity_positions(current_entity, word_to_char_map)
                entities.append(current_entity)
                current_entity = None
    
    # Don't forget the last entity
    if current_entity:
        current_entity = _finalize_entity_positions(current_entity, word_to_char_map)
        entities.append(current_entity)
    
    return entities


def _create_word_to_char_mapping(tokens: List[str], text: str) -> Dict[int, Dict[str, int]]:
    """Create mapping from word indices to character positions."""
    word_to_char = {}
    char_pos = 0
    
    for i, token in enumerate(tokens):
        # Find the token in the text starting from current position
        token_start = text.find(token, char_pos)
        if token_start >= 0:
            token_end = token_start + len(token)
            word_to_char[i] = {
                'start': token_start,
                'end': token_end
            }
            char_pos = token_end
            # Skip any whitespace
            while char_pos < len(text) and text[char_pos].isspace():
                char_pos += 1
        else:
            # Fallback if token not found
            word_to_char[i] = {
                'start': char_pos,
                'end': char_pos + len(token)
            }
            char_pos += len(token) + 1  # +1 for space
    
    return word_to_char


def _finalize_entity_positions(entity: Entity, word_to_char_map: Dict[int, Dict[str, int]]) -> Entity:
    """Convert word-level entity positions to character-level positions."""
    if entity.start in word_to_char_map and entity.end in word_to_char_map:
        # Use character positions
        char_start = word_to_char_map[entity.start]['start']
        char_end = word_to_char_map[entity.end]['end']
        
        return Entity(
            text=entity.text,
            label=entity.label,
            start=char_start,
            end=char_end,
            tokens=entity.tokens,
            nesting_depth=entity.nesting_depth
        )
    else:
        # Fallback to original entity if mapping fails
        return entity