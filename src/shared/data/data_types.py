"""
Data types and classes for NER tasks.

This module contains the data structures used throughout the NER pipeline.
Uses original class names for compatibility with existing pipeline code.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Entity:
    """Represents a named entity with its properties."""
    text: str
    label: str  # Use 'label' to match existing pipeline expectations
    start: int
    end: int
    tokens: List[str]
    nesting_depth: int = 0
    is_nested: bool = False


@dataclass
class NERExample:
    """Represents a single NER example."""
    text: str
    tokens: List[str]
    labels: List[str]
    entities: List[Entity]
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None