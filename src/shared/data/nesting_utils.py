"""
Utility for detecting nested entities universally across datasets.
"""
from typing import List
from .data_types import Entity

def detect_nesting(entities: List[Entity]) -> List[Entity]:
    """
    Detect nested entities using geometric constraints and mark them.
    
    This function performs an O(N^2) check to identify entities that are strictly
    contained within other entities.
    
    Args:
        entities: List of Entity objects
        
    Returns:
        List of Entity objects with 'is_nested' flag set correctly
    """
    # Reset all nestings first to avoid stale state
    for entity in entities:
        entity.is_nested = False
        
    if len(entities) < 2:
        return entities
        
    # Sort by length descending to check if smaller fits in larger
    # (Though logic works O(N^2) regardless, structure helps debugging)
    
    for i, inner in enumerate(entities):
        for j, outer in enumerate(entities):
            if i == j:
                continue
                
            # Check for geometric containment
            # Inner entity must start at or after Outer start
            # Inner entity must end at or before Outer end
            # Must not be identical (unless duplicates exist, in which case one doesn't nest the other in strict sense, but let's assume strict subset for nesting)
            
            if (inner.start >= outer.start and 
                inner.end <= outer.end):
                
                # Check strict containment or identity
                if inner.start == outer.start and inner.end == outer.end:
                    # Identical spans: usually duplicates or multi-type
                    # Do not count as nesting of each other to avoid cycles
                    continue
                else:
                    # Strict subset found
                    inner.is_nested = True
                    break # reducing finding one outer parent is enough
                    
    return entities
