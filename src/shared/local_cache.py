"""
Local Cache Utility Module for Datasets and Models.

Provides a local-first loading strategy that checks project directories 
before falling back to HuggingFace Hub. This ensures reproducibility 
and protects against upstream dataset/model removals.

Design Principles:
- Local First: Always check local `datasets/` and `models/` folders first
- Fallback to HuggingFace: Download from HuggingFace if not found locally
- Auto-Cache: Save downloaded resources locally for future use
- Transparency: Log all loading sources for debugging and verification
"""

from pathlib import Path
from typing import Optional, Tuple
import shutil
import logging

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Absolute path to project root (parent of src/)
    """
    # Navigate from this file: src/utils/local_cache.py -> src -> project_root
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_local_datasets_dir() -> Path:
    """
    Get the local datasets directory path.
    
    Returns:
        Path: Absolute path to datasets directory
    """
    return get_project_root() / "datasets"


def get_local_models_dir() -> Path:
    """
    Get the local models directory path.
    
    Returns:
        Path: Absolute path to models directory
    """
    return get_project_root() / "models"


def get_local_dataset_path(dataset_name: str, config: Optional[str] = None) -> Optional[Path]:
    """
    Check if a dataset exists in the local datasets directory.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'tner/conll2003')
        config: Optional dataset configuration (e.g., 'ai' for CrossNER)
        
    Returns:
        Path to local dataset if found, None otherwise
    """
    datasets_dir = get_local_datasets_dir()
    
    # Convert HuggingFace name to directory format
    # e.g., 'tner/conll2003' -> 'tner___conll2003'
    safe_name = dataset_name.replace("/", "___")
    
    # Check for dataset directory
    dataset_path = datasets_dir / safe_name
    
    if config:
        # Check for config-specific subdirectory
        config_path = dataset_path / config
        if config_path.exists():
            logger.debug(f"Found local dataset at {config_path}")
            return config_path
    
    if dataset_path.exists():
        logger.debug(f"Found local dataset at {dataset_path}")
        return dataset_path
    
    logger.debug(f"Local dataset not found: {dataset_path}")
    return None


def get_local_model_path(model_name: str) -> Optional[Path]:
    """
    Check if a model exists in the local models directory.
    
    Args:
        model_name: HuggingFace model name (e.g., 'dslim/bert-base-NER')
        
    Returns:
        Path to local model if found, None otherwise
    """
    models_dir = get_local_models_dir()
    
    # Convert HuggingFace name to directory format
    # e.g., 'dslim/bert-base-NER' -> 'dslim___bert-base-NER'
    safe_name = model_name.replace("/", "___")
    
    model_path = models_dir / safe_name
    
    if model_path.exists():
        logger.debug(f"Found local model at {model_path}")
        return model_path
    
    logger.debug(f"Local model not found: {model_path}")
    return None


def save_dataset_locally(dataset_name: str, source_path: Path, config: Optional[str] = None) -> Path:
    """
    Copy a dataset from source (e.g., HuggingFace cache) to local datasets directory.
    
    Args:
        dataset_name: HuggingFace dataset name
        source_path: Source path (typically HuggingFace cache)
        config: Optional dataset configuration
        
    Returns:
        Path to the saved local dataset
        
    Raises:
        OSError: If copy operation fails
    """
    datasets_dir = get_local_datasets_dir()
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert name to safe directory format
    safe_name = dataset_name.replace("/", "___")
    
    target_path = datasets_dir / safe_name
    if config:
        target_path = target_path / config
    
    if target_path.exists():
        logger.info(f"Dataset already exists at {target_path}")
        return target_path
    
    logger.info(f"Saving dataset locally: {source_path} -> {target_path}")
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
    else:
        shutil.copy2(source_path, target_path)
    
    logger.info(f"Dataset saved to {target_path}")
    return target_path


def save_model_locally(model_name: str, source_path: Path) -> Path:
    """
    Copy a model from source (e.g., HuggingFace cache) to local models directory.
    
    Args:
        model_name: HuggingFace model name
        source_path: Source path (typically HuggingFace cache)
        
    Returns:
        Path to the saved local model
        
    Raises:
        OSError: If copy operation fails
    """
    models_dir = get_local_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert name to safe directory format
    safe_name = model_name.replace("/", "___")
    
    target_path = models_dir / safe_name
    
    if target_path.exists():
        logger.info(f"Model already exists at {target_path}")
        return target_path
    
    logger.info(f"Saving model locally: {source_path} -> {target_path}")
    
    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
    else:
        shutil.copy2(source_path, target_path)
    
    logger.info(f"Model saved to {target_path}")
    return target_path


def get_huggingface_cache_path() -> Path:
    """
    Get the HuggingFace cache directory path.
    
    Returns:
        Path to HuggingFace cache directory
    """
    return Path.home() / ".cache" / "huggingface"


def find_cached_dataset(dataset_name: str, config: Optional[str] = None) -> Optional[Path]:
    """
    Find a dataset in the HuggingFace cache.
    
    Args:
        dataset_name: HuggingFace dataset name
        config: Optional dataset configuration
        
    Returns:
        Path to cached dataset if found, None otherwise
    """
    cache_dir = get_huggingface_cache_path() / "datasets"
    
    # Convert name to cache directory format
    safe_name = dataset_name.replace("/", "___")
    
    dataset_path = cache_dir / safe_name
    
    if config:
        config_path = dataset_path / config
        if config_path.exists():
            return config_path
    
    if dataset_path.exists():
        return dataset_path
    
    return None


def find_cached_model(model_name: str) -> Optional[Path]:
    """
    Find a model in the HuggingFace cache.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Path to cached model if found, None otherwise
    """
    cache_dir = get_huggingface_cache_path() / "hub"
    
    # HuggingFace cache uses 'models--org--name' format
    safe_name = f"models--{model_name.replace('/', '--')}"
    
    model_path = cache_dir / safe_name
    
    if model_path.exists():
        # Find the snapshots directory with actual model files
        snapshots_path = model_path / "snapshots"
        if snapshots_path.exists():
            # Get the most recent snapshot
            snapshots = list(snapshots_path.iterdir())
            if snapshots:
                return sorted(snapshots)[-1]
        return model_path
    
    return None
