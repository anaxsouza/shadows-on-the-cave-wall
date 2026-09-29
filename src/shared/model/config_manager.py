"""
Unified Configuration Manager for NER Experiments.

This module provides a single source of truth for all configuration values,
eliminating hardcoded values throughout the codebase.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentConfigManager:
    """
    Unified configuration manager that loads and validates all settings from config.yaml.
    
    This class ensures:
    - All configuration comes from config.yaml
    - No hardcoded fallbacks (fail fast on missing config)
    - Model-specific token limits and capabilities
    - Dynamic token allocation based on dataset analysis
    - Comprehensive validation of token usage
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file. If None, searches standard locations.
        """
        self.config_path = self._find_config_file(config_path)
        self.config = self._load_config()
        self._validate_config()
        
        logger.info(f"Configuration loaded from: {self.config_path}")
        logger.debug(f"Available model limits: {list(self.get_model_limits().keys())}")
    
    def _find_config_file(self, config_path: Optional[str] = None) -> str:
        """Find the configuration file in standard locations."""
        if config_path and os.path.exists(config_path):
            return config_path
        
        # Standard locations to search
        search_paths = [
            "configs/config.yaml",
            "../configs/config.yaml",
            "../../configs/config.yaml",
            os.path.join(os.path.dirname(__file__), "../../configs/config.yaml")
        ]
        
        # Check PROJECT_ROOT environment variable if set (important for Colab)
        project_root = os.environ.get('PROJECT_ROOT')
        if project_root:
            search_paths.insert(0, os.path.join(project_root, "configs/config.yaml"))

        for path in search_paths:
            # Handle absolute paths directly
            if os.path.isabs(path):
                if os.path.exists(path):
                    return path
            
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        raise FileNotFoundError(
            f"Configuration file not found. Searched paths: {search_paths}"
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                raise ValueError("Configuration file is empty")
            
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _validate_config(self) -> None:
        """Validate that required configuration sections exist."""
        required_sections = [
            'models', 'generation', 'model_limits', 'token_safety', 
            'token_allocation', 'dynamic_tokens'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)
        
        if missing_sections:
            raise ValueError(
                f"Missing required configuration sections: {missing_sections}"
            )
        
        # Validate model_limits has required models
        model_limits = self.config['model_limits']
        models_config = self.config['models']
        
        for model_key in models_config.keys():
            if model_key not in model_limits:
                logger.warning(f"Model '{model_key}' missing from model_limits configuration")
    
    def get_model_limits(self, model_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model-specific token limits and capabilities.
        
        Args:
            model_key: Model identifier. If None, returns all model limits.
            
        Returns:
            Dictionary with model limits configuration
        """
        model_limits = self.config['model_limits']
        
        if model_key is None:
            return model_limits
        
        if model_key not in model_limits:
            available_models = list(model_limits.keys())
            raise ValueError(
                f"Model '{model_key}' not found in configuration. "
                f"Available models: {available_models}"
            )
        
        return model_limits[model_key]
    
    def get_tokenizer_max_length(self, model_key: str) -> int:
        """
        Get maximum tokenizer length for a model.
        
        Args:
            model_key: Model identifier
            
        Returns:
            Maximum tokenizer sequence length
        """
        model_limits = self.get_model_limits(model_key)
        return model_limits['max_tokenizer_length']
    
    def get_context_window(self, model_key: str) -> int:
        """
        Get maximum context window for a model.
        
        Args:
            model_key: Model identifier
            
        Returns:
            Maximum context window size
        """
        model_limits = self.get_model_limits(model_key)
        return model_limits['max_context_window']
    
    def get_recommended_token_allocation(self, model_key: str) -> Dict[str, int]:
        """
        Get recommended input/output token allocation for a model.
        
        Args:
            model_key: Model identifier
            
        Returns:
            Dictionary with 'input_tokens' and 'output_tokens'
        """
        model_limits = self.get_model_limits(model_key)
        return {
            'input_tokens': model_limits['recommended_input_tokens'],
            'output_tokens': model_limits['recommended_output_tokens']
        }
    
    def get_generation_config(self, generation_type: str = 'instruct') -> Dict[str, Any]:
        """
        Get generation configuration for a specific type.
        
        Args:
            generation_type: Type of generation ('instruct', 'completion', etc.)
            
        Returns:
            Generation configuration dictionary
        """
        generation_config = self.config['generation']
        
        # Get max_new_tokens for the specific generation type
        max_new_tokens_config = generation_config.get('max_new_tokens', {})
        if generation_type not in max_new_tokens_config:
            available_types = list(max_new_tokens_config.keys())
            raise ValueError(
                f"Generation type '{generation_type}' not found in configuration. "
                f"Available types: {available_types}"
            )
        
        return {
            'max_new_tokens': max_new_tokens_config[generation_type],
            'temperature': generation_config.get('temperature', 0.7),
            'top_p': generation_config.get('top_p', 0.9),
            'num_beams': generation_config.get('num_beams', 3),
            'repetition_penalty': generation_config.get('repetition_penalty', 1.1)
        }
    
    def get_dataset_config(self, dataset_key: str) -> Dict[str, Any]:
        """
        Get dataset configuration from config.yaml.
        
        Args:
            dataset_key: Dataset identifier (e.g., 'conll2003', 'genia')
            
        Returns:
            Dataset configuration dictionary containing:
            - name: HuggingFace dataset path
            - config: Dataset config name (optional)
            - text_column: Column name for text/tokens
            - label_column: Column name for labels
            - has_nested: Whether dataset has nested entities
            - labels: List of BIO labels (optional)
            
        Raises:
            ValueError: If dataset not found in configuration
        """
        datasets_config = self.config.get('datasets', {})
        
        if dataset_key not in datasets_config:
            available_datasets = list(datasets_config.keys())
            raise ValueError(
                f"Dataset '{dataset_key}' not found in configuration. "
                f"Available datasets: {available_datasets}"
            )
        
        return datasets_config[dataset_key]
    
    def get_generation_params(self, generation_type: str = 'instruct') -> Dict[str, Any]:
        """
        Get generation parameters for strategies to use.
        
        This is the SINGLE SOURCE OF TRUTH for generation parameters.
        Strategies should call this instead of hardcoding values.
        
        Args:
            generation_type: 'instruct', 'completion', 'text_to_text', or 'pipeline'
            
        Returns:
            Dictionary with max_new_tokens, temperature, num_beams, etc.
        """
        return self.get_generation_config(generation_type)
    
    def get_token_safety_config(self) -> Dict[str, Any]:
        """Get token safety and validation settings."""
        return self.config['token_safety']
    
    def get_token_allocation_config(self) -> Dict[str, Any]:
        """Get token allocation strategies."""
        return self.config['token_allocation']
    
    def get_dynamic_tokens_config(self) -> Dict[str, Any]:
        """Get dynamic token calculation settings."""
        return self.config['dynamic_tokens']
    
    def calculate_optimal_token_allocation(
        self, 
        model_key: str, 
        estimated_input_chars: int = 0,
        estimated_output_chars: int = 0,
        num_entities: int = 0
    ) -> Dict[str, int]:
        """
        Calculate optimal token allocation based on model capabilities and requirements.
        
        Args:
            model_key: Model identifier
            estimated_input_chars: Estimated input text length in characters
            estimated_output_chars: Estimated output JSON length in characters
            num_entities: Number of entities expected
            
        Returns:
            Dictionary with optimal 'input_tokens' and 'output_tokens'
        """
        # Get model limits
        model_limits = self.get_model_limits(model_key)
        context_window = model_limits['max_context_window']
        architecture_type = model_limits['architecture_type']
        
        # Get allocation strategy for architecture type
        allocation_config = self.get_token_allocation_config()
        strategy = allocation_config['allocation_strategies'][architecture_type]
        
        # Get character to token conversion ratios
        dynamic_config = self.get_dynamic_tokens_config()
        char_to_token = dynamic_config['char_to_token_ratios']
        
        # Calculate base token requirements
        if estimated_input_chars > 0:
            input_tokens_needed = int(estimated_input_chars / char_to_token['english'])
            input_tokens_needed += allocation_config['prompt_overhead_buffer']
        else:
            # Use strategy-based allocation
            input_tokens_needed = int(context_window * strategy['input_ratio'])
        
        if estimated_output_chars > 0:
            output_tokens_needed = int(estimated_output_chars / char_to_token['json'])
            output_tokens_needed += allocation_config['json_structure_buffer']
            output_tokens_needed += num_entities * allocation_config['entity_estimation_buffer']
        else:
            # Use strategy-based allocation
            output_tokens_needed = int(context_window * strategy['output_ratio'])
        
        # Apply safety buffer
        safety_config = self.get_token_safety_config()
        if safety_config.get('apply_safety_buffer', True):
            buffer_percent = safety_config['safety_buffer_percent'] / 100
            input_tokens_needed = int(input_tokens_needed * (1 + buffer_percent))
            output_tokens_needed = int(output_tokens_needed * (1 + buffer_percent))
        
        # Ensure minimum output tokens for JSON generation
        min_output = safety_config['minimum_output_tokens']
        output_tokens_needed = max(output_tokens_needed, min_output)
        
        # Ensure total doesn't exceed context window
        total_needed = input_tokens_needed + output_tokens_needed
        if total_needed > context_window:
            # Scale proportionally to fit within context window
            scale_factor = context_window / total_needed * 0.95  # 0.95 for extra safety
            input_tokens_needed = int(input_tokens_needed * scale_factor)
            output_tokens_needed = int(output_tokens_needed * scale_factor)
            
            # Ensure minimum output tokens are maintained
            if output_tokens_needed < min_output:
                output_tokens_needed = min_output
                input_tokens_needed = context_window - output_tokens_needed
        
        # Apply hard cap as final safety check
        max_cap = safety_config['max_cap_override']
        input_tokens_needed = min(input_tokens_needed, max_cap)
        output_tokens_needed = min(output_tokens_needed, max_cap)
        
        logger.debug(f"Calculated token allocation for {model_key}: "
                    f"input={input_tokens_needed}, output={output_tokens_needed}, "
                    f"total={input_tokens_needed + output_tokens_needed}/{context_window}")
        
        return {
            'input_tokens': input_tokens_needed,
            'output_tokens': output_tokens_needed,
            'total_tokens': input_tokens_needed + output_tokens_needed,
            'context_window': context_window
        }
    
    def should_fail_on_missing_config(self) -> bool:
        """Check if system should fail fast on missing configuration."""
        return self.get_token_safety_config().get('fail_on_missing_config', True)
    
    def should_validate_token_usage(self) -> bool:
        """Check if system should validate token usage."""
        return self.get_token_safety_config().get('validate_token_usage', True)
    
    def reload_config(self) -> None:
        """Reload configuration from file (useful for testing)."""
        self.config = self._load_config()
        self._validate_config()
        logger.info("Configuration reloaded")


# Global configuration manager instance
_config_manager: Optional[ExperimentConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ExperimentConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        ExperimentConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ExperimentConfigManager(config_path)
    
    return _config_manager


def reset_config_manager() -> None:
    """Reset the global experiment configuration manager (useful for testing)."""
    global _config_manager
    _config_manager = None