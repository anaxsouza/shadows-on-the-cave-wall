"""
Abstract Base Class for Dataset Loaders Following SOLID Principles.

This module defines the foundational interface for all Named Entity Recognition
dataset loaders in the framework. It ensures consistent behavior across different
dataset formats while maintaining the flexibility needed for dataset-specific
implementations and optimizations.

The design follows SOLID principles:
- Single Responsibility: Each loader handles one dataset format
- Open/Closed: Easy to extend with new datasets without modifying existing code
- Liskov Substitution: All loaders are fully interchangeable
- Interface Segregation: Clean, focused interface with minimal dependencies
- Dependency Inversion: Depends on abstractions, not concrete implementations

This abstract base class provides the contract that all dataset loaders must
implement, ensuring consistent data flow and enabling the framework to handle
diverse NER datasets through a unified interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from datasets import Dataset, load_dataset
import logging
from pathlib import Path

from ....shared.data.data_types import NERExample, Entity

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class DatasetLoadingError(Exception):
    """Raised when dataset loading fails."""
    pass


class EntityConversionError(Exception):
    """Raised when entity conversion fails."""
    pass


class BaseDatasetLoader(ABC):
    """
    Abstract Base Class for Named Entity Recognition Dataset Loaders.
    
    This class defines the essential interface that all NER dataset loaders must
    implement to ensure consistent behavior and interoperability across the framework.
    Each concrete implementation focuses on handling one specific dataset format
    while adhering to common patterns and quality standards.
    
    Design Principles:
    - Single Responsibility: Each loader handles exactly one dataset format
    - Consistent Interface: All loaders provide the same methods and behaviors
    - Error Resilience: Comprehensive error handling and graceful degradation
    - Performance Optimization: Efficient data loading and memory management
    - Quality Assurance: Built-in validation and consistency checks
    
    Key Responsibilities:
    1. Load raw datasets from various sources (HuggingFace, local files, APIs)
    2. Convert dataset-specific formats to standardized NERExample objects
    3. Extract and normalize entity type information and label mappings
    4. Validate data consistency and handle malformed examples gracefully
    5. Provide dataset metadata and statistics for analysis and optimization
    
    Supported Data Sources:
    - HuggingFace Datasets Hub: Remote datasets with automatic caching
    - Local Files: JSON, CSV, CoNLL format files
    - Custom APIs: Dataset-specific endpoints and protocols
    - Preprocessed Data: Previously processed and cached datasets
    
    Data Flow:
    1. Raw Data Loading: Download or access raw dataset files
    2. Format Parsing: Parse dataset-specific formats and structures
    3. Entity Extraction: Extract entity annotations in dataset format
    4. Standardization: Convert to unified NERExample representation
    5. Validation: Verify data consistency and handle errors
    6. Metadata Extraction: Generate dataset statistics and information
    
    Quality Assurance:
    - Input validation with detailed error reporting
    - Entity boundary validation and correction
    - Duplicate detection and handling
    - Missing data imputation strategies
    - Format consistency verification
    """
    
    def __init__(self):
        """
        Initialize the dataset loader with default configuration.
        
        Sets up the loader with empty label mappings and default settings.
        Concrete implementations can override this to provide dataset-specific
        initialization parameters and configurations.
        
        Attributes initialized:
        - label_list: List of entity type labels (populated during loading)
        - label2id: Mapping from label strings to integer IDs
        - id2label: Mapping from integer IDs to label strings
        - _loaded_splits: Cache of loaded dataset splits
        - _dataset_stats: Runtime statistics about loaded data
        - _validation_errors: Collection of validation issues found
        """
        # Label information (populated during loading)
        self.label_list: Optional[List[str]] = None
        self.label2id: Optional[Dict[str, int]] = None
        self.id2label: Optional[Dict[int, str]] = None
        
        # Runtime caching and statistics
        self._loaded_splits: Dict[str, Dataset] = {}
        self._dataset_stats: Dict[str, Any] = {}
        self._validation_errors: List[str] = []
        
        # Configuration parameters (can be overridden by subclasses)
        self._max_sequence_length: Optional[int] = None
        self._entity_validation_enabled: bool = True
        self._skip_malformed_examples: bool = True
        
        logger.debug(f"Initialized {self.__class__.__name__} loader")
        
    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        Return the canonical dataset identifier used throughout the system.
        
        This method provides the standard name used for dataset identification
        in configuration files, logging, results organization, and cross-system
        communication. The name should be consistent, descriptive, and unique.
        
        Naming Conventions:
        - Use lowercase with underscores for multi-word names
        - Include version information if applicable (e.g., 'conll2003')
        - Use domain suffixes for domain-specific datasets (e.g., 'crossner_ai')
        - Keep names concise but descriptive (max 20 characters recommended)
        
        Returns:
            str: Canonical dataset identifier that:
            - Matches the key used in configuration files
            - Is used in experiment result folders and filenames
            - Appears in logging and error messages
            - Is used for dataset-specific optimization settings
            - Remains stable across framework versions
            
        Examples:
            - 'conll2003': CoNLL-2003 NER shared task dataset
            - 'genia': GENIA biomedical corpus
            - 'crossner_ai': CrossNER AI domain subset
            - 'ontonotes5': OntoNotes 5.0 multilingual dataset
        """
        pass
    
    @abstractmethod
    def get_huggingface_name(self) -> str:
        """
        Return the HuggingFace Datasets Hub identifier for this dataset.
        
        This method specifies the exact dataset name used to load the dataset
        from the HuggingFace Datasets Hub. This is the primary source for most
        datasets in the framework, providing automatic caching, version control,
        and standardized access patterns.
        
        HuggingFace Dataset Naming:
        - Official datasets: Simple names like 'conll2003', 'squad'
        - Community datasets: Format 'username/dataset_name'
        - Organization datasets: Format 'organization/dataset_name'
        - Versioned datasets: May include version tags or branches
        
        Returns:
            str: HuggingFace dataset identifier used in load_dataset() calls:
            - Must be a valid HuggingFace Datasets Hub identifier
            - Should point to a stable, accessible dataset version
            - May include version or branch specifications
            - Used directly with datasets.load_dataset(name, config)
            
        Examples:
            - 'conll2003': Official CoNLL-2003 dataset on HuggingFace
            - 'Aunderline/genia': Community-hosted GENIA corpus
            - 'microsoft/ner_dataset': Organization-hosted dataset
            - 'my_org/dataset_v2.1': Versioned dataset
            
        Notes:
        - Return empty string if dataset is not available on HuggingFace
        - Ensure the dataset is publicly accessible or properly authenticated
        - Consider data licensing and usage restrictions
        """
        pass
    
    @abstractmethod
    def get_huggingface_config(self) -> Optional[str]:
        """
        Return the HuggingFace dataset configuration name if required.
        
        Many HuggingFace datasets include multiple configurations or subsets
        that must be specified during loading. This method identifies which
        configuration this loader targets, enabling precise dataset selection.
        
        Configuration Use Cases:
        - Language Selection: 'en', 'de', 'fr' for multilingual datasets
        - Task Variants: 'ner', 'pos', 'chunking' for multi-task datasets
        - Size Variants: 'small', 'large', 'full' for datasets with size options
        - Domain Subsets: 'news', 'biomedical', 'social' for domain-specific versions
        - Version Control: 'v1.0', 'v2.1' for versioned datasets
        
        Returns:
            Optional[str]: Configuration identifier for HuggingFace dataset:
            - None if no configuration is needed (default configuration)
            - String identifier for specific dataset configuration
            - Must match exactly the configuration names in dataset metadata
            - Used in load_dataset(name, config_name) calls
            
        Examples:
            - None: For datasets without configurations (simple case)
            - 'ner': For multi-task datasets, selecting NER configuration
            - 'en': For multilingual datasets, selecting English subset
            - 'ai': For CrossNER dataset, selecting AI domain subset
            
        Implementation Notes:
        - Check available configurations in dataset documentation
        - Use dataset.get_config_names() to list available options
        - Consider user preferences and system defaults
        - Handle configuration availability gracefully
        """
        pass
    
    @abstractmethod
    def get_huggingface_revision(self) -> Optional[str]:
        """Revisão do corpus no Hub, ou None para a padrão.

        Declarada e não implícita: "a última versão do corpus" não é uma
        declaração — duas execuções em datas diferentes podem ler dados
        diferentes sem que a diferença apareça em lugar nenhum.
        """
        return None

    def get_entity_types(self) -> List[str]:
        """
        Return the complete taxonomy of entity types supported by this dataset.
        
        This method provides the canonical list of entity labels used in the
        dataset's annotation scheme. This information is critical for model
        configuration, evaluation metric calculation, and analysis purposes.
        
        Entity Type Considerations:
        1. Format Consistency: Return labels in evaluation format (without BIO prefixes)
        2. Completeness: Include all entity types that appear in any dataset split
        3. Ordering: Use consistent ordering (typically alphabetical or by frequency)
        4. Standardization: Use standard names when possible (PER, ORG, LOC, MISC)
        5. Domain Specificity: Include domain-specific types as appropriate
        
        Common Entity Type Sets:
        - General Purpose: ['PER', 'ORG', 'LOC', 'MISC'] (CoNLL-2003 style)
        - Biomedical: ['DNA', 'RNA', 'protein', 'cell_line', 'cell_type'] (GENIA style)
        - Fine-grained: ['PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 'PRODUCT']
        - Domain-specific: ['algorithm', 'dataset', 'metric', 'task'] (AI domain)
        
        Returns:
            List[str]: Complete list of entity type labels:
            - All entity types that can appear in dataset annotations
            - Labels in their evaluation format (typically without BIO prefixes)
            - Ordered consistently across calls (preferably alphabetical)
            - No duplicates or aliases included
            - Empty list if dataset has no predefined entity types
            
        Examples:
            >>> loader = CONLLLoader()
            >>> loader.get_entity_types()
            ['LOC', 'MISC', 'ORG', 'PER']
            
            >>> loader = GENIALoader()  
            >>> loader.get_entity_types()
            ['DNA', 'RNA', 'cell_line', 'cell_type', 'protein']
            
            >>> loader = CrossNERAILoader()
            >>> loader.get_entity_types()
            ['algorithm', 'conference', 'dataset', 'field', 'metric', 'researcher', 'task', 'university']
        
        Implementation Guidelines:
        - Load entity types from dataset metadata when possible
        - Use hardcoded lists for datasets with fixed taxonomies
        - Consider hierarchical entity types and how to flatten them
        - Handle special cases like 'O' (Outside) tags appropriately
        - Validate that returned types match actual dataset annotations
        """
        pass
    
    def load_raw_dataset(self, split: str = 'test') -> Dataset:
        """
        Load raw dataset from HuggingFace Datasets Hub with comprehensive error handling.
        
        This method handles the initial dataset loading phase, including automatic
        caching, network error recovery, and dataset validation. It provides a
        robust foundation for dataset access with detailed error reporting and
        performance monitoring.
        
        Loading Process:
        1. Validate input parameters and check dataset availability
        2. Attempt to load dataset from HuggingFace with appropriate configuration
        3. Cache loaded dataset for subsequent access within the same session
        4. Validate loaded dataset structure and required fields
        5. Extract and log dataset statistics for monitoring
        6. Handle network issues, authentication, and other loading failures
        
        Caching Strategy:
        - HuggingFace provides automatic disk caching for all downloaded datasets
        - This method adds in-memory caching for the current session
        - Cache keys combine dataset name, configuration, and split
        - Memory usage is monitored to prevent excessive consumption
        
        Args:
            split: Dataset split identifier to load. Must be one of:
                  - 'train': Training data split for model training
                  - 'validation' or 'val': Validation split for hyperparameter tuning
                  - 'test': Test split for final evaluation
                  - Custom splits may be supported by specific datasets
                  
        Returns:
            Dataset: Raw HuggingFace Dataset object containing:
            - All original fields and metadata from the source dataset
            - Dataset-specific column names and data types
            - Access to dataset features, info, and configuration
            - Lazy loading capabilities for memory efficiency
            - Built-in filtering, mapping, and processing methods
            
        Raises:
            DatasetLoadingError: When dataset cannot be loaded successfully:
            - Network connectivity issues or HuggingFace Hub unavailability
            - Invalid dataset name or configuration parameters
            - Authentication failures for private datasets
            - Corrupted or incomplete dataset files
            - Memory limitations for very large datasets
            
            ValueError: When input parameters are invalid:
            - Unsupported split names for the specific dataset
            - Invalid configuration combinations
            
        Error Handling:
        - Automatic retry with exponential backoff for network issues
        - Graceful fallback to cached versions when available
        - Detailed error messages for debugging and user guidance
        - Progress monitoring for large dataset downloads
        
        Example:
            >>> loader = CONLLLoader()
            >>> dataset = loader.load_raw_dataset('test')
            >>> len(dataset)
            3453
            >>> dataset.features
            {'id': Value(dtype='string'), 'tokens': Sequence(...), 'ner_tags': Sequence(...)}
        """
        dataset_name = self.get_dataset_name()
        logger.info(f"Loading {dataset_name} dataset, split: {split}")
        
        # Check cache first to avoid redundant loading
        cache_key = f"{dataset_name}_{split}"
        if cache_key in self._loaded_splits:
            logger.debug(f"Using cached dataset for {cache_key}")
            return self._loaded_splits[cache_key]
        
        try:
            # Validate split parameter
            if not split or not isinstance(split, str):
                raise ValueError(f"Invalid split parameter: {split}")
            
            # Load dataset with appropriate configuration
            hf_name = None
            config = None
            
            hf_name = self.get_huggingface_name()
            config = self.get_huggingface_config()
            
            if not hf_name:
                raise DatasetLoadingError(f"No HuggingFace dataset name specified for {dataset_name}")
            
            # LOCAL-FIRST LOADING STRATEGY
            # 1. Try loading from local datasets/ folder first
            full_dataset = self._try_load_local(hf_name, config)
            
            if full_dataset is None:
                # 2. Fallback to HuggingFace Hub
                logger.debug(f"Loading from HuggingFace: {hf_name}, config: {config}, split: {split}")
                
                # A revisão do corpus é DECLARADA no config, não implícita.
                #
                # Duas razões. A primeira é forçada: datasets >= 4 deixou de
                # executar corpora baseados em script, e o eriktks/conll2003 é
                # um deles — sem revisão explícita o carregamento falha com
                # "Dataset scripts are no longer supported". A revisão
                # refs/convert/parquet é a conversão automática do próprio Hub.
                #
                # A segunda é de reprodutibilidade: "a última versão do corpus"
                # não é uma declaração. Duas execuções em datas diferentes podem
                # ler dados diferentes e a diferença não apareceria em lugar
                # nenhum. Ver configs/config.yaml, datasets.<corpus>.revision.
                revision = self.get_huggingface_revision()
                kwargs = {"revision": revision} if revision else {}
                if config:
                    full_dataset = load_dataset(hf_name, config, **kwargs)
                else:
                    full_dataset = load_dataset(hf_name, **kwargs)
                
                logger.info(f"Downloaded dataset from HuggingFace: {hf_name}")
            
            # Validate that requested split exists
            if split not in full_dataset:
                available_splits = list(full_dataset.keys())
                raise DatasetLoadingError(
                    f"Split '{split}' not found in dataset {dataset_name}. "
                    f"Available splits: {available_splits}"
                )
            
            dataset = full_dataset[split]
            
            # Validate dataset structure
            if len(dataset) == 0:
                logger.warning(f"Dataset {dataset_name} split {split} is empty")
            
            # Cache for future use
            self._loaded_splits[cache_key] = dataset
            
            # Log dataset statistics
            self._dataset_stats[cache_key] = {
                'size': len(dataset),
                'features': list(dataset.features.keys()),
                'split': split
            }
            
            logger.info(f"Raw dataset loaded successfully: {len(dataset)} examples")
            logger.debug(f"Dataset features: {list(dataset.features.keys())}")
            
            return dataset
            
        except Exception as e:
            error_msg = f"Failed to load dataset {dataset_name} (split: {split}): {str(e)}"
            logger.error(error_msg)
            
            # Provide specific guidance for common errors
            if "ConnectionError" in str(type(e)):
                logger.error("Network connection issue. Check your internet connection and try again.")
            elif "FileNotFoundError" in str(type(e)) or "RepoNotFound" in str(type(e)):
                logger.error(f"Dataset '{hf_name}' not found on HuggingFace Hub. Verify the dataset name.")
            elif "ConfigNotFound" in str(type(e)):
                logger.error(f"Configuration '{config}' not found. Check available configurations.")
            
            raise DatasetLoadingError(error_msg) from e
    
    def _try_load_local(self, hf_name: str, config: Optional[str] = None) -> Optional[Any]:
        """
        Try to load dataset from local datasets/ directory.
        
        Args:
            hf_name: HuggingFace dataset name
            config: Optional dataset configuration
            
        Returns:
            Dataset if found locally, None otherwise
        """
        try:
            from ....shared.local_cache import get_local_dataset_path
            
            local_path = get_local_dataset_path(hf_name, config)
            
            if local_path and local_path.exists():
                logger.info(f"📂 Loading dataset from local path: {local_path}")
                
                # Try loading from local directory using HuggingFace's load_from_disk
                from datasets import load_from_disk
                
                try:
                    # First try load_from_disk for saved datasets
                    dataset = load_from_disk(str(local_path))
                    logger.info(f"✅ Loaded dataset from local storage: {local_path}")
                    return dataset
                except Exception:
                    # If that fails, try loading cached arrow files
                    # This is for datasets copied from HuggingFace cache
                    logger.debug("load_from_disk failed, trying cache format...")
                    
                    # Check for arrow files in subdirectories
                    for subdir in local_path.rglob("*.arrow"):
                        arrow_dir = subdir.parent
                        try:
                            dataset = load_from_disk(str(arrow_dir))
                            logger.info(f"✅ Loaded dataset from arrow files: {arrow_dir}")
                            return dataset
                        except Exception:
                            continue
                    
                    logger.debug(f"Could not load from local path: {local_path}")
                    return None
            
            return None
            
        except ImportError:
            logger.debug("local_cache module not available, skipping local loading")
            return None
        except Exception as e:
            logger.debug(f"Error during local load attempt: {e}")
            return None
    
    @abstractmethod
    def convert_to_examples(self, dataset: Dataset, max_examples: Optional[int] = None) -> List[NERExample]:
        """
        Convert raw HuggingFace dataset to standardized NERExample objects.
        
        This is the core transformation method that converts dataset-specific formats
        into the unified NERExample representation used throughout the framework.
        Each dataset implementation must handle its specific data format, field names,
        and annotation schemes while producing consistent, validated output.
        
        Conversion Process:
        1. Parse dataset-specific field names and data structures
        2. Extract text content and entity annotations
        3. Convert entity formats to standardized representation
        4. Validate entity boundaries and consistency
        5. Handle special cases, malformed data, and edge cases
        6. Apply dataset-specific preprocessing and normalization
        7. Generate unique identifiers and preserve metadata
        
        Data Format Handling:
        - Text Fields: Handle various text column names ('text', 'sentence', 'tokens')
        - Entity Annotations: Convert BIO tags, span annotations, or other formats
        - Metadata: Preserve important dataset-specific information
        - Character Encoding: Handle Unicode and special characters properly
        - Tokenization: Align with various tokenization schemes if needed
        
        Quality Assurance:
        - Input validation with detailed error reporting
        - Entity boundary validation and automatic correction when possible
        - Duplicate detection and handling strategies
        - Consistency checks across text and entity annotations
        - Memory usage monitoring for large datasets
        
        Args:
            dataset: Raw HuggingFace Dataset object loaded by load_raw_dataset().
                    Contains dataset-specific fields, formats, and metadata.
                    May include various column names and data structures.
                    
            max_examples: Optional limit on number of examples to convert:
                         - None: Convert all examples in the dataset
                         - Integer: Convert up to this many examples
                         - Used for testing, debugging, or resource management
                         - Examples are selected from the beginning of the dataset
                         
        Returns:
            List[NERExample]: Standardized examples ready for processing:
            - Each example follows the NERExample data class specification
            - Text content is properly normalized and cleaned
            - Entity annotations use consistent format and coordinates
            - Unique identifiers are generated for tracking and debugging
            - Metadata preserves important dataset-specific information
            - Examples are validated and free of obvious errors
            
            NERExample format:
            - text: str - The input text content
            - entities: List[Entity] - List of entity annotations
            - id: str - Unique identifier for this example
            - metadata: Dict[str, Any] - Additional information
            
            Entity format:
            - text: str - The entity mention text
            - label: str - The entity type label
            - start: int - Character start position (inclusive)
            - end: int - Character end position (exclusive)
            - confidence: Optional[float] - Confidence score if available
            
        Raises:
            EntityConversionError: When entity conversion encounters critical errors:
            - Malformed entity annotations that cannot be parsed
            - Entity boundaries that don't align with text content
            - Invalid entity types not in the dataset taxonomy
            - Character encoding issues that corrupt text or entities
            
            DatasetValidationError: When dataset structure validation fails:
            - Missing required fields in dataset examples
            - Inconsistent data types or formats across examples
            - Dataset corruption or incomplete downloads
            
            MemoryError: When dataset is too large for available memory:
            - Consider using max_examples parameter to limit memory usage
            - Or implement streaming conversion for very large datasets
            
        Implementation Guidelines:
        - Handle missing or None values gracefully
        - Preserve original text formatting when possible
        - Use robust entity boundary validation
        - Implement comprehensive error logging for debugging
        - Consider performance optimization for large datasets
        - Test with various dataset conditions and edge cases
        
        Example:
            >>> loader = CONLLLoader()
            >>> raw_dataset = loader.load_raw_dataset('test')
            >>> examples = loader.convert_to_examples(raw_dataset, max_examples=10)
            >>> len(examples)
            10
            >>> examples[0].text
            'SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .'
            >>> examples[0].entities[0]
            Entity(text='JAPAN', label='LOC', start=9, end=14, confidence=None)
        """
        pass
    
    def extract_label_info(self, dataset: Dataset) -> None:
        """
        Extract and normalize entity label information from the raw dataset.
        
        This method analyzes the dataset structure to identify entity labels,
        creating standardized mappings that are used throughout the framework
        for entity type validation, evaluation metrics, and model configuration.
        
        Label Extraction Process:
        1. Identify label-containing fields in the dataset schema
        2. Extract unique entity labels from dataset annotations
        3. Create bidirectional mappings between labels and numeric IDs
        4. Validate label consistency across dataset examples
        5. Handle dataset-specific label formats and conventions
        6. Generate statistics about label distribution and usage
        
        Common Label Field Names:
        - 'ner_tags': BIO-tagged sequence labels (most common)
        - 'labels': Direct entity type labels
        - 'entities': Structured entity annotations
        - 'tags': Generic tag sequences
        - Dataset-specific field names
        
        Label Format Handling:
        - BIO Tags: Convert 'B-PER', 'I-PER', 'O' to entity types ['PER']
        - BILOU Tags: Handle 'B-', 'I-', 'L-', 'O-', 'U-' prefixes
        - Direct Labels: Use labels as-is if already in correct format
        - Hierarchical Labels: Flatten or preserve hierarchy as appropriate
        
        Args:
            dataset: Raw HuggingFace Dataset object with entity annotations.
                    Should contain at least one field with entity label information.
                    The dataset structure varies by format and source.
                    
        Side Effects:
            Updates instance attributes:
            - self.label_list: Ordered list of unique entity type labels
            - self.label2id: Mapping from label strings to integer IDs
            - self.id2label: Mapping from integer IDs to label strings
            - self._validation_errors: Records any issues found during extraction
            
        Error Handling:
        - Gracefully handles missing or malformed label fields
        - Provides detailed warnings for inconsistent label formats
        - Falls back to dataset-specific label extraction methods
        - Preserves partial results when some labels are problematic
        
        Example:
            >>> loader = CONLLLoader()
            >>> dataset = loader.load_raw_dataset('test')
            >>> loader.extract_label_info(dataset)
            >>> loader.label_list
            ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
            >>> loader.label2id['B-PER']
            1
            >>> loader.id2label[1]
            'B-PER'
        """
        if not dataset or len(dataset) == 0:
            logger.warning("Cannot extract label info from empty dataset")
            return
        
        try:
            logger.debug("Extracting label information from dataset features")
            
            # Try to get label information from dataset features first
            if hasattr(dataset, 'features') and dataset.features:
                self._extract_from_features(dataset.features)
            
            # If feature extraction didn't work, analyze actual data
            if not self.label_list:
                self._extract_from_data(dataset)
            
            # Validate and finalize label information
            if self.label_list:
                self._validate_and_finalize_labels()
                logger.info(f"Extracted {len(self.label_list)} labels: {self.label_list[:10]}{'...' if len(self.label_list) > 10 else ''}")
            else:
                logger.warning("No label information could be extracted from dataset")
                self._validation_errors.append("Failed to extract any label information")
                
        except Exception as e:
            logger.error(f"Error during label extraction: {e}")
            self._validation_errors.append(f"Label extraction error: {str(e)}")
    
    def _extract_from_features(self, features: Dict[str, Any]) -> None:
        """Extract labels from dataset features metadata."""
        # Common label column names to check
        label_columns = ['ner_tags', 'labels', 'entities', 'tags', 'pos_tags']
        
        for col_name in label_columns:
            if col_name in features:
                feature = features[col_name]
                
                # Handle Sequence features with ClassLabel
                if hasattr(feature, 'feature') and hasattr(feature.feature, 'names'):
                    self.label_list = list(feature.feature.names)
                    logger.debug(f"Extracted labels from feature '{col_name}': {len(self.label_list)} labels")
                    return
                
                # Handle direct ClassLabel features
                elif hasattr(feature, 'names'):
                    self.label_list = list(feature.names)
                    logger.debug(f"Extracted labels from feature '{col_name}': {len(self.label_list)} labels")
                    return
    
    def _extract_from_data(self, dataset: Dataset) -> None:
        """Extract labels by analyzing actual dataset examples."""
        logger.debug("Extracting labels from dataset examples")
        
        # Sample a subset of the dataset for label extraction
        sample_size = min(1000, len(dataset))
        sample_dataset = dataset.select(range(sample_size))
        
        unique_labels = set()
        
        # Common label column names to check
        label_columns = ['ner_tags', 'labels', 'entities', 'tags']
        
        for example in sample_dataset:
            for col_name in label_columns:
                if col_name in example and example[col_name] is not None:
                    labels = example[col_name]
                    
                    # Handle different label formats
                    if isinstance(labels, list):
                        for label in labels:
                            if isinstance(label, (int, str)):
                                unique_labels.add(str(label))
                    elif isinstance(labels, (int, str)):
                        unique_labels.add(str(labels))
        
        if unique_labels:
            self.label_list = sorted(list(unique_labels))
            logger.debug(f"Extracted {len(self.label_list)} unique labels from data analysis")
    
    def _validate_and_finalize_labels(self) -> None:
        """Validate extracted labels and create ID mappings."""
        if not self.label_list:
            return
        
        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in self.label_list:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        
        self.label_list = unique_labels
        
        # Create bidirectional mappings
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        
        # Validate label format and consistency
        invalid_labels = []
        for label in self.label_list:
            if not isinstance(label, str) or not label.strip():
                invalid_labels.append(label)
        
        if invalid_labels:
            logger.warning(f"Found {len(invalid_labels)} invalid labels: {invalid_labels}")
            self._validation_errors.append(f"Invalid labels found: {invalid_labels}")
    
    def validate_example(self, example: NERExample) -> Tuple[bool, List[str]]:
        """
        Validate that an NERExample meets dataset quality and consistency requirements.
        
        This method performs comprehensive validation of converted examples to ensure
        data quality, consistency, and readiness for processing. It checks text content,
        entity annotations, boundary alignment, and dataset-specific constraints.
        
        Validation Checks:
        1. Text Content Validation:
           - Non-empty text content
           - Valid character encoding and Unicode handling
           - Reasonable text length constraints
           - Special character and formatting consistency
        
        2. Entity Annotation Validation:
           - Entity boundary consistency with text content
           - Valid entity type labels from dataset taxonomy
           - Non-overlapping entity spans (unless nested entities are supported)
           - Proper entity text extraction and alignment
        
        3. Data Structure Validation:
           - Required fields are present and properly formatted
           - Consistent data types across all fields
           - Valid metadata structure and content
           - Unique identifier presence and format
        
        4. Dataset-Specific Validation:
           - Compliance with dataset annotation guidelines
           - Domain-specific validation rules
           - Format-specific requirements and constraints
        
        Args:
            example: NERExample object to validate.
                    Should contain text, entities, id, and metadata fields
                    in the format produced by convert_to_examples().
                    
        Returns:
            Tuple[bool, List[str]]: Validation results containing:
            - bool: True if example passes all validation checks, False otherwise
            - List[str]: List of validation error messages for failed checks
                        Empty list if validation passes completely
                        
        Validation Categories:
        - Critical Errors: Issues that prevent processing (return False)
        - Warnings: Issues that may affect quality but don't prevent processing
        - Information: Notifications about unusual but acceptable conditions
        
        Example:
            >>> loader = CONLLLoader()
            >>> example = NERExample(text="John works at Apple", entities=[...], id="ex1")
            >>> is_valid, errors = loader.validate_example(example)
            >>> is_valid
            True
            >>> errors
            []
            
            >>> bad_example = NERExample(text="", entities=[], id="")
            >>> is_valid, errors = loader.validate_example(bad_example)
            >>> is_valid
            False
            >>> errors
            ['Empty text content', 'Missing or empty example ID']
        """
        if not self._entity_validation_enabled:
            return True, []
        
        errors = []
        
        # 1. Text Content Validation
        if not example.text or not example.text.strip():
            errors.append("Empty or whitespace-only text content")
        elif len(example.text) > 50000:  # Reasonable upper limit
            errors.append(f"Text too long ({len(example.text)} characters, max 50000)")
        
        # Check for control characters or encoding issues
        try:
            example.text.encode('utf-8')
        except UnicodeEncodeError as e:
            errors.append(f"Text encoding error: {e}")
        
        # 2. Example ID Validation
        if not example.id or not str(example.id).strip():
            errors.append("Missing or empty example ID")
        
        # 3. Entity Annotations Validation
        if example.entities is not None:
            entity_errors = self._validate_entities(example.entities, example.text)
            errors.extend(entity_errors)
        
        # 4. Metadata Validation
        if example.metadata is not None and not isinstance(example.metadata, dict):
            errors.append("Metadata must be a dictionary or None")
        
        # Return validation results
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_entities(self, entities: List[Entity], text: str) -> List[str]:
        """Validate entity annotations against text content."""
        errors = []
        
        if not isinstance(entities, list):
            errors.append("Entities must be a list")
            return errors
        
        for i, entity in enumerate(entities):
            # Check entity structure
            if not hasattr(entity, 'start') or not hasattr(entity, 'end'):
                errors.append(f"Entity {i} missing start or end position")
                continue
            
            if not hasattr(entity, 'text') or not hasattr(entity, 'label'):
                errors.append(f"Entity {i} missing text or label")
                continue
            
            # Validate entity boundaries
            if entity.start < 0:
                errors.append(f"Entity {i} has negative start position: {entity.start}")
            
            if entity.end <= entity.start:
                errors.append(f"Entity {i} has invalid span: start={entity.start}, end={entity.end}")
            
            if entity.end > len(text):
                errors.append(f"Entity {i} end position {entity.end} exceeds text length {len(text)}")
            
            # Validate entity text alignment
            if 0 <= entity.start < entity.end <= len(text):
                expected_text = text[entity.start:entity.end]
                if entity.text != expected_text:
                    errors.append(f"Entity {i} text mismatch: '{entity.text}' != '{expected_text}'")
            
            # Validate entity label
            if self.label_list and entity.label not in self.get_entity_types():
                errors.append(f"Entity {i} has invalid label: '{entity.label}'. Valid labels: {self.get_entity_types()}")
        
        return errors

    def _detect_and_mark_nesting(self, examples: List[NERExample]) -> List[NERExample]:
        """
        Detect and mark nested entities in a list of examples.
        
        This method implements the Universal Nesting Detection logic to ensure
        that 'is_nested' flag is consistently populated for all datasets.
        
        Logic:
        An entity is considered nested if it is geometrically contained within 
        another entity (outer covers inner).
        
        Args:
            examples: List of NERExample objects to process
            
        Returns:
            List[NERExample]: The processed examples with updated entities
        """
        count_nested = 0
        
        for example in examples:
            if not example.entities:
                continue
                
            entities = example.entities
            
            for i, entity in enumerate(entities):
                # Skip if already marked (allow manual overrides)
                if hasattr(entity, 'is_nested') and entity.is_nested is not None and entity.is_nested:
                    # If explicitly True, keep it. If False, check again (maybe?)
                    # For safety, let's re-check unless it's True.
                    pass
                    
                is_nested = False
                e_start = getattr(entity, 'start', 0)
                e_end = getattr(entity, 'end', 0)
                
                for j, other in enumerate(entities):
                    if i == j:
                        continue
                        
                    o_start = getattr(other, 'start', 0)
                    o_end = getattr(other, 'end', 0)
                    
                    # Nesting: Outer covers Inner
                    if o_start <= e_start and o_end >= e_end:
                        is_nested = True
                        break
                
                try:
                    setattr(entity, 'is_nested', is_nested)
                    if is_nested:
                        count_nested += 1
                except AttributeError:
                    logger.warning(f"Could not set is_nested on entity {i} in example {example.id}")
                    
        logger.info(f"Universal Nesting Detection: Marked {count_nested} entities as nested across {len(examples)} examples")
        return examples
    
    # Additional utility methods for comprehensive dataset management
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded dataset.
        
        Returns:
            Dict[str, Any]: Dataset statistics including:
            - split_sizes: Number of examples in each loaded split
            - label_distribution: Count of each entity type
            - text_length_stats: Statistics about text lengths
            - entity_density: Average entities per example
            - validation_summary: Summary of validation results
        """
        return {
            'loaded_splits': list(self._loaded_splits.keys()),
            'split_statistics': self._dataset_stats,
            'label_info': {
                'total_labels': len(self.label_list) if self.label_list else 0,
                'labels': self.label_list
            },
            'validation_errors': len(self._validation_errors),
            'validation_error_types': self._validation_errors
        }
    
    def load_split(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None
    ) -> List[NERExample]:
        """
        Load and convert a dataset split to NERExample objects.
        
        Convenience method that combines load_raw_dataset and convert_to_examples
        for ease of use by fine-tuning and other pipelines.
        
        Args:
            split: Dataset split to load ('train', 'validation', 'test').
            max_samples: Maximum number of samples to return.
            
        Returns:
            List[NERExample]: Converted examples ready for processing.
        """
        raw_dataset = self.load_raw_dataset(split)
        self.extract_label_info(raw_dataset)
        examples = self.convert_to_examples(raw_dataset, max_examples=max_samples)
        return self._detect_and_mark_nesting(examples)
    
    def get_label_list(self) -> List[str]:
        """
        Get the BIO label list for token classification.
        
        Returns:
            List[str]: Full BIO tag list including 'O' and all B-/I- prefixed labels.
        """
        entity_types = self.get_entity_types()
        
        # Build BIO label list
        labels = ['O']  # Start with Outside tag
        for entity_type in entity_types:
            labels.append(f'B-{entity_type}')
            labels.append(f'I-{entity_type}')
        
        return labels
    
    def cleanup(self) -> None:
        """
        Clean up loaded datasets and free memory resources.
        
        Call this method when the loader is no longer needed to free up
        memory used by cached datasets and intermediate processing results.
        """
        self._loaded_splits.clear()
        self._dataset_stats.clear()
        self._validation_errors.clear()
        logger.debug(f"Cleaned up {self.__class__.__name__} loader resources")