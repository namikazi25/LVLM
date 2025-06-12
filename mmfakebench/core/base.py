"""Base classes for the MMFakeBench toolkit.

This module defines the fundamental abstract base classes that all components
of the benchmarking system should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class BaseClient(ABC):
    """Abstract base class for all model clients.
    
    This class defines the interface that all model providers (OpenAI, Gemini, etc.)
    must implement to ensure consistent behavior across different models.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the client with API credentials.
        
        Args:
            api_key: API key for the model provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                image_data: Optional[str] = None,
                **kwargs) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: Text prompt for the model
            image_data: Base64 encoded image data (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement generate method")
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the current model.
        
        Returns:
            Model name string
        """
        raise NotImplementedError("Subclasses must implement get_model_name method")
    
    @abstractmethod
    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """Estimate the cost of a request.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Estimated cost in USD
        """
        raise NotImplementedError("Subclasses must implement estimate_cost method")


class BaseDataset(ABC):
    """Abstract base class for all dataset loaders.
    
    This class defines the interface for loading and processing different
    misinformation detection datasets.
    """
    
    def __init__(self, data_path: Union[str, Path], **kwargs):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the dataset files
            **kwargs: Additional dataset-specific configuration
        """
        self.data_path = Path(data_path)
        self.config = kwargs
        self._data = None
    
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load the dataset from files.
        
        Returns:
            List of dataset items, each as a dictionary
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement load method")
    
    @abstractmethod
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate a single dataset item.
        
        Args:
            item: Dataset item to validate
            
        Returns:
            True if item is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_item method")
    
    @abstractmethod
    def preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single dataset item.
        
        Args:
            item: Raw dataset item
            
        Returns:
            Preprocessed dataset item
        """
        raise NotImplementedError("Subclasses must implement preprocess_item method")
    
    def __len__(self) -> int:
        """Get the number of items in the dataset.
        
        Returns:
            Number of dataset items
        """
        if self._data is None:
            self._data = self.load()
        return len(self._data)
    
    def __iter__(self):
        """Iterate over dataset items.
        
        Yields:
            Preprocessed dataset items
        """
        if self._data is None:
            self._data = self.load()
        
        for item in self._data:
            if self.validate_item(item):
                yield self.preprocess_item(item)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self._data is None:
            self._data = self.load()
        
        return {
            'total_items': len(self._data),
            'valid_items': sum(1 for item in self._data if self.validate_item(item)),
            'data_path': str(self.data_path)
        }


class BasePipelineModule(ABC):
    """Abstract base class for all pipeline modules.
    
    This class defines the interface for processing components in the
    misinformation detection pipeline.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the pipeline module.
        
        Args:
            name: Name of the module
            config: Module configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the module with its configuration.
        
        This method should set up any required resources, validate
        configuration, and prepare the module for processing.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement initialize method")
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through this module.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed data dictionary
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_input method")
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema for this module.
        
        Returns:
            Dictionary describing the output schema
        """
        raise NotImplementedError("Subclasses must implement get_output_schema method")
    
    def ensure_initialized(self) -> None:
        """Ensure the module is initialized before processing.
        
        Raises:
            RuntimeError: If module initialization fails
        """
        if not self._initialized:
            try:
                self.initialize()
                self._initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to initialize module {self.name}: {e}")
    
    def safe_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely process data with error handling.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed data dictionary with error information if processing fails
        """
        try:
            self.ensure_initialized()
            
            if not self.validate_input(data):
                return {
                    **data,
                    'error': f'Invalid input for module {self.name}',
                    'module_status': 'failed'
                }
            
            result = self.process(data)
            result['module_status'] = 'success'
            return result
            
        except Exception as e:
            return {
                **data,
                'error': f'Module {self.name} failed: {str(e)}',
                'module_status': 'failed'
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information.
        
        Returns:
            Dictionary containing module metadata
        """
        return {
            'name': self.name,
            'config': self.config,
            'initialized': self._initialized,
            'output_schema': self.get_output_schema()
        }