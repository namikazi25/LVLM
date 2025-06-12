"""Base dataset interface for MMFakeBench.

This module provides the abstract base class for all dataset loaders
in the MMFakeBench framework.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import logging


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
        self.logger = logging.getLogger(self.__class__.__name__)
    
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
            else:
                self.logger.warning(f"Invalid item skipped: {item}")
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """Get a specific item by index.
        
        Args:
            index: Item index
            
        Returns:
            Preprocessed dataset item
            
        Raises:
            IndexError: If index is out of range
        """
        if self._data is None:
            self._data = self.load()
        
        if index < 0 or index >= len(self._data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._data)}")
        
        item = self._data[index]
        if not self.validate_item(item):
            raise ValueError(f"Item at index {index} is invalid")
        
        return self.preprocess_item(item)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self._data is None:
            self._data = self.load()
        
        valid_items = [item for item in self._data if self.validate_item(item)]
        
        return {
            'total_items': len(self._data),
            'valid_items': len(valid_items),
            'invalid_items': len(self._data) - len(valid_items),
            'data_path': str(self.data_path),
            'config': self.config
        }
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the entire dataset.
        
        Returns:
            Validation report with statistics and errors
        """
        if self._data is None:
            self._data = self.load()
        
        validation_errors = []
        valid_count = 0
        
        for i, item in enumerate(self._data):
            try:
                if self.validate_item(item):
                    valid_count += 1
                else:
                    validation_errors.append(f"Item {i}: Validation failed")
            except Exception as e:
                validation_errors.append(f"Item {i}: {str(e)}")
        
        return {
            'total_items': len(self._data),
            'valid_items': valid_count,
            'invalid_items': len(self._data) - valid_count,
            'validation_errors': validation_errors[:10],  # Limit to first 10 errors
            'error_count': len(validation_errors)
        }