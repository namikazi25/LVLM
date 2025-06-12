"""MMFakeBench dataset loader.

This module provides the dataset loader for the MMFakeBench dataset,
which contains multimodal misinformation detection data.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import logging

from .base import BaseDataset


class MMFakeBenchDataset(BaseDataset):
    """Dataset loader for MMFakeBench dataset.
    
    The MMFakeBench dataset contains image-text pairs with binary and
    multiclass labels for misinformation detection.
    """
    
    def __init__(self, data_path: Union[str, Path], images_base_dir: Optional[Union[str, Path]] = None, 
                 limit: Optional[int] = None, **kwargs):
        """Initialize MMFakeBench dataset.
        
        Args:
            data_path: Path to the JSON file containing dataset annotations
            images_base_dir: Base directory where images are stored. If None, 
                           assumes images are relative to data_path directory
            limit: Optional limit on number of samples to load
            **kwargs: Additional configuration options
        """
        super().__init__(data_path, **kwargs)
        
        if images_base_dir is None:
            self.images_base_dir = self.data_path.parent
        else:
            self.images_base_dir = Path(images_base_dir)
        
        self.limit = limit
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self) -> List[Dict[str, Any]]:
        """Load the MMFakeBench dataset from JSON file.
        
        Returns:
            List of dataset items
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in dataset file: {e}")
        
        # Apply limit if specified
        if self.limit:
            data = data[:self.limit]
            self.logger.info(f"Limited dataset to {self.limit} items")
        
        items = []
        for entry in data:
            # Handle image path - make it absolute
            image_path = entry.get('image_path', '')
            if image_path:
                # Remove leading slash if present and join with base directory
                image_path = os.path.join(self.images_base_dir, image_path.lstrip('/'))
                
                # Only include items where image file exists
                if os.path.exists(image_path):
                    items.append({
                        'image_path': image_path,
                        'text': entry.get('text', ''),
                        'label_binary': entry.get('gt_answers', ''),  # "True" or "Fake"
                        'label_multiclass': entry.get('fake_cls', ''),  # "original", "mismatch", etc.
                        'text_source': entry.get('text_source', ''),
                        'image_source': entry.get('image_source', ''),
                        'original_entry': entry  # Keep original for reference
                    })
                else:
                    self.logger.warning(f"Image file not found: {image_path}")
            else:
                self.logger.warning(f"Missing image_path in entry: {entry}")
        
        self.logger.info(f"Loaded {len(items)} valid items from {len(data)} total entries")
        return items
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate a single dataset item.
        
        Args:
            item: Dataset item to validate
            
        Returns:
            True if item is valid, False otherwise
        """
        required_fields = ['image_path', 'text', 'label_binary']
        
        # Check required fields
        for field in required_fields:
            if field not in item or not item[field]:
                self.logger.debug(f"Missing or empty required field: {field}")
                return False
        
        # Check if image file exists
        if not os.path.exists(item['image_path']):
            self.logger.debug(f"Image file does not exist: {item['image_path']}")
            return False
        
        # Validate label values
        valid_binary_labels = ['True', 'Fake', 'true', 'fake', 'TRUE', 'FAKE']
        if item['label_binary'] not in valid_binary_labels:
            self.logger.debug(f"Invalid binary label: {item['label_binary']}")
            return False
        
        return True
    
    def preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single dataset item.
        
        Args:
            item: Raw dataset item
            
        Returns:
            Preprocessed dataset item
        """
        processed_item = item.copy()
        
        # Normalize text
        processed_item['text'] = processed_item['text'].strip()
        
        # Add headline field (map from text for compatibility)
        processed_item['headline'] = processed_item['text']
        
        # Normalize binary label to consistent format
        binary_label = processed_item['label_binary'].lower()
        processed_item['label_binary'] = 'fake' if binary_label in ['fake', 'false'] else 'true'
        
        # Ensure image path is absolute
        processed_item['image_path'] = os.path.abspath(processed_item['image_path'])
        
        # Add metadata
        processed_item['dataset_name'] = 'mmfakebench'
        processed_item['item_id'] = f"mmfb_{hash(processed_item['image_path'] + processed_item['text']) % 1000000}"
        
        return processed_item
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics specific to MMFakeBench.
        
        Returns:
            Dictionary containing dataset statistics
        """
        base_stats = super().get_statistics()
        
        if self._data is None:
            self._data = self.load()
        
        # Count labels
        binary_labels = {}
        multiclass_labels = {}
        text_sources = {}
        image_sources = {}
        
        valid_items = [item for item in self._data if self.validate_item(item)]
        
        for item in valid_items:
            # Binary labels
            binary_label = item.get('label_binary', 'unknown')
            binary_labels[binary_label] = binary_labels.get(binary_label, 0) + 1
            
            # Multiclass labels
            multiclass_label = item.get('label_multiclass', 'unknown')
            multiclass_labels[multiclass_label] = multiclass_labels.get(multiclass_label, 0) + 1
            
            # Sources
            text_source = item.get('text_source', 'unknown')
            text_sources[text_source] = text_sources.get(text_source, 0) + 1
            
            image_source = item.get('image_source', 'unknown')
            image_sources[image_source] = image_sources.get(image_source, 0) + 1
        
        base_stats.update({
            'binary_label_distribution': binary_labels,
            'multiclass_label_distribution': multiclass_labels,
            'text_source_distribution': text_sources,
            'image_source_distribution': image_sources,
            'images_base_dir': str(self.images_base_dir),
            'limit': self.limit
        })
        
        return base_stats