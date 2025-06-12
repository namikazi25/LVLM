"""MOCHEG dataset loader.

This module provides the dataset loader for the MOCHEG (Multimodal Out-of-Context
Harmful mEme Generation) dataset for misinformation detection.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import logging

from .base import BaseDataset


class MOCHEGDataset(BaseDataset):
    """Dataset loader for MOCHEG dataset.
    
    The MOCHEG dataset contains multimodal data for detecting out-of-context
    harmful memes and misinformation.
    """
    
    def __init__(self, data_path: Union[str, Path], images_base_dir: Optional[Union[str, Path]] = None,
                 split: str = 'test', limit: Optional[int] = None, **kwargs):
        """Initialize MOCHEG dataset.
        
        Args:
            data_path: Path to the dataset directory or JSON file
            images_base_dir: Base directory where images are stored. If None,
                           assumes images are relative to data_path directory
            split: Dataset split to load ('train', 'val', 'test')
            limit: Optional limit on number of samples to load
            **kwargs: Additional configuration options
        """
        super().__init__(data_path, **kwargs)
        
        if images_base_dir is None:
            if self.data_path.is_file():
                self.images_base_dir = self.data_path.parent
            else:
                self.images_base_dir = self.data_path
        else:
            self.images_base_dir = Path(images_base_dir)
        
        self.split = split
        self.limit = limit
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self) -> List[Dict[str, Any]]:
        """Load the MOCHEG dataset.
        
        Returns:
            List of dataset items
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        # Determine the actual data file path
        if self.data_path.is_file():
            data_file = self.data_path
        else:
            # Look for split-specific files
            possible_files = [
                self.data_path / f"{self.split}.json",
                self.data_path / f"mocheg_{self.split}.json",
                self.data_path / f"MOCHEG_{self.split}.json",
                self.data_path / "annotations.json",
                self.data_path / "data.json"
            ]
            
            data_file = None
            for file_path in possible_files:
                if file_path.exists():
                    data_file = file_path
                    break
            
            if data_file is None:
                raise FileNotFoundError(
                    f"No valid dataset file found in {self.data_path}. "
                    f"Looked for: {[str(f) for f in possible_files]}"
                )
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in dataset file: {e}")
        
        # Handle different data structures
        if isinstance(data, dict):
            if self.split in data:
                data = data[self.split]
            elif 'data' in data:
                data = data['data']
            elif 'annotations' in data:
                data = data['annotations']
            else:
                # Assume the dict values are the data items
                data = list(data.values())
        
        # Apply limit if specified
        if self.limit:
            data = data[:self.limit]
            self.logger.info(f"Limited dataset to {self.limit} items")
        
        items = []
        for i, entry in enumerate(data):
            try:
                # Handle different field names that might exist in MOCHEG
                image_path = self._get_image_path(entry)
                text = self._get_text(entry)
                label = self._get_label(entry)
                
                if image_path and text is not None and label is not None:
                    # Make image path absolute
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(self.images_base_dir, image_path)
                    
                    # Only include items where image file exists
                    if os.path.exists(image_path):
                        items.append({
                            'image_path': image_path,
                            'text': text,
                            'label': label,
                            'label_binary': self._normalize_binary_label(label),
                            'item_index': i,
                            'split': self.split,
                            'original_entry': entry
                        })
                    else:
                        self.logger.warning(f"Image file not found: {image_path}")
                else:
                    self.logger.warning(f"Incomplete entry at index {i}: {entry}")
            except Exception as e:
                self.logger.warning(f"Error processing entry {i}: {e}")
        
        self.logger.info(f"Loaded {len(items)} valid items from {len(data)} total entries")
        return items
    
    def _get_image_path(self, entry: Dict[str, Any]) -> Optional[str]:
        """Extract image path from entry, handling different field names."""
        possible_fields = ['image_path', 'image', 'img_path', 'image_file', 'img']
        for field in possible_fields:
            if field in entry and entry[field]:
                return entry[field]
        return None
    
    def _get_text(self, entry: Dict[str, Any]) -> Optional[str]:
        """Extract text from entry, handling different field names."""
        possible_fields = ['text', 'caption', 'description', 'content', 'meme_text']
        for field in possible_fields:
            if field in entry and entry[field] is not None:
                return str(entry[field])
        return None
    
    def _get_label(self, entry: Dict[str, Any]) -> Optional[str]:
        """Extract label from entry, handling different field names."""
        possible_fields = ['label', 'gt_label', 'ground_truth', 'annotation', 'class']
        for field in possible_fields:
            if field in entry and entry[field] is not None:
                return str(entry[field])
        return None
    
    def _normalize_binary_label(self, label: str) -> str:
        """Normalize label to binary format."""
        label_lower = label.lower().strip()
        
        # Map various label formats to binary
        fake_indicators = ['fake', 'false', 'harmful', 'out-of-context', 'misleading', '1', 'positive']
        true_indicators = ['true', 'real', 'authentic', 'in-context', 'safe', '0', 'negative']
        
        if any(indicator in label_lower for indicator in fake_indicators):
            return 'fake'
        elif any(indicator in label_lower for indicator in true_indicators):
            return 'true'
        else:
            # Default mapping - you may need to adjust based on actual MOCHEG labels
            return 'fake' if label_lower in ['1', 'positive'] else 'true'
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate a single dataset item.
        
        Args:
            item: Dataset item to validate
            
        Returns:
            True if item is valid, False otherwise
        """
        required_fields = ['image_path', 'text', 'label']
        
        # Check required fields
        for field in required_fields:
            if field not in item or item[field] is None:
                self.logger.debug(f"Missing or None required field: {field}")
                return False
        
        # Check if text is not empty
        if not str(item['text']).strip():
            self.logger.debug("Empty text field")
            return False
        
        # Check if image file exists
        if not os.path.exists(item['image_path']):
            self.logger.debug(f"Image file does not exist: {item['image_path']}")
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
        processed_item['text'] = str(processed_item['text']).strip()
        
        # Ensure image path is absolute
        processed_item['image_path'] = os.path.abspath(processed_item['image_path'])
        
        # Add metadata
        processed_item['dataset_name'] = 'mocheg'
        processed_item['item_id'] = f"mocheg_{processed_item.get('item_index', 0)}_{self.split}"
        
        return processed_item
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics specific to MOCHEG.
        
        Returns:
            Dictionary containing dataset statistics
        """
        base_stats = super().get_statistics()
        
        if self._data is None:
            self._data = self.load()
        
        # Count labels
        labels = {}
        binary_labels = {}
        
        valid_items = [item for item in self._data if self.validate_item(item)]
        
        for item in valid_items:
            # Original labels
            label = item.get('label', 'unknown')
            labels[label] = labels.get(label, 0) + 1
            
            # Binary labels
            binary_label = item.get('label_binary', 'unknown')
            binary_labels[binary_label] = binary_labels.get(binary_label, 0) + 1
        
        base_stats.update({
            'label_distribution': labels,
            'binary_label_distribution': binary_labels,
            'split': self.split,
            'images_base_dir': str(self.images_base_dir),
            'limit': self.limit
        })
        
        return base_stats