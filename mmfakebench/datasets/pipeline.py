"""Data pipeline for MisinfoBench datasets.

This module provides a comprehensive data pipeline that integrates
data loading, sampling, augmentation, and analysis capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from collections import Counter, defaultdict
import random

from .base import BaseDataset
from .sampling import BaseSampler, create_sampler
from .augmentation import BaseAugmentation, create_augmentation_pipeline


class DataPipeline:
    """Comprehensive data pipeline for dataset processing."""
    
    def __init__(self, 
                 dataset: BaseDataset,
                 sampler: Optional[BaseSampler] = None,
                 sample_size: Union[int, float] = 1.0,
                 augmentation: Optional[BaseAugmentation] = None,
                 cache_enabled: bool = True,
                 random_seed: Optional[int] = None):
        """Initialize the data pipeline.
        
        Args:
            dataset: Base dataset to process
            sampler: Data sampler for subset selection
            sample_size: Size for sampling (int for count, float for fraction)
            augmentation: Data augmentation pipeline
            cache_enabled: Whether to cache processed data
            random_seed: Random seed for reproducible processing
        """
        self.dataset = dataset
        self.sampler = sampler
        self.sample_size = sample_size
        self.augmentation = augmentation
        self.cache_enabled = cache_enabled
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = {}
        self._processed_data = None
        self._statistics = None
    
    def process(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Process the dataset through the pipeline.
        
        Args:
            force_reload: Whether to force reprocessing even if cached
            
        Returns:
            List of processed data items
        """
        if self._processed_data is not None and not force_reload:
            return self._processed_data
        
        self.logger.info("Processing dataset through pipeline")
        
        # Load and preprocess data
        if self.dataset._data is None:
            self.dataset._data = self.dataset.load()
        
        # Preprocess all valid items
        processed_data = []
        for item in self.dataset._data:
            if self.dataset.validate_item(item):
                processed_item = self.dataset.preprocess_item(item)
                processed_data.append(processed_item)
            else:
                self.logger.warning(f"Invalid item skipped during preprocessing")
        self.logger.info(f"Loaded {len(processed_data)} items from dataset")
        
        # Apply sampling
        if self.sampler is not None:
            processed_data = self.sampler.sample(processed_data, self.sample_size)
            self.logger.info(f"Sampled {len(processed_data)} items using {self.sampler.__class__.__name__}")
        
        # Apply augmentation
        if self.augmentation is not None:
            augmented_data = []
            for item in processed_data:
                augmented_item = self.augmentation(item)
                augmented_data.append(augmented_item)
            processed_data = augmented_data
            self.logger.info(f"Applied augmentation to {len(processed_data)} items")
        
        self._processed_data = processed_data
        self._statistics = None  # Reset statistics cache
        
        return processed_data
    
    def get_statistics(self, force_recalculate: bool = False) -> Dict[str, Any]:
        """Get comprehensive statistics about the processed dataset.
        
        Args:
            force_recalculate: Whether to force recalculation of statistics
            
        Returns:
            Dictionary containing dataset statistics
        """
        if self._statistics is not None and not force_recalculate:
            return self._statistics
        
        if self._processed_data is None:
            self.process()
        
        stats = {
            'total_items': len(self._processed_data),
            'pipeline_config': {
                'dataset_type': self.dataset.__class__.__name__,
                'sampler_type': self.sampler.__class__.__name__ if self.sampler else None,
                'augmentation_enabled': self.augmentation is not None,
                'cache_enabled': self.cache_enabled,
                'random_seed': self.random_seed
            }
        }
        
        # Label distribution
        label_counts = Counter()
        binary_label_counts = Counter()
        multiclass_label_counts = Counter()
        
        # Text statistics
        text_lengths = []
        has_text_count = 0
        
        # Image statistics
        has_image_count = 0
        image_extensions = Counter()
        
        # Augmentation statistics
        augmented_count = 0
        augmentation_types = Counter()
        
        for item in self._processed_data:
            # Label analysis
            if 'label' in item:
                label_counts[item['label']] += 1
            if 'label_binary' in item:
                binary_label_counts[item['label_binary']] += 1
            if 'label_multiclass' in item:
                multiclass_label_counts[item['label_multiclass']] += 1
            
            # Text analysis
            if 'text' in item and item['text']:
                has_text_count += 1
                text_lengths.append(len(item['text']))
            
            # Image analysis
            if 'image_path' in item and item['image_path']:
                has_image_count += 1
                image_path = Path(item['image_path'])
                image_extensions[image_path.suffix.lower()] += 1
            
            # Augmentation analysis
            if item.get('is_augmented', False) or item.get('augmentation_applied', False):
                augmented_count += 1
                if 'augmentation_type' in item:
                    augmentation_types[item['augmentation_type']] += 1
        
        # Compile statistics
        stats.update({
            'labels': {
                'distribution': dict(label_counts),
                'binary_distribution': dict(binary_label_counts),
                'multiclass_distribution': dict(multiclass_label_counts),
                'unique_labels': len(label_counts),
                'unique_binary_labels': len(binary_label_counts),
                'unique_multiclass_labels': len(multiclass_label_counts)
            },
            'text': {
                'items_with_text': has_text_count,
                'text_coverage': has_text_count / len(self._processed_data) if self._processed_data else 0,
                'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                'min_text_length': min(text_lengths) if text_lengths else 0,
                'max_text_length': max(text_lengths) if text_lengths else 0
            },
            'images': {
                'items_with_images': has_image_count,
                'image_coverage': has_image_count / len(self._processed_data) if self._processed_data else 0,
                'image_extensions': dict(image_extensions)
            },
            'augmentation': {
                'augmented_items': augmented_count,
                'augmentation_rate': augmented_count / len(self._processed_data) if self._processed_data else 0,
                'augmentation_types': dict(augmentation_types)
            }
        })
        
        # Add dataset-specific statistics
        if hasattr(self.dataset, 'get_statistics'):
            dataset_stats = self.dataset.get_statistics()
            stats['dataset_specific'] = dataset_stats
        
        self._statistics = stats
        return stats
    
    def preview_data(self, num_samples: int = 5, include_augmented: bool = True) -> List[Dict[str, Any]]:
        """Get a preview of processed data items.
        
        Args:
            num_samples: Number of samples to include in preview
            include_augmented: Whether to include augmented items in preview
            
        Returns:
            List of sample data items for preview
        """
        if self._processed_data is None:
            self.process()
        
        preview_items = []
        available_items = self._processed_data.copy()
        
        if not include_augmented:
            available_items = [
                item for item in available_items 
                if not item.get('is_augmented', False) and not item.get('augmentation_applied', False)
            ]
        
        # Sample items for preview
        sample_size = min(num_samples, len(available_items))
        if sample_size > 0:
            sampled_items = random.sample(available_items, sample_size)
            
            for item in sampled_items:
                preview_item = {
                    'id': item.get('id', 'unknown'),
                    'text_preview': item.get('text', '')[:100] + '...' if item.get('text', '') else 'No text',
                    'image_path': item.get('image_path', 'No image'),
                    'label': item.get('label', 'No label'),
                    'label_binary': item.get('label_binary', 'No binary label'),
                    'is_augmented': item.get('is_augmented', False),
                    'augmentation_type': item.get('augmentation_type', 'None')
                }
                preview_items.append(preview_item)
        
        return preview_items
    
    def export_statistics(self, output_path: Union[str, Path]) -> None:
        """Export statistics to a JSON file.
        
        Args:
            output_path: Path to save the statistics file
        """
        stats = self.get_statistics()
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Statistics exported to {output_path}")
    
    def export_preview(self, output_path: Union[str, Path], num_samples: int = 10) -> None:
        """Export data preview to a JSON file.
        
        Args:
            output_path: Path to save the preview file
            num_samples: Number of samples to include in preview
        """
        preview = self.preview_data(num_samples=num_samples)
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(preview, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Data preview exported to {output_path}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over processed data items."""
        if self._processed_data is None:
            self.process()
        
        for item in self._processed_data:
            yield item
    
    def __len__(self) -> int:
        """Get the number of processed data items."""
        if self._processed_data is None:
            self.process()
        
        return len(self._processed_data)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a specific processed data item by index."""
        if self._processed_data is None:
            self.process()
        
        return self._processed_data[index]


def create_dataset(config: Dict[str, Any]) -> BaseDataset:
    """Create a dataset from configuration.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        Configured dataset instance
    """
    # Import dataset classes
    from .misinfobench import MisinfoBenchDataset
    from .mocheg import MOCHEGDataset
    
    dataset_type = config['type'].lower()
    
    if dataset_type == 'misinfobench':
        dataset = MisinfoBenchDataset(
            data_path=config['data_path'],
            image_dir=config.get('image_dir')
        )
    elif dataset_type == 'mocheg':
        dataset = MOCHEGDataset(
            data_path=config['data_path'],
            image_dir=config.get('image_dir')
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return dataset


def create_pipeline_from_config(config: Dict[str, Any]) -> DataPipeline:
    """Create a data pipeline from configuration.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        Configured DataPipeline instance
    """
    # Import dataset classes
    from .misinfobench import MisinfoBenchDataset
    from .mocheg import MOCHEGDataset
    
    # Create dataset
    dataset_config = config['dataset']
    dataset_type = dataset_config['type'].lower()
    
    if dataset_type == 'misinfobench':
        dataset = MisinfoBenchDataset(
            data_path=dataset_config['data_path'],
            image_dir=dataset_config.get('image_dir')
        )
    elif dataset_type == 'mocheg':
        dataset = MOCHEGDataset(
            data_path=dataset_config['data_path'],
            image_dir=dataset_config.get('image_dir')
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Create sampler
    sampler = None
    if 'sampling' in config and config['sampling'].get('enabled', False):
        sampling_config = config['sampling']
        sampler_type = sampling_config.get('type', 'random')
        sampler_kwargs = {
            'random_seed': sampling_config.get('random_seed', None)
        }
        # Add label_key for stratified and balanced samplers
        if sampler_type in ['stratified', 'balanced']:
            sampler_kwargs['label_key'] = sampling_config.get('label_key', 'gt_answers')
        
        sampler = create_sampler(sampler_type, **sampler_kwargs)
    
    # Create augmentation
    augmentation = None
    if 'augmentation' in config:
        augmentation = create_augmentation_pipeline(config['augmentation'])
    
    # Create pipeline
    sample_size = 1.0  # Default to full dataset
    if 'sampling' in config and config['sampling'].get('enabled', False):
        sample_size = config['sampling'].get('sample_size', 1.0)
    
    pipeline = DataPipeline(
        dataset=dataset,
        sampler=sampler,
        sample_size=sample_size,
        augmentation=augmentation,
        cache_enabled=config.get('cache_enabled', True),
        random_seed=config.get('random_seed')
    )
    
    return pipeline