"""Data sampling strategies for MMFakeBench datasets.

This module provides various sampling strategies for creating subsets
of datasets for training, validation, and testing.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict, Counter
import logging
import math


class BaseSampler(ABC):
    """Abstract base class for data samplers."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the sampler.
        
        Args:
            random_seed: Random seed for reproducible sampling
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def sample(self, data: List[Dict[str, Any]], sample_size: Union[int, float]) -> List[Dict[str, Any]]:
        """Sample data from the input dataset.
        
        Args:
            data: List of dataset items
            sample_size: Number of items to sample (int) or fraction (float)
            
        Returns:
            Sampled subset of data
        """
        raise NotImplementedError("Subclasses must implement sample method")
    
    def _get_sample_count(self, data_size: int, sample_size: Union[int, float]) -> int:
        """Convert sample_size to actual count.
        
        Args:
            data_size: Total size of the dataset
            sample_size: Requested sample size (int or float)
            
        Returns:
            Actual number of samples to take
        """
        if isinstance(sample_size, float):
            if not 0.0 <= sample_size <= 1.0:
                raise ValueError(f"Sample size as fraction must be between 0 and 1, got {sample_size}")
            return int(data_size * sample_size)
        elif isinstance(sample_size, int):
            if sample_size < 0:
                raise ValueError(f"Sample size must be non-negative, got {sample_size}")
            return min(sample_size, data_size)
        else:
            raise TypeError(f"Sample size must be int or float, got {type(sample_size)}")


class RandomSampler(BaseSampler):
    """Random sampling without replacement."""
    
    def sample(self, data: List[Dict[str, Any]], sample_size: Union[int, float]) -> List[Dict[str, Any]]:
        """Randomly sample data without replacement.
        
        Args:
            data: List of dataset items
            sample_size: Number of items to sample (int) or fraction (float)
            
        Returns:
            Randomly sampled subset of data
        """
        if not data:
            return []
        
        sample_count = self._get_sample_count(len(data), sample_size)
        
        if sample_count == 0:
            return []
        
        if sample_count >= len(data):
            return data.copy()
        
        sampled_data = random.sample(data, sample_count)
        self.logger.info(f"Randomly sampled {len(sampled_data)} items from {len(data)} total items")
        
        return sampled_data


class StratifiedSampler(BaseSampler):
    """Stratified sampling to maintain class distribution."""
    
    def __init__(self, label_key: str = 'label_binary', random_seed: Optional[int] = None):
        """Initialize stratified sampler.
        
        Args:
            label_key: Key in data items that contains the label for stratification
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(random_seed)
        self.label_key = label_key
    
    def sample(self, data: List[Dict[str, Any]], sample_size: Union[int, float]) -> List[Dict[str, Any]]:
        """Sample data while maintaining class distribution.
        
        Args:
            data: List of dataset items
            sample_size: Number of items to sample (int) or fraction (float)
            
        Returns:
            Stratified sampled subset of data
        """
        if not data:
            return []
        
        sample_count = self._get_sample_count(len(data), sample_size)
        
        if sample_count == 0:
            return []
        
        if sample_count >= len(data):
            return data.copy()
        
        # Group data by labels
        label_groups = defaultdict(list)
        for item in data:
            label = item.get(self.label_key, 'unknown')
            label_groups[label].append(item)
        
        # Calculate samples per class
        total_items = len(data)
        sampled_data = []
        
        for label, items in label_groups.items():
            # Proportional sampling based on class distribution
            class_proportion = len(items) / total_items
            class_sample_count = max(1, int(sample_count * class_proportion))
            
            # Don't sample more than available
            class_sample_count = min(class_sample_count, len(items))
            
            if class_sample_count > 0:
                class_samples = random.sample(items, class_sample_count)
                sampled_data.extend(class_samples)
        
        # If we haven't reached the target sample size, randomly add more
        if len(sampled_data) < sample_count:
            remaining_items = [item for item in data if item not in sampled_data]
            additional_count = min(sample_count - len(sampled_data), len(remaining_items))
            if additional_count > 0:
                additional_samples = random.sample(remaining_items, additional_count)
                sampled_data.extend(additional_samples)
        
        # If we have too many samples, randomly remove some
        elif len(sampled_data) > sample_count:
            sampled_data = random.sample(sampled_data, sample_count)
        
        self.logger.info(f"Stratified sampled {len(sampled_data)} items from {len(data)} total items")
        self._log_class_distribution(sampled_data)
        
        return sampled_data
    
    def _log_class_distribution(self, data: List[Dict[str, Any]]) -> None:
        """Log the class distribution of sampled data."""
        label_counts = Counter(item.get(self.label_key, 'unknown') for item in data)
        self.logger.info(f"Class distribution: {dict(label_counts)}")


class BalancedSampler(BaseSampler):
    """Balanced sampling to ensure equal representation of all classes."""
    
    def __init__(self, label_key: str = 'label_binary', random_seed: Optional[int] = None):
        """Initialize balanced sampler.
        
        Args:
            label_key: Key in data items that contains the label for balancing
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(random_seed)
        self.label_key = label_key
    
    def sample(self, data: List[Dict[str, Any]], sample_size: Union[int, float]) -> List[Dict[str, Any]]:
        """Sample data with equal representation from each class.
        
        Args:
            data: List of dataset items
            sample_size: Number of items to sample (int) or fraction (float)
            
        Returns:
            Balanced sampled subset of data
        """
        if not data:
            return []
        
        sample_count = self._get_sample_count(len(data), sample_size)
        
        if sample_count == 0:
            return []
        
        # Group data by labels
        label_groups = defaultdict(list)
        for item in data:
            label = item.get(self.label_key, 'unknown')
            label_groups[label].append(item)
        
        if not label_groups:
            return []
        
        # Calculate samples per class (equal for all classes)
        num_classes = len(label_groups)
        samples_per_class = sample_count // num_classes
        
        sampled_data = []
        
        for label, items in label_groups.items():
            # Sample equally from each class
            class_sample_count = min(samples_per_class, len(items))
            
            if class_sample_count > 0:
                class_samples = random.sample(items, class_sample_count)
                sampled_data.extend(class_samples)
        
        # If we have remaining quota, distribute among classes with more data
        remaining_quota = sample_count - len(sampled_data)
        if remaining_quota > 0:
            # Sort classes by available remaining items
            available_items = []
            for label, items in label_groups.items():
                used_items = [item for item in sampled_data if item.get(self.label_key) == label]
                remaining_items = [item for item in items if item not in used_items]
                if remaining_items:
                    available_items.extend(remaining_items)
            
            if available_items:
                additional_count = min(remaining_quota, len(available_items))
                additional_samples = random.sample(available_items, additional_count)
                sampled_data.extend(additional_samples)
        
        self.logger.info(f"Balanced sampled {len(sampled_data)} items from {len(data)} total items")
        self._log_class_distribution(sampled_data)
        
        return sampled_data
    
    def _log_class_distribution(self, data: List[Dict[str, Any]]) -> None:
        """Log the class distribution of sampled data."""
        label_counts = Counter(item.get(self.label_key, 'unknown') for item in data)
        self.logger.info(f"Class distribution: {dict(label_counts)}")


class SequentialSampler(BaseSampler):
    """Sequential sampling (first N items)."""
    
    def sample(self, data: List[Dict[str, Any]], sample_size: Union[int, float]) -> List[Dict[str, Any]]:
        """Sample the first N items sequentially.
        
        Args:
            data: List of dataset items
            sample_size: Number of items to sample (int) or fraction (float)
            
        Returns:
            First N items from the dataset
        """
        if not data:
            return []
        
        sample_count = self._get_sample_count(len(data), sample_size)
        
        sampled_data = data[:sample_count]
        self.logger.info(f"Sequentially sampled {len(sampled_data)} items from {len(data)} total items")
        
        return sampled_data


def create_sampler(strategy: str, **kwargs) -> BaseSampler:
    """Factory function to create samplers.
    
    Args:
        strategy: Sampling strategy ('random', 'stratified', 'balanced', 'sequential')
        **kwargs: Additional arguments for the sampler
        
    Returns:
        Configured sampler instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    strategy = strategy.lower()
    
    if strategy == 'random':
        return RandomSampler(**kwargs)
    elif strategy == 'stratified':
        return StratifiedSampler(**kwargs)
    elif strategy == 'balanced':
        return BalancedSampler(**kwargs)
    elif strategy == 'sequential':
        return SequentialSampler(**kwargs)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}. "
                        f"Supported strategies: random, stratified, balanced, sequential")