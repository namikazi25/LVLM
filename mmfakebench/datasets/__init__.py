"""Dataset loaders for MisinfoBench.

This module provides dataset loading capabilities for various
misinformation detection datasets, along with data pipeline,
sampling, and augmentation utilities.
"""

from .base import BaseDataset
from .misinfobench import MisinfoBenchDataset
from .mocheg import MOCHEGDataset
from .sampling import (
    BaseSampler, RandomSampler, StratifiedSampler, 
    BalancedSampler, SequentialSampler, create_sampler
)
from .augmentation import (
    BaseAugmentation, TextAugmentation, LabelPreservingAugmentation,
    CompositeAugmentation, create_augmentation_pipeline
)
from .pipeline import DataPipeline, create_pipeline_from_config, create_dataset

__all__ = [
    # Base classes
    'BaseDataset',
    # Dataset implementations
    'MisinfoBenchDataset', 
    'MOCHEGDataset',
    # Sampling
    'BaseSampler',
    'RandomSampler',
    'StratifiedSampler',
    'BalancedSampler', 
    'SequentialSampler',
    'create_sampler',
    # Augmentation
    'BaseAugmentation',
    'TextAugmentation',
    'LabelPreservingAugmentation',
    'CompositeAugmentation',
    'create_augmentation_pipeline',
    # Pipeline
    'DataPipeline',
    'create_pipeline_from_config',
    'create_dataset'
]