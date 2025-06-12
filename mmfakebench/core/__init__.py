"""Core components for MMFakeBench.

This module contains the fundamental building blocks of the benchmarking toolkit,
including base classes, configuration management, and pipeline orchestration.
"""

from .base import BaseClient, BaseDataset, BasePipelineModule
from .runner import BenchmarkRunner, DatasetPreviewer
from .pipeline import PipelineManager, Pipeline, PipelineBuilder
from .io import ResultsWriter, Logger, ConfigLoader, FileManager
from .config import ConfigManager, ConfigValidator
from .registry import ModuleRegistry, registry

__all__ = [
    'BaseClient',
    'BaseDataset', 
    'BasePipelineModule',
    'BenchmarkRunner',
    'DatasetPreviewer',
    'PipelineManager',
    'Pipeline',
    'PipelineBuilder',
    'ResultsWriter',
    'Logger',
    'ConfigLoader',
    'FileManager',
    'ConfigManager',
    'ConfigValidator',
    'ModuleRegistry',
    'registry'
]