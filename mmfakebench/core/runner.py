"""Benchmark runner for orchestrating evaluation pipelines.

This module provides the main BenchmarkRunner class that coordinates
the execution of misinformation detection benchmarks.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseDataset, BasePipelineModule
from .pipeline import PipelineManager
from .io import ResultsWriter, Logger


class BenchmarkRunner:
    """Main orchestrator for running misinformation detection benchmarks.
    
    This class coordinates the execution of evaluation pipelines,
    manages datasets, and handles result collection and output.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 output_dir: Optional[Union[str, Path]] = None):
        """Initialize the benchmark runner.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for output files
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = Logger(self.output_dir / "benchmark.log")
        self.results_writer = ResultsWriter(self.output_dir)
        self.pipeline_manager = PipelineManager()
        
        # Runtime state
        self.model_router = None
        self.dataset = None
        self.pipeline_modules = []
        self.results = []
        
        # Runtime options
        self.sample_limit = None
        self.checkpoint_file = None
        
        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'total_cost': 0.0
        }
    
    def setup_model_router(self, model_config: Dict[str, Any]) -> None:
        """Setup the model router with configuration.
        
        Args:
            model_config: Model configuration dictionary
        """
        try:
            import os
            from models.router import ModelRouter
            
            # Get API key from config or environment
            api_key = (
                model_config.get('api_key') or 
                model_config.get('openai_api_key') or
                os.getenv('OPENAI_API_KEY') or
                'test-key-for-demo'  # Fallback for testing
            )
            
            self.model_router = ModelRouter(
                model_name=model_config['name'],
                api_key=api_key,
                temperature=model_config.get('temperature', 0.2),
                max_retries=model_config.get('max_retries', 5),
                **model_config.get('additional_params', {})
            )
            logging.info(f"Model router initialized with {model_config['name']}")
        except Exception as e:
            logging.error(f"Failed to setup model router: {e}")
            raise
    
    def setup_dataset(self, dataset_config: Dict[str, Any]) -> None:
        """Setup the dataset with configuration.
        
        Args:
            dataset_config: Dataset configuration dictionary
        """
        try:
            dataset_type = dataset_config['type']
            
            if dataset_type == 'misinfobench':
                from datasets.misinfobench import MisinfoBenchDataset
                self.dataset = MisinfoBenchDataset(
                    data_path=dataset_config['data_path'],
                    **dataset_config.get('params', {})
                )
            elif dataset_type == "mocheg":
                from datasets.mocheg import MOCHEGDataset
                self.dataset = MOCHEGDataset(
                    data_path=dataset_config['data_path'],
                    **dataset_config.get('params', {})
                )
            elif dataset_type == 'custom':
                from datasets.mmfakebench import MMFakeBenchDataset
                self.dataset = MMFakeBenchDataset(
                    data_path=dataset_config['data_path'],
                    **dataset_config.get('params', {})
                )
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
            logging.info(f"Dataset initialized: {dataset_type}")
            self.stats['total_items'] = len(self.dataset)
            
        except Exception as e:
            logging.error(f"Failed to setup dataset: {e}")
            raise
    
    def setup_pipeline(self, pipeline_config: List[Dict[str, Any]]) -> None:
        """Setup the processing pipeline.
        
        Args:
            pipeline_config: List of pipeline module configurations
        """
        try:
            self.pipeline_modules = []
            
            for module_config in pipeline_config:
                module = self.pipeline_manager.create_module(
                    module_config['type'],
                    module_config['name'],
                    module_config.get('config', {})
                )
                
                # Inject model router if module needs it
                if hasattr(module, 'set_model_router'):
                    module.set_model_router(self.model_router)
                
                self.pipeline_modules.append(module)
            
            logging.info(f"Pipeline initialized with {len(self.pipeline_modules)} modules")
            
        except Exception as e:
            logging.error(f"Failed to setup pipeline: {e}")
            raise
    
    def run_benchmark(self, 
                     limit: Optional[int] = None,
                     save_intermediate: bool = True) -> Dict[str, Any]:
        """Run the complete benchmark.
        
        Args:
            limit: Optional limit on number of items to process
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary containing benchmark results and statistics
        """
        logging.info("Starting benchmark run")
        self.stats['start_time'] = time.time()
        
        try:
            # Validate setup
            self._validate_setup()
            
            # Process dataset
            self.results = []
            processed_count = 0
            
            for item in self.dataset:
                # Check both the method parameter and instance variable
                effective_limit = limit or self.sample_limit
                if effective_limit and processed_count >= effective_limit:
                    break
                
                try:
                    result = self._process_item(item)
                    self.results.append(result)
                    self.stats['processed_items'] += 1
                    
                    # Update cost tracking
                    if self.model_router:
                        self.stats['total_cost'] += self.model_router.get_usage_stats()['estimated_cost']
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        logging.info(f"Processed {processed_count} items")
                    
                    # Save intermediate results
                    if save_intermediate and processed_count % 50 == 0:
                        self._save_intermediate_results()
                        
                except Exception as e:
                    logging.error(f"Failed to process item {processed_count}: {e}")
                    self.stats['failed_items'] += 1
                    continue
            
            # Finalize results
            self.stats['end_time'] = time.time()
            self.stats['duration'] = self.stats['end_time'] - self.stats['start_time']
            
            # Save final results
            self._save_final_results()
            
            logging.info(f"Benchmark completed. Processed {self.stats['processed_items']} items")
            return self._get_summary()
            
        except Exception as e:
            logging.error(f"Benchmark failed: {e}")
            raise
    
    def _validate_setup(self) -> None:
        """Validate that all components are properly setup.
        
        Raises:
            RuntimeError: If setup is invalid
        """
        if not self.model_router:
            raise RuntimeError("Model router not initialized")
        
        if not self.dataset:
            raise RuntimeError("Dataset not initialized")
        
        if not self.pipeline_modules:
            raise RuntimeError("Pipeline modules not initialized")
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single dataset item through the pipeline.
        
        Args:
            item: Dataset item to process
            
        Returns:
            Processed result dictionary
        """
        result = item.copy()
        result['pipeline_results'] = {}
        result['processing_time'] = time.time()
        
        # Run through pipeline modules
        for module in self.pipeline_modules:
            try:
                module_result = module.safe_process(result)
                result.update(module_result)
                result['pipeline_results'][module.name] = {
                    'status': module_result.get('module_status', 'unknown'),
                    'output': module_result
                }
            except Exception as e:
                logging.error(f"Module {module.name} failed: {e}")
                result['pipeline_results'][module.name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        result['processing_time'] = time.time() - result['processing_time']
        return result
    
    def _save_intermediate_results(self) -> None:
        """Save intermediate results to file."""
        try:
            filename = f"intermediate_results_{int(time.time())}.csv"
            self.results_writer.save_csv(self.results, filename)
            logging.info(f"Saved intermediate results to {filename}")
        except Exception as e:
            logging.error(f"Failed to save intermediate results: {e}")
    
    def _save_final_results(self) -> None:
        """Save final results and statistics."""
        try:
            # Save results
            timestamp = int(time.time())
            self.results_writer.save_csv(self.results, f"results_{timestamp}.csv")
            self.results_writer.save_json(self.results, f"results_{timestamp}.json")
            
            # Save statistics
            summary = self._get_summary()
            self.results_writer.save_json(summary, f"summary_{timestamp}.json")
            
            logging.info("Final results saved")
            
        except Exception as e:
            logging.error(f"Failed to save final results: {e}")
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary.
        
        Returns:
            Dictionary containing benchmark summary
        """
        summary = {
            'config': self.config,
            'statistics': self.stats.copy(),
            'model_info': self.model_router.get_info() if self.model_router else None,
            'dataset_info': self.dataset.get_statistics() if self.dataset else None,
            'pipeline_info': [module.get_info() for module in self.pipeline_modules]
        }
        
        # Add performance metrics
        if self.stats['processed_items'] > 0:
            summary['performance'] = {
                'items_per_second': self.stats['processed_items'] / self.stats.get('duration', 1),
                'success_rate': self.stats['processed_items'] / self.stats['total_items'],
                'failure_rate': self.stats['failed_items'] / self.stats['total_items'],
                'cost_per_item': self.stats['total_cost'] / self.stats['processed_items']
            }
        
        return summary
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get the current results.
        
        Returns:
            List of result dictionaries
        """
        return self.results.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()
    
    def set_sample_limit(self, limit: int) -> None:
        """Set the maximum number of samples to process.
        
        Args:
            limit: Maximum number of samples
        """
        self.sample_limit = limit
        logging.info(f"Sample limit set to {limit}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint file to resume processing.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            import json
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.results = checkpoint.get('results', [])
            self.stats.update(checkpoint.get('stats', {}))
            self.checkpoint_file = checkpoint_path
            
            logging.info(f"Loaded checkpoint from {checkpoint_path}")
            logging.info(f"Resuming from {len(self.results)} processed items")
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise
    
    def save_checkpoint(self) -> str:
        """Save current progress to a checkpoint file.
        
        Returns:
            Path to the saved checkpoint file
        """
        try:
            import json
            timestamp = int(time.time())
            checkpoint_path = self.output_dir / f"checkpoint_{timestamp}.json"
            
            checkpoint = {
                'results': self.results,
                'stats': self.stats,
                'config': self.config,
                'timestamp': timestamp
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            
            logging.info(f"Checkpoint saved to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            raise


class DatasetPreviewer:
    """Utility class for previewing datasets before running benchmarks."""
    
    def __init__(self, dataset: BaseDataset):
        """Initialize the previewer.
        
        Args:
            dataset: Dataset to preview
        """
        self.dataset = dataset
    
    def preview_items(self, count: int = 5) -> List[Dict[str, Any]]:
        """Preview a few items from the dataset.
        
        Args:
            count: Number of items to preview
            
        Returns:
            List of preview items
        """
        items = []
        for i, item in enumerate(self.dataset):
            if i >= count:
                break
            items.append(item)
        return items
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.dataset.get_statistics()
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the entire dataset.
        
        Returns:
            Validation results
        """
        valid_count = 0
        invalid_count = 0
        errors = []
        
        for i, item in enumerate(self.dataset):
            try:
                if self.dataset.validate_item(item):
                    valid_count += 1
                else:
                    invalid_count += 1
                    errors.append(f"Item {i}: Validation failed")
            except Exception as e:
                invalid_count += 1
                errors.append(f"Item {i}: {str(e)}")
        
        return {
            'total_items': valid_count + invalid_count,
            'valid_items': valid_count,
            'invalid_items': invalid_count,
            'validation_rate': valid_count / (valid_count + invalid_count) if (valid_count + invalid_count) > 0 else 0,
            'errors': errors[:10]  # Limit to first 10 errors
        }