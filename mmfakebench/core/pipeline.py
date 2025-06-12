"""Pipeline management for modular processing workflows.

This module provides the PipelineManager class and related utilities
for creating and managing processing pipelines.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from .base import BasePipelineModule


class PipelineManager:
    """Manager for creating and organizing pipeline modules.
    
    This class handles the registration and instantiation of
    different types of pipeline modules.
    """
    
    def __init__(self):
        """Initialize the pipeline manager."""
        self._module_registry = {}
        self._register_builtin_modules()
    
    def _register_builtin_modules(self) -> None:
        """Register built-in pipeline modules."""
        # Import and register built-in modules
        try:
            from modules.detection import DetectionModule
            self.register_module('detection', DetectionModule)
        except ImportError:
            logging.warning("DetectionModule not available")
        
        try:
            from modules.validation import ValidationModule
            self.register_module('validation', ValidationModule)
        except ImportError:
            logging.warning("ValidationModule not available")
        
        try:
            from modules.preprocessing import PreprocessingModule
            self.register_module('preprocessing', PreprocessingModule)
        except ImportError:
            logging.warning("PreprocessingModule not available")
        
        # Register agentic workflow modules
        try:
            from modules.relevance_checker import ImageHeadlineRelevancyChecker
            self.register_module('relevance_checker', ImageHeadlineRelevancyChecker)
        except ImportError:
            logging.warning("ImageHeadlineRelevancyChecker not available")
        
        try:
            from modules.claim_enrichment import ClaimEnrichmentTool
            self.register_module('claim_enrichment', ClaimEnrichmentTool)
        except ImportError:
            logging.warning("ClaimEnrichmentTool not available")
        
        try:
            from modules.question_generator import QAGenerationTool
            self.register_module('qa_generation', QAGenerationTool)
        except ImportError:
            logging.warning("QAGenerationTool not available")
        
        try:
            from modules.evidence_tagger import EvidenceTagger
            self.register_module('evidence_tagger', EvidenceTagger)
        except ImportError:
            logging.warning("EvidenceTagger not available")
        
        try:
            from modules.synthesizer import Synthesizer
            self.register_module('synthesizer', Synthesizer)
        except ImportError:
            logging.warning("Synthesizer not available")
        
        try:
            from modules.web_searcher import WebSearcher
            self.register_module('web_searcher', WebSearcher)
        except ImportError:
            logging.warning("WebSearcher not available")
        
        # Register agentic orchestrator
        try:
            from modules.agentic_orchestrator import AgenticOrchestrator
            self.register_module('agentic_orchestrator', AgenticOrchestrator)
        except ImportError:
            logging.warning("AgenticOrchestrator not available")
    
    def register_module(self, module_type: str, module_class: Type[BasePipelineModule]) -> None:
        """Register a pipeline module type.
        
        Args:
            module_type: String identifier for the module type
            module_class: Class that implements BasePipelineModule
        """
        if not issubclass(module_class, BasePipelineModule):
            raise ValueError(f"Module class must inherit from BasePipelineModule")
        
        self._module_registry[module_type] = module_class
        logging.info(f"Registered pipeline module: {module_type}")
    
    def create_module(self, 
                     module_type: str, 
                     name: str, 
                     config: Optional[Dict[str, Any]] = None) -> BasePipelineModule:
        """Create a pipeline module instance.
        
        Args:
            module_type: Type of module to create
            name: Name for the module instance
            config: Configuration dictionary for the module
            
        Returns:
            Instantiated pipeline module
            
        Raises:
            ValueError: If module type is not registered
        """
        if module_type not in self._module_registry:
            available_types = list(self._module_registry.keys())
            raise ValueError(f"Unknown module type '{module_type}'. Available types: {available_types}")
        
        module_class = self._module_registry[module_type]
        config = config or {}
        
        try:
            module = module_class(name=name, config=config)
            logging.info(f"Created pipeline module: {name} ({module_type})")
            return module
        except Exception as e:
            logging.error(f"Failed to create module {name} ({module_type}): {e}")
            raise
    
    def get_available_modules(self) -> List[str]:
        """Get list of available module types.
        
        Returns:
            List of registered module type names
        """
        return list(self._module_registry.keys())
    
    def get_module_info(self, module_type: str) -> Dict[str, Any]:
        """Get information about a module type.
        
        Args:
            module_type: Type of module to get info for
            
        Returns:
            Dictionary containing module information
            
        Raises:
            ValueError: If module type is not registered
        """
        if module_type not in self._module_registry:
            raise ValueError(f"Unknown module type: {module_type}")
        
        module_class = self._module_registry[module_type]
        
        return {
            'type': module_type,
            'class_name': module_class.__name__,
            'module': module_class.__module__,
            'docstring': module_class.__doc__,
            'required_config': getattr(module_class, 'REQUIRED_CONFIG', []),
            'optional_config': getattr(module_class, 'OPTIONAL_CONFIG', [])
        }


class Pipeline:
    """A pipeline of processing modules.
    
    This class represents an ordered sequence of pipeline modules
    that can be executed on data items.
    """
    
    def __init__(self, name: str, modules: Optional[List[BasePipelineModule]] = None):
        """Initialize the pipeline.
        
        Args:
            name: Name of the pipeline
            modules: List of pipeline modules
        """
        self.name = name
        self.modules = modules or []
        self.stats = {
            'items_processed': 0,
            'items_failed': 0,
            'total_processing_time': 0.0
        }
    
    def add_module(self, module: BasePipelineModule) -> None:
        """Add a module to the pipeline.
        
        Args:
            module: Pipeline module to add
        """
        self.modules.append(module)
        logging.info(f"Added module {module.name} to pipeline {self.name}")
    
    def remove_module(self, module_name: str) -> bool:
        """Remove a module from the pipeline.
        
        Args:
            module_name: Name of the module to remove
            
        Returns:
            True if module was removed, False if not found
        """
        for i, module in enumerate(self.modules):
            if module.name == module_name:
                removed_module = self.modules.pop(i)
                logging.info(f"Removed module {removed_module.name} from pipeline {self.name}")
                return True
        return False
    
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process an item through the pipeline.
        
        Args:
            item: Data item to process
            
        Returns:
            Processed item with results from all modules
        """
        import time
        
        start_time = time.time()
        result = item.copy()
        result['pipeline_results'] = {}
        
        try:
            for module in self.modules:
                try:
                    module_result = module.safe_process(result)
                    result.update(module_result)
                    result['pipeline_results'][module.name] = {
                        'status': 'success',
                        'output': module_result
                    }
                except Exception as e:
                    logging.error(f"Module {module.name} failed in pipeline {self.name}: {e}")
                    result['pipeline_results'][module.name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            self.stats['items_processed'] += 1
            
        except Exception as e:
            logging.error(f"Pipeline {self.name} failed: {e}")
            self.stats['items_failed'] += 1
            raise
        
        finally:
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            result['pipeline_processing_time'] = processing_time
        
        return result
    
    def get_module_names(self) -> List[str]:
        """Get names of all modules in the pipeline.
        
        Returns:
            List of module names
        """
        return [module.name for module in self.modules]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        stats = self.stats.copy()
        
        if stats['items_processed'] > 0:
            stats['average_processing_time'] = stats['total_processing_time'] / stats['items_processed']
            stats['success_rate'] = stats['items_processed'] / (stats['items_processed'] + stats['items_failed'])
        else:
            stats['average_processing_time'] = 0.0
            stats['success_rate'] = 0.0
        
        stats['module_count'] = len(self.modules)
        stats['module_names'] = self.get_module_names()
        
        return stats
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """Validate the pipeline configuration.
        
        Returns:
            Dictionary containing validation results
        """
        issues = []
        warnings = []
        
        # Check for empty pipeline
        if not self.modules:
            issues.append("Pipeline is empty")
        
        # Check for duplicate module names
        module_names = self.get_module_names()
        if len(module_names) != len(set(module_names)):
            duplicates = [name for name in set(module_names) if module_names.count(name) > 1]
            issues.append(f"Duplicate module names: {duplicates}")
        
        # Validate each module
        for module in self.modules:
            try:
                module_validation = module.validate_config()
                if not module_validation.get('valid', True):
                    issues.extend(module_validation.get('errors', []))
                warnings.extend(module_validation.get('warnings', []))
            except Exception as e:
                issues.append(f"Module {module.name} validation failed: {e}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'module_count': len(self.modules),
            'module_names': module_names
        }
    
    def reset_statistics(self) -> None:
        """Reset pipeline statistics."""
        self.stats = {
            'items_processed': 0,
            'items_failed': 0,
            'total_processing_time': 0.0
        }
        logging.info(f"Reset statistics for pipeline {self.name}")


class PipelineBuilder:
    """Builder class for creating pipelines from configuration."""
    
    def __init__(self, pipeline_manager: PipelineManager):
        """Initialize the pipeline builder.
        
        Args:
            pipeline_manager: Manager for creating modules
        """
        self.pipeline_manager = pipeline_manager
    
    def build_from_config(self, config: Dict[str, Any]) -> Pipeline:
        """Build a pipeline from configuration.
        
        Args:
            config: Pipeline configuration dictionary
            
        Returns:
            Configured pipeline
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'name' not in config:
            raise ValueError("Pipeline configuration must include 'name'")
        
        if 'modules' not in config:
            raise ValueError("Pipeline configuration must include 'modules'")
        
        pipeline = Pipeline(name=config['name'])
        
        for module_config in config['modules']:
            if 'type' not in module_config:
                raise ValueError("Module configuration must include 'type'")
            
            if 'name' not in module_config:
                raise ValueError("Module configuration must include 'name'")
            
            module = self.pipeline_manager.create_module(
                module_type=module_config['type'],
                name=module_config['name'],
                config=module_config.get('config', {})
            )
            
            pipeline.add_module(module)
        
        # Validate the built pipeline
        validation = pipeline.validate_pipeline()
        if not validation['valid']:
            raise ValueError(f"Invalid pipeline configuration: {validation['issues']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logging.warning(f"Pipeline {pipeline.name}: {warning}")
        
        logging.info(f"Built pipeline {pipeline.name} with {len(pipeline.modules)} modules")
        return pipeline
    
    def build_simple_pipeline(self, 
                            name: str, 
                            module_types: List[str],
                            base_config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """Build a simple pipeline with default configurations.
        
        Args:
            name: Name of the pipeline
            module_types: List of module types to include
            base_config: Base configuration to apply to all modules
            
        Returns:
            Configured pipeline
        """
        pipeline = Pipeline(name=name)
        base_config = base_config or {}
        
        for i, module_type in enumerate(module_types):
            module_name = f"{module_type}_{i+1}"
            module = self.pipeline_manager.create_module(
                module_type=module_type,
                name=module_name,
                config=base_config.copy()
            )
            pipeline.add_module(module)
        
        logging.info(f"Built simple pipeline {name} with modules: {module_types}")
        return pipeline