"""Module Registry System for MMFakeBench.

This module provides a centralized registry for managing and loading
pipeline modules dynamically by name.
"""

import importlib
import inspect
from typing import Dict, Type, List, Optional
from .base import BasePipelineModule


class ModuleRegistry:
    """Registry for managing pipeline modules.
    
    Provides functionality to register, discover, and load modules
    dynamically by name.
    """
    
    def __init__(self):
        self._modules: Dict[str, Type[BasePipelineModule]] = {}
        self._auto_discover()
    
    def register(self, name: str, module_class: Type[BasePipelineModule]) -> None:
        """Register a module class with a given name.
        
        Args:
            name: The name to register the module under
            module_class: The module class to register
            
        Raises:
            ValueError: If the module class is not a subclass of BasePipelineModule
        """
        if not issubclass(module_class, BasePipelineModule):
            raise ValueError(f"Module {module_class.__name__} must inherit from BasePipelineModule")
        
        self._modules[name] = module_class
    
    def get(self, name: str) -> Type[BasePipelineModule]:
        """Get a module class by name.
        
        Args:
            name: The name of the module to retrieve
            
        Returns:
            The module class
            
        Raises:
            KeyError: If the module is not registered
        """
        if name not in self._modules:
            available = list(self._modules.keys())
            raise KeyError(f"Module '{name}' not found. Available modules: {available}")
        
        return self._modules[name]
    
    def list_modules(self) -> List[str]:
        """List all registered module names.
        
        Returns:
            List of registered module names
        """
        return list(self._modules.keys())
    
    def create_module(self, name: str, **kwargs) -> BasePipelineModule:
        """Create an instance of a registered module.
        
        Args:
            name: The name of the module to create
            **kwargs: Arguments to pass to the module constructor
            
        Returns:
            An instance of the requested module
            
        Raises:
            KeyError: If the module is not registered
        """
        module_class = self.get(name)
        return module_class(**kwargs)
    
    def _auto_discover(self) -> None:
        """Automatically discover and register modules from the modules package."""
        try:
            # Import the modules package
            modules_package = importlib.import_module('mmfakebench.modules')
            
            # List of known module files
            module_files = [
                'relevance_checker',
                'claim_enrichment', 
                'question_generator',
                'evidence_tagger',
                'synthesizer',
                'web_searcher'
            ]
            
            for module_file in module_files:
                try:
                    # Import the module
                    module = importlib.import_module(f'mmfakebench.modules.{module_file}')
                    
                    # Find classes that inherit from BasePipelineModule
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BasePipelineModule) and 
                            obj != BasePipelineModule and
                            obj.__module__ == module.__name__):
                            # Register using the class name
                            self.register(name, obj)
                            
                except ImportError as e:
                    # Skip modules that can't be imported
                    continue
                    
        except ImportError:
            # If modules package can't be imported, skip auto-discovery
            pass


# Global registry instance
registry = ModuleRegistry()