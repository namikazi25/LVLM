"""Configuration management for MisinfoBench.

This module provides utilities for loading, validating, and managing
benchmark configurations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .io import ConfigLoader


class ConfigManager:
    """Manager for benchmark configurations.
    
    This class handles loading, validation, and management of
    configuration files for benchmarks.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.configs = {}
        self.validator = ConfigValidator()
    
    def load_config(self, config_name: str, 
                   config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load a configuration file.
        
        Args:
            config_name: Name of the configuration
            config_path: Optional path to config file
            
        Returns:
            Configuration dictionary
        """
        if config_path:
            filepath = Path(config_path)
        else:
            # Try different extensions
            for ext in ['.json', '.yaml', '.yml']:
                filepath = self.config_dir / f"{config_name}{ext}"
                if filepath.exists():
                    break
            else:
                raise FileNotFoundError(f"Configuration '{config_name}' not found in {self.config_dir}")
        
        # Load based on file extension
        if filepath.suffix.lower() == '.json':
            config = ConfigLoader.load_json(filepath)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            config = ConfigLoader.load_yaml(filepath)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")
        
        # Validate configuration
        validation = self.validator.validate_benchmark_config(config)
        if not validation['valid']:
            raise ValueError(f"Invalid configuration: {validation['errors']}")
        
        # Store configuration
        self.configs[config_name] = config
        logging.info(f"Loaded configuration: {config_name}")
        
        return config
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a loaded configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary
            
        Raises:
            KeyError: If configuration is not loaded
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration '{config_name}' not loaded")
        
        return self.configs[config_name].copy()
    
    def list_configs(self) -> List[str]:
        """List all loaded configurations.
        
        Returns:
            List of configuration names
        """
        return list(self.configs.keys())
    
    def apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter overrides to a configuration.
        
        Args:
            config: Base configuration dictionary
            overrides: Override parameters in dot notation (e.g., 'model.name')
            
        Returns:
            Updated configuration dictionary
        """
        import copy
        
        # Create a deep copy to avoid modifying the original
        updated_config = copy.deepcopy(config)
        
        for key, value in overrides.items():
            # Handle dot notation (e.g., 'model.name' -> ['model', 'name'])
            key_parts = key.split('.')
            current = updated_config
            
            # Navigate to the parent of the target key
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the final value
            current[key_parts[-1]] = value
            
        return updated_config
    
    def create_default_config(self, config_type: str = 'basic') -> Dict[str, Any]:
        """Create a default configuration.
        
        Args:
            config_type: Type of default configuration
            
        Returns:
            Default configuration dictionary
        """
        if config_type == 'basic':
            return {
                'name': 'basic_benchmark',
                'description': 'Basic misinformation detection benchmark',
                'model': {
                    'name': 'gpt-4-vision-preview',
                    'temperature': 0.2,
                    'max_retries': 3
                },
                'dataset': {
                    'type': 'misinfobench',
                    'data_path': 'data/misinfobench',
                    'params': {
                        'limit': None,
                        'shuffle': False
                    }
                },
                'pipeline': [
                    {
                        'type': 'preprocessing',
                        'name': 'preprocessor',
                        'config': {}
                    },
                    {
                        'type': 'detection',
                        'name': 'detector',
                        'config': {
                            'prompt_template': 'default'
                        }
                    }
                ],
                'output': {
                    'directory': 'results',
                    'formats': ['json', 'csv'],
                    'save_intermediate': True
                }
            }
        
        elif config_type == 'advanced':
            return {
                'name': 'advanced_benchmark',
                'description': 'Advanced misinformation detection benchmark with analysis',
                'model': {
                    'name': 'gpt-4-vision-preview',
                    'temperature': 0.1,
                    'max_retries': 5,
                    'additional_params': {
                        'max_tokens': 1000
                    }
                },
                'dataset': {
                    'type': 'misinfobench',
                    'data_path': 'data/misinfobench',
                    'params': {
                        'limit': None,
                        'shuffle': True,
                        'validation_split': 0.2
                    }
                },
                'pipeline': [
                    {
                        'type': 'preprocessing',
                        'name': 'preprocessor',
                        'config': {
                            'image_resize': True,
                            'text_cleaning': True
                        }
                    },
                    {
                        'type': 'validation',
                        'name': 'validator',
                        'config': {
                            'strict_mode': True
                        }
                    },
                    {
                        'type': 'detection',
                        'name': 'detector',
                        'config': {
                            'prompt_template': 'detailed',
                            'include_reasoning': True
                        }
                    },
                    {
                        'type': 'analysis',
                        'name': 'analyzer',
                        'config': {
                            'confidence_analysis': True,
                            'error_analysis': True
                        }
                    }
                ],
                'output': {
                    'directory': 'results',
                    'formats': ['json', 'csv', 'txt'],
                    'save_intermediate': True,
                    'generate_report': True
                }
            }
        
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    def save_config(self, config: Dict[str, Any], 
                   config_name: str,
                   format: str = 'json') -> Path:
        """Save a configuration to file.
        
        Args:
            config: Configuration dictionary
            config_name: Name for the configuration
            format: File format ('json' or 'yaml')
            
        Returns:
            Path to saved configuration file
        """
        # Validate configuration before saving
        validation = self.validator.validate_benchmark_config(config)
        if not validation['valid']:
            raise ValueError(f"Invalid configuration: {validation['errors']}")
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        if format.lower() == 'json':
            filepath = self.config_dir / f"{config_name}.json"
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        elif format.lower() in ['yaml', 'yml']:
            try:
                import yaml
            except ImportError:
                raise ImportError("PyYAML is required for YAML format")
            
            filepath = self.config_dir / f"{config_name}.yaml"
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Store in memory
        self.configs[config_name] = config
        
        logging.info(f"Saved configuration '{config_name}' to {filepath}")
        return filepath


class ConfigValidator:
    """Validator for benchmark configurations."""
    
    def __init__(self):
        """Initialize the validator."""
        self.required_keys = {
            'benchmark': ['name', 'model', 'dataset', 'pipeline'],
            'model': ['name'],
            'dataset': ['type', 'data_path'],
            'pipeline_module': ['type', 'name']
        }
        
        self.valid_model_names = [
            'gpt-4-vision-preview',
            'gpt-4o',
            'gpt-4o-mini',
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307',
            'gemini-pro-vision',
            'gemini-1.5-pro'
        ]
        
        self.valid_dataset_types = [
            'misinfobench',
            'mocheg',
            'custom'
        ]
        
        self.valid_module_types = [
            'preprocessing',
            'validation',
            'detection',
            'analysis'
        ]
    
    def validate_benchmark_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete benchmark configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check required top-level keys
        missing_keys = []
        for key in self.required_keys['benchmark']:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Validate model configuration
        model_validation = self.validate_model_config(config['model'])
        if not model_validation['valid']:
            errors.extend([f"Model: {err}" for err in model_validation['errors']])
        warnings.extend([f"Model: {warn}" for warn in model_validation['warnings']])
        
        # Validate dataset configuration
        dataset_validation = self.validate_dataset_config(config['dataset'])
        if not dataset_validation['valid']:
            errors.extend([f"Dataset: {err}" for err in dataset_validation['errors']])
        warnings.extend([f"Dataset: {warn}" for warn in dataset_validation['warnings']])
        
        # Validate pipeline configuration
        pipeline_validation = self.validate_pipeline_config(config['pipeline'])
        if not pipeline_validation['valid']:
            errors.extend([f"Pipeline: {err}" for err in pipeline_validation['errors']])
        warnings.extend([f"Pipeline: {warn}" for warn in pipeline_validation['warnings']])
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check required keys
        missing_keys = []
        for key in self.required_keys['model']:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
        
        # Validate model name
        if 'name' in config:
            if config['name'] not in self.valid_model_names:
                warnings.append(f"Unknown model name: {config['name']}")
        
        # Validate temperature
        if 'temperature' in config:
            temp = config['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append("Temperature must be a number between 0 and 2")
        
        # Validate max_retries
        if 'max_retries' in config:
            retries = config['max_retries']
            if not isinstance(retries, int) or retries < 0:
                errors.append("max_retries must be a non-negative integer")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_dataset_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset configuration.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check required keys
        missing_keys = []
        for key in self.required_keys['dataset']:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
        
        # Validate dataset type
        if 'type' in config:
            if config['type'] not in self.valid_dataset_types:
                errors.append(f"Invalid dataset type: {config['type']}")
        
        # Validate data path
        if 'data_path' in config:
            data_path = Path(config['data_path'])
            if not data_path.exists():
                warnings.append(f"Data path does not exist: {data_path}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_pipeline_config(self, config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate pipeline configuration.
        
        Args:
            config: Pipeline configuration (list of modules)
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        if not isinstance(config, list):
            errors.append("Pipeline must be a list of modules")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        if len(config) == 0:
            warnings.append("Pipeline is empty")
        
        module_names = []
        
        for i, module_config in enumerate(config):
            # Check required keys
            missing_keys = []
            for key in self.required_keys['pipeline_module']:
                if key not in module_config:
                    missing_keys.append(key)
            
            if missing_keys:
                errors.append(f"Module {i}: Missing required keys: {missing_keys}")
                continue
            
            # Validate module type
            module_type = module_config['type']
            if module_type not in self.valid_module_types:
                errors.append(f"Module {i}: Invalid module type: {module_type}")
            
            # Check for duplicate names
            module_name = module_config['name']
            if module_name in module_names:
                errors.append(f"Module {i}: Duplicate module name: {module_name}")
            module_names.append(module_name)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_config_template(self, config_type: str = 'basic') -> Dict[str, Any]:
        """Get a configuration template.
        
        Args:
            config_type: Type of template to generate
            
        Returns:
            Configuration template
        """
        manager = ConfigManager()
        return manager.create_default_config(config_type)
    
    def validate(self, config_path: str) -> int:
        """Validate a configuration file from CLI.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            # Load configuration using ConfigManager
            manager = ConfigManager()
            config_name = Path(config_path).stem
            config = manager.load_config(config_name, config_path)
            
            print(f"✅ Configuration '{config_path}' is valid")
            print(f"   Name: {config.get('name', 'N/A')}")
            print(f"   Description: {config.get('description', 'N/A')}")
            print(f"   Model: {config.get('model', {}).get('name', 'N/A')}")
            print(f"   Dataset: {config.get('dataset', {}).get('type', 'N/A')}")
            print(f"   Pipeline modules: {len(config.get('pipeline', []))}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return 1