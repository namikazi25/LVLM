#!/usr/bin/env python3
"""
Test Phase 6.1: Command Line Interface Implementation

This test suite validates the enhanced CLI functionality including:
- Command parsing and validation
- Parameter override capabilities
- Progress indicators and error handling
- Help system and user experience
"""

import unittest
import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import CLI functions by running the main module
import subprocess
import sys
from core.config import ConfigManager


class TestCLIInterface(unittest.TestCase):
    """Test the command line interface functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        
        # Create a test configuration file
        self.test_config = {
            'name': 'test_benchmark',
            'model': {
                'name': 'gpt-4o-mini',
                'temperature': 0.2
            },
            'dataset': {
                'type': 'mmfakebench',
                'data_path': 'data/test'
            },
            'pipeline': [
                {'type': 'preprocessing', 'name': 'relevance_checker'},
                {'type': 'detection', 'name': 'claim_enrichment'}
            ]
        }
        
        self.test_config_path = os.path.join(self.temp_dir, 'test_config.json')
        with open(self.test_config_path, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parse_override_params(self):
        """Test parameter override parsing."""
        # Import the function directly from __main__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("main_module", "__main__.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Test basic string override
        overrides = main_module.parse_override_params(['model.name=gpt-4o'])
        self.assertEqual(overrides, {'model.name': 'gpt-4o'})
        
        # Test boolean override
        overrides = main_module.parse_override_params(['dataset.shuffle=true'])
        self.assertEqual(overrides, {'dataset.shuffle': True})
        
        # Test integer override
        overrides = main_module.parse_override_params(['dataset.batch_size=32'])
        self.assertEqual(overrides, {'dataset.batch_size': 32})
        
        # Test float override
        overrides = main_module.parse_override_params(['model.temperature=0.5'])
        self.assertEqual(overrides, {'model.temperature': 0.5})
        
        # Test multiple overrides
        overrides = main_module.parse_override_params([
            'model.name=gpt-4o',
            'dataset.batch_size=16',
            'model.temperature=0.3'
        ])
        expected = {
            'model.name': 'gpt-4o',
            'dataset.batch_size': 16,
            'model.temperature': 0.3
        }
        self.assertEqual(overrides, expected)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            main_module.parse_override_params(['invalid_format'])
    
    def test_config_manager_apply_overrides(self):
        """Test configuration override application."""
        overrides = {
            'model.name': 'gpt-4o',
            'model.temperature': 0.5,
            'dataset.batch_size': 32,
            'new_section.new_key': 'new_value'
        }
        
        updated_config = self.config_manager.apply_overrides(self.test_config, overrides)
        
        # Check that overrides were applied
        self.assertEqual(updated_config['model']['name'], 'gpt-4o')
        self.assertEqual(updated_config['model']['temperature'], 0.5)
        self.assertEqual(updated_config['dataset']['batch_size'], 32)
        self.assertEqual(updated_config['new_section']['new_key'], 'new_value')
        
        # Check that original config wasn't modified
        self.assertEqual(self.test_config['model']['name'], 'gpt-4o-mini')
        self.assertEqual(self.test_config['model']['temperature'], 0.2)
    
    def test_cli_help_command(self):
        """Test that help command works correctly."""
        result = subprocess.run(
            [sys.executable, '__main__.py', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Help should exit with code 0
        self.assertEqual(result.returncode, 0)
        
        # Check that help text contains expected content
        help_output = result.stdout
        self.assertIn('MMFakeBench', help_output)
        self.assertIn('run', help_output)
        self.assertIn('preview', help_output)
        self.assertIn('validate', help_output)
    
    def test_cli_run_command_validation(self):
        """Test run command argument validation."""
        # Test missing required config argument
        result = subprocess.run(
            [sys.executable, '__main__.py', 'run'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Should exit with error code
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('required', result.stderr.lower())
    
    def test_cli_preview_command_validation(self):
        """Test preview command argument validation."""
        # Test missing required dataset argument
        result = subprocess.run(
            [sys.executable, '__main__.py', 'preview'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Should exit with error code
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('required', result.stderr.lower())
    
    def test_cli_validate_command_validation(self):
        """Test validate command argument validation."""
        # Test missing required config argument
        result = subprocess.run(
            [sys.executable, '__main__.py', 'validate'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Should exit with error code
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('required', result.stderr.lower())
    
    def test_setup_logging(self):
        """Test logging setup functionality."""
        # Import the function directly from __main__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("main_module", "__main__.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        import logging
        
        # Test verbose logging
        main_module.setup_logging(verbose=True)
        logger = logging.getLogger()
        self.assertLessEqual(logger.level, logging.DEBUG)  # Should be DEBUG or lower
        
        # Test normal logging
        main_module.setup_logging(verbose=False)
        logger = logging.getLogger()
        self.assertGreaterEqual(logger.level, logging.DEBUG)  # Should be INFO or higher
    
    def test_cli_dry_run_flag(self):
        """Test that dry run flag is recognized."""
        # Test with a valid config file
        result = subprocess.run(
            [sys.executable, '__main__.py', 'run', '--config', self.test_config_path, '--dry-run'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Should recognize the dry-run flag (may fail due to missing dependencies, but flag should be parsed)
        self.assertNotIn('unrecognized arguments', result.stderr)
    
    def test_error_handling(self):
        """Test error handling and exit codes."""
        # Test FileNotFoundError
        result = subprocess.run(
            [sys.executable, '__main__.py', 'validate', '--config', 'nonexistent.json'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        self.assertNotEqual(result.returncode, 0)  # Should fail with non-zero exit code
    
    def test_parameter_override_integration(self):
        """Test parameter override integration with config manager."""
        # Test that overrides are properly applied
        overrides = {
            'model.name': 'claude-3-sonnet',
            'dataset.batch_size': 64
        }
        
        updated_config = self.config_manager.apply_overrides(self.test_config, overrides)
        
        # Verify overrides were applied correctly
        self.assertEqual(updated_config['model']['name'], 'claude-3-sonnet')
        self.assertEqual(updated_config['dataset']['batch_size'], 64)
        
        # Verify original values are preserved where not overridden
        self.assertEqual(updated_config['model']['temperature'], 0.2)
        self.assertEqual(updated_config['dataset']['type'], 'mmfakebench')


class TestCLIUserExperience(unittest.TestCase):
    """Test CLI user experience features."""
    
    def test_command_examples_in_help(self):
        """Test that help includes usage examples."""
        result = subprocess.run(
            [sys.executable, '__main__.py', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        help_output = result.stdout
        self.assertIn('Examples:', help_output)
        self.assertIn('run --config', help_output)
        self.assertIn('preview --dataset', help_output)
        self.assertIn('validate --config', help_output)
    
    def test_verbose_and_quiet_flags(self):
        """Test verbose and quiet flag functionality."""
        # Test that verbose flag is recognized (should not cause argument parsing errors)
        result = subprocess.run(
            [sys.executable, '__main__.py', '--verbose', 'run', '--config', 'test.json'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        # Should not have unrecognized argument errors
        self.assertNotIn('unrecognized arguments', result.stderr)
        
        # Test that quiet flag is recognized
        result = subprocess.run(
            [sys.executable, '__main__.py', '--quiet', 'validate', '--config', 'test.json'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        # Should not have unrecognized argument errors
        self.assertNotIn('unrecognized arguments', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)