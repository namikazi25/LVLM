"""Validation module for data integrity and format checking.

This module contains the ValidationModule class that validates input data
format, checks data integrity, and ensures compliance with expected schemas.
"""

import logging
import base64
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.base import BasePipelineModule


class ValidationModule(BasePipelineModule):
    """Pipeline module for data validation and integrity checking.
    
    This module validates input data format, checks image validity,
    text constraints, and ensures data meets pipeline requirements.
    """
    
    def __init__(self, 
                 name: str = "validator",
                 strict_mode: bool = False,
                 check_image_format: bool = True,
                 check_text_length: bool = True,
                 max_text_length: int = 10000,
                 min_text_length: int = 1,
                 allowed_image_formats: Optional[List[str]] = None,
                 max_image_size: int = 10 * 1024 * 1024,  # 10MB
                 **kwargs):
        """Initialize the validation module.
        
        Args:
            name: Name of the module
            strict_mode: Whether to use strict validation rules
            check_image_format: Whether to validate image formats
            check_text_length: Whether to check text length constraints
            max_text_length: Maximum allowed text length
            min_text_length: Minimum required text length
            allowed_image_formats: List of allowed image formats
            max_image_size: Maximum image size in bytes
            **kwargs: Additional configuration parameters
        """
        config = {
            'strict_mode': strict_mode,
            'check_image_format': check_image_format,
            'check_text_length': check_text_length,
            'max_text_length': max_text_length,
            'min_text_length': min_text_length,
            'allowed_image_formats': allowed_image_formats or ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
            'max_image_size': max_image_size,
            **kwargs
        }
        super().__init__(name, config)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validation_errors = []
        self.validation_warnings = []
    
    def initialize(self) -> None:
        """Initialize the validation module."""
        try:
            self.validation_errors = []
            self.validation_warnings = []
            
            self.logger.info(f"ValidationModule initialized with strict_mode={self.config['strict_mode']}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ValidationModule: {e}")
            raise
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through validation checks.
        
        Args:
            data: Input data to validate
            
        Returns:
            Processed data with validation results
        """
        try:
            self.validation_errors = []
            self.validation_warnings = []
            
            # Perform validation checks
            is_valid = self._validate_data_structure(data)
            
            if self.config['check_text_length']:
                is_valid &= self._validate_text_fields(data)
            
            if self.config['check_image_format']:
                is_valid &= self._validate_image_data(data)
            
            # Additional strict mode checks
            if self.config['strict_mode']:
                is_valid &= self._validate_strict_requirements(data)
            
            # Prepare result
            result = {
                **data,
                'validation_result': {
                    'is_valid': is_valid,
                    'errors': self.validation_errors.copy(),
                    'warnings': self.validation_warnings.copy(),
                    'error_count': len(self.validation_errors),
                    'warning_count': len(self.validation_warnings)
                }
            }
            
            if is_valid:
                self.logger.debug("Data validation passed")
            else:
                self.logger.warning(f"Data validation failed with {len(self.validation_errors)} errors")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation processing failed: {e}")
            return {
                **data,
                'validation_result': {
                    'is_valid': False,
                    'errors': [f"Validation error: {e}"],
                    'warnings': [],
                    'error_count': 1,
                    'warning_count': 0
                },
                'error': str(e)
            }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(data, dict):
            self.logger.warning("Input data must be a dictionary")
            return False
        
        if not data:
            self.logger.warning("Input data cannot be empty")
            return False
        
        return True
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema for this module.
        
        Returns:
            Dictionary describing the output schema
        """
        return {
            'validation_result': {
                'type': 'object',
                'properties': {
                    'is_valid': {'type': 'boolean'},
                    'errors': {'type': 'array', 'items': {'type': 'string'}},
                    'warnings': {'type': 'array', 'items': {'type': 'string'}},
                    'error_count': {'type': 'integer', 'minimum': 0},
                    'warning_count': {'type': 'integer', 'minimum': 0}
                },
                'required': ['is_valid', 'errors', 'warnings', 'error_count', 'warning_count']
            }
        }
    
    def _validate_data_structure(self, data: Dict[str, Any]) -> bool:
        """Validate basic data structure requirements.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        is_valid = True
        
        # Check for required fields
        required_fields = ['headline']
        for field in required_fields:
            if field not in data:
                self.validation_errors.append(f"Missing required field: {field}")
                is_valid = False
            elif not data[field]:
                self.validation_errors.append(f"Required field '{field}' is empty")
                is_valid = False
        
        # Check data types
        if 'headline' in data and not isinstance(data['headline'], str):
            self.validation_errors.append("Field 'headline' must be a string")
            is_valid = False
        
        if 'image_data' in data and data['image_data'] and not isinstance(data['image_data'], str):
            self.validation_errors.append("Field 'image_data' must be a string (base64 encoded)")
            is_valid = False
        
        return is_valid
    
    def _validate_text_fields(self, data: Dict[str, Any]) -> bool:
        """Validate text field constraints.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if text fields are valid, False otherwise
        """
        is_valid = True
        
        # Validate headline length
        if 'headline' in data and data['headline']:
            headline_length = len(data['headline'])
            
            if headline_length < self.config['min_text_length']:
                self.validation_errors.append(
                    f"Headline too short: {headline_length} < {self.config['min_text_length']}"
                )
                is_valid = False
            
            if headline_length > self.config['max_text_length']:
                self.validation_errors.append(
                    f"Headline too long: {headline_length} > {self.config['max_text_length']}"
                )
                is_valid = False
            
            # Check for suspicious patterns
            if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', data['headline']):
                self.validation_warnings.append("Headline contains control characters")
        
        return is_valid
    
    def _validate_image_data(self, data: Dict[str, Any]) -> bool:
        """Validate image data format and constraints.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if image data is valid, False otherwise
        """
        is_valid = True
        
        if 'image_data' in data and data['image_data']:
            try:
                # Check if it's valid base64
                image_bytes = base64.b64decode(data['image_data'])
                
                # Check image size
                if len(image_bytes) > self.config['max_image_size']:
                    self.validation_errors.append(
                        f"Image too large: {len(image_bytes)} > {self.config['max_image_size']} bytes"
                    )
                    is_valid = False
                
                # Check image format by magic bytes
                image_format = self._detect_image_format(image_bytes)
                if image_format and image_format not in self.config['allowed_image_formats']:
                    self.validation_errors.append(
                        f"Unsupported image format: {image_format}. Allowed: {self.config['allowed_image_formats']}"
                    )
                    is_valid = False
                elif not image_format:
                    self.validation_warnings.append("Could not detect image format")
                
            except Exception as e:
                self.validation_errors.append(f"Invalid image data: {e}")
                is_valid = False
        
        return is_valid
    
    def _validate_strict_requirements(self, data: Dict[str, Any]) -> bool:
        """Validate strict mode requirements.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if strict requirements are met, False otherwise
        """
        is_valid = True
        
        # In strict mode, image data is required
        if 'image_data' not in data or not data['image_data']:
            self.validation_errors.append("Image data is required in strict mode")
            is_valid = False
        
        # Additional strict checks can be added here
        
        return is_valid
    
    def _detect_image_format(self, image_bytes: bytes) -> Optional[str]:
        """Detect image format from magic bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Detected format string or None
        """
        if not image_bytes:
            return None
        
        # Check magic bytes for common formats
        if image_bytes.startswith(b'\xFF\xD8\xFF'):
            return 'jpg'
        elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
            return 'gif'
        elif image_bytes.startswith(b'BM'):
            return 'bmp'
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            return 'webp'
        
        return None