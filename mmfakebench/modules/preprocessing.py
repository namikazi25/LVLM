"""Preprocessing module for data cleaning and preparation.

This module contains the PreprocessingModule class that handles data cleaning,
normalization, and preparation for downstream pipeline processing.
"""

import logging
import re
import base64
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import html
import unicodedata

from core.base import BasePipelineModule


class PreprocessingModule(BasePipelineModule):
    """Pipeline module for data preprocessing and cleaning.
    
    This module cleans and normalizes input data, handles text preprocessing,
    image format conversion, and prepares data for analysis modules.
    """
    
    def __init__(self, 
                 name: str = "preprocessor",
                 clean_text: bool = True,
                 normalize_unicode: bool = True,
                 remove_html: bool = True,
                 remove_urls: bool = False,
                 remove_mentions: bool = False,
                 remove_hashtags: bool = False,
                 lowercase: bool = False,
                 max_text_length: Optional[int] = None,
                 image_preprocessing: bool = True,
                 resize_images: bool = False,
                 target_image_size: Tuple[int, int] = (224, 224),
                 **kwargs):
        """Initialize the preprocessing module.
        
        Args:
            name: Name of the module
            clean_text: Whether to perform text cleaning
            normalize_unicode: Whether to normalize unicode characters
            remove_html: Whether to remove HTML tags
            remove_urls: Whether to remove URLs from text
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags
            lowercase: Whether to convert text to lowercase
            max_text_length: Maximum text length (truncate if longer)
            image_preprocessing: Whether to preprocess images
            resize_images: Whether to resize images
            target_image_size: Target size for image resizing (width, height)
            **kwargs: Additional configuration parameters
        """
        config = {
            'clean_text': clean_text,
            'normalize_unicode': normalize_unicode,
            'remove_html': remove_html,
            'remove_urls': remove_urls,
            'remove_mentions': remove_mentions,
            'remove_hashtags': remove_hashtags,
            'lowercase': lowercase,
            'max_text_length': max_text_length,
            'image_preprocessing': image_preprocessing,
            'resize_images': resize_images,
            'target_image_size': target_image_size,
            **kwargs
        }
        super().__init__(name, config)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Compile regex patterns for efficiency
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._mention_pattern = re.compile(r'@\w+')
        self._hashtag_pattern = re.compile(r'#\w+')
        self._html_pattern = re.compile(r'<[^>]+>')
        self._whitespace_pattern = re.compile(r'\s+')
    
    def initialize(self) -> None:
        """Initialize the preprocessing module."""
        try:
            self.logger.info("PreprocessingModule initialized")
            self.logger.debug(f"Text cleaning enabled: {self.config['clean_text']}")
            self.logger.debug(f"Image preprocessing enabled: {self.config['image_preprocessing']}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PreprocessingModule: {e}")
            raise
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through preprocessing steps.
        
        Args:
            data: Input data to preprocess
            
        Returns:
            Preprocessed data
        """
        try:
            result = data.copy()
            preprocessing_info = {
                'steps_applied': [],
                'original_text_length': 0,
                'processed_text_length': 0,
                'text_changes': [],
                'image_changes': []
            }
            
            # Preprocess text fields
            if self.config['clean_text']:
                result, text_info = self._preprocess_text_fields(result)
                preprocessing_info.update(text_info)
            
            # Preprocess image data
            if self.config['image_preprocessing'] and 'image_data' in result:
                result, image_info = self._preprocess_image_data(result)
                preprocessing_info['image_changes'] = image_info
            
            # Add preprocessing metadata
            result['preprocessing_info'] = preprocessing_info
            
            self.logger.debug(f"Preprocessing completed. Steps applied: {preprocessing_info['steps_applied']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return {
                **data,
                'preprocessing_info': {
                    'steps_applied': [],
                    'error': str(e)
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
            'preprocessing_info': {
                'type': 'object',
                'properties': {
                    'steps_applied': {'type': 'array', 'items': {'type': 'string'}},
                    'original_text_length': {'type': 'integer', 'minimum': 0},
                    'processed_text_length': {'type': 'integer', 'minimum': 0},
                    'text_changes': {'type': 'array', 'items': {'type': 'string'}},
                    'image_changes': {'type': 'array', 'items': {'type': 'string'}}
                },
                'required': ['steps_applied']
            }
        }
    
    def _preprocess_text_fields(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Preprocess text fields in the data.
        
        Args:
            data: Input data containing text fields
            
        Returns:
            Tuple of (processed_data, preprocessing_info)
        """
        result = data.copy()
        info = {
            'steps_applied': [],
            'original_text_length': 0,
            'processed_text_length': 0,
            'text_changes': []
        }
        
        # Process headline field
        if 'headline' in result and result['headline']:
            original_text = result['headline']
            info['original_text_length'] = len(original_text)
            
            processed_text = self._clean_text(original_text, info)
            result['headline'] = processed_text
            info['processed_text_length'] = len(processed_text)
            
            if original_text != processed_text:
                info['text_changes'].append(f"Headline: {len(original_text)} -> {len(processed_text)} chars")
        
        # Process other text fields if present
        text_fields = ['description', 'content', 'body', 'text']
        for field in text_fields:
            if field in result and result[field]:
                original_text = result[field]
                processed_text = self._clean_text(original_text, info)
                result[field] = processed_text
                
                if original_text != processed_text:
                    info['text_changes'].append(f"{field}: {len(original_text)} -> {len(processed_text)} chars")
        
        return result, info
    
    def _clean_text(self, text: str, info: Dict[str, Any]) -> str:
        """Clean and normalize text content.
        
        Args:
            text: Input text to clean
            info: Info dictionary to track applied steps
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        cleaned_text = text
        
        # Remove HTML tags
        if self.config['remove_html']:
            cleaned_text = html.unescape(cleaned_text)  # Decode HTML entities first
            cleaned_text = self._html_pattern.sub(' ', cleaned_text)
            if 'html_removal' not in info['steps_applied']:
                info['steps_applied'].append('html_removal')
        
        # Remove URLs
        if self.config['remove_urls']:
            cleaned_text = self._url_pattern.sub(' ', cleaned_text)
            if 'url_removal' not in info['steps_applied']:
                info['steps_applied'].append('url_removal')
        
        # Remove mentions
        if self.config['remove_mentions']:
            cleaned_text = self._mention_pattern.sub(' ', cleaned_text)
            if 'mention_removal' not in info['steps_applied']:
                info['steps_applied'].append('mention_removal')
        
        # Remove hashtags
        if self.config['remove_hashtags']:
            cleaned_text = self._hashtag_pattern.sub(' ', cleaned_text)
            if 'hashtag_removal' not in info['steps_applied']:
                info['steps_applied'].append('hashtag_removal')
        
        # Normalize unicode
        if self.config['normalize_unicode']:
            cleaned_text = unicodedata.normalize('NFKC', cleaned_text)
            if 'unicode_normalization' not in info['steps_applied']:
                info['steps_applied'].append('unicode_normalization')
        
        # Convert to lowercase
        if self.config['lowercase']:
            cleaned_text = cleaned_text.lower()
            if 'lowercase' not in info['steps_applied']:
                info['steps_applied'].append('lowercase')
        
        # Normalize whitespace
        cleaned_text = self._whitespace_pattern.sub(' ', cleaned_text).strip()
        if 'whitespace_normalization' not in info['steps_applied']:
            info['steps_applied'].append('whitespace_normalization')
        
        # Truncate if too long
        if self.config['max_text_length'] and len(cleaned_text) > self.config['max_text_length']:
            cleaned_text = cleaned_text[:self.config['max_text_length']].strip()
            if 'text_truncation' not in info['steps_applied']:
                info['steps_applied'].append('text_truncation')
        
        return cleaned_text
    
    def _preprocess_image_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Preprocess image data.
        
        Args:
            data: Input data containing image data
            
        Returns:
            Tuple of (processed_data, list_of_changes)
        """
        result = data.copy()
        changes = []
        
        if 'image_data' not in result or not result['image_data']:
            return result, changes
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(result['image_data'])
            original_size = len(image_bytes)
            
            # Basic image validation
            if not self._is_valid_image(image_bytes):
                changes.append("Invalid image format detected")
                return result, changes
            
            # Image format conversion or optimization could be added here
            # For now, we just validate and pass through
            
            changes.append(f"Image validated: {original_size} bytes")
            
            # Add image metadata
            result['image_info'] = {
                'original_size_bytes': original_size,
                'format': self._detect_image_format(image_bytes),
                'processed': True
            }
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            changes.append(f"Image preprocessing error: {e}")
        
        return result, changes
    
    def _is_valid_image(self, image_bytes: bytes) -> bool:
        """Check if image bytes represent a valid image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            True if valid image, False otherwise
        """
        if not image_bytes or len(image_bytes) < 10:
            return False
        
        # Check for common image format magic bytes
        return (
            image_bytes.startswith(b'\xFF\xD8\xFF') or  # JPEG
            image_bytes.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
            image_bytes.startswith(b'GIF87a') or  # GIF87a
            image_bytes.startswith(b'GIF89a') or  # GIF89a
            image_bytes.startswith(b'BM') or  # BMP
            (image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12])  # WebP
        )
    
    def _detect_image_format(self, image_bytes: bytes) -> Optional[str]:
        """Detect image format from magic bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Detected format string or None
        """
        if not image_bytes:
            return None
        
        if image_bytes.startswith(b'\xFF\xD8\xFF'):
            return 'jpeg'
        elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
            return 'gif'
        elif image_bytes.startswith(b'BM'):
            return 'bmp'
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            return 'webp'
        
        return 'unknown'