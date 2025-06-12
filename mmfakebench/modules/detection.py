"""Detection module for misinformation detection pipeline.

This module contains the DetectionModule class that performs the core
misinformation detection logic using configured models and prompts.
"""

import os
import logging
from typing import Dict, Any, Optional

from core.base import BasePipelineModule
from models.router import ModelRouter
from core.prompts import PromptManager


class DetectionModule(BasePipelineModule):
    """Pipeline module for misinformation detection.
    
    This module performs the core misinformation detection task by analyzing
    image-headline pairs and determining their veracity using configured models.
    """
    
    def __init__(self, 
                 name: str = "detector",
                 config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """Initialize the detection module.
        
        Args:
            name: Name of the module
            config: Module configuration dictionary
            **kwargs: Additional configuration parameters
        """
        # Set default configuration
        default_config = {
            'model_name': 'gpt-4o',
            'provider_name': 'openai',
            'prompt_template': 'baseline',
            'include_reasoning': True,
            'confidence_threshold': 0.5,
            'temperature': 0.1,
            'max_retries': 3,
        }
        
        # Merge with provided config, giving priority to provided config
        final_config = {**default_config, **(config or {}), **kwargs}
        
        # Override with YAML config values if present
        if config:
            if 'llm_provider' in config:
                final_config['provider_name'] = config['llm_provider']
            if 'model_name' in config:
                final_config['model_name'] = config['model_name']
        
        super().__init__(name, final_config)
        
        self.model_router = None
        self.prompt_manager = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize(self) -> None:
        """Initialize the detection module with model and prompt resources."""
        try:
            # Initialize model router
            api_key = self.config.get('api_key') or os.getenv(f"{self.config['provider_name'].upper()}_API_KEY")
            if not api_key:
                self.logger.warning(f"No API key found for {self.config['provider_name']}")
            
            self.model_router = ModelRouter(
                model_name=self.config['model_name'],
                api_key=api_key,
                temperature=self.config['temperature'],
                max_retries=self.config['max_retries'],
                provider_name=self.config['provider_name']
            )
            
            # Initialize prompt manager
            self.prompt_manager = PromptManager()
            
            self.logger.info(f"DetectionModule initialized with model {self.config['model_name']}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DetectionModule: {e}")
            raise
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data to detect misinformation.
        
        Args:
            data: Input data containing image and headline information
            
        Returns:
            Processed data with detection results
        """
        try:
            # Extract required fields
            headline = data.get('headline', '')
            image_data = data.get('image_data')
            
            # Get prompt template
            prompt = self.prompt_manager.render_template(
                self.config['prompt_template'],
                {
                    'headline': headline,
                    'include_reasoning': self.config['include_reasoning']
                }
            )
            
            # Generate prediction
            response = self.model_router.generate(
                prompt=prompt,
                image_data=image_data,
                temperature=self.config['temperature']
            )
            
            # Parse response and extract prediction
            prediction_result = self._parse_response(response)
            
            # Add detection results to data
            result = {
                **data,
                'detection_result': prediction_result,
                'model_response': response,
                'prompt_used': prompt,
                'model_name': self.config['model_name'],
                'confidence': prediction_result.get('confidence', 0.0)
            }
            
            self.logger.debug(f"Detection completed for item: {prediction_result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return {
                **data,
                'detection_result': {'prediction': 'error', 'confidence': 0.0},
                'error': str(e)
            }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        required_fields = ['headline']
        
        # Check required fields
        for field in required_fields:
            if field not in data or not data[field]:
                self.logger.warning(f"Missing or empty required field: {field}")
                return False
        
        # Validate headline is string
        if not isinstance(data['headline'], str):
            self.logger.warning("Headline must be a string")
            return False
        
        # Image data is optional but should be valid if present
        if 'image_data' in data and data['image_data']:
            if not isinstance(data['image_data'], str):
                self.logger.warning("Image data must be a base64 string")
                return False
        
        return True
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema for this module.
        
        Returns:
            Dictionary describing the output schema
        """
        return {
            'detection_result': {
                'type': 'object',
                'properties': {
                    'prediction': {'type': 'string', 'enum': ['real', 'fake', 'error']},
                    'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                    'reasoning': {'type': 'string'}
                },
                'required': ['prediction', 'confidence']
            },
            'model_response': {'type': 'string'},
            'prompt_used': {'type': 'string'},
            'model_name': {'type': 'string'},
            'confidence': {'type': 'number'}
        }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response to extract prediction and confidence.
        
        Args:
            response: Raw model response
            
        Returns:
            Dictionary with prediction, confidence, and reasoning
        """
        try:
            # Simple parsing logic - can be enhanced based on prompt format
            response_lower = response.lower().strip()
            
            # Extract prediction
            if 'fake' in response_lower or 'false' in response_lower or 'misinformation' in response_lower:
                prediction = 'fake'
            elif 'real' in response_lower or 'true' in response_lower or 'accurate' in response_lower:
                prediction = 'real'
            else:
                prediction = 'error'
            
            # Extract confidence (simple heuristic)
            confidence = 0.5  # Default
            if 'very confident' in response_lower or 'highly confident' in response_lower:
                confidence = 0.9
            elif 'confident' in response_lower:
                confidence = 0.8
            elif 'somewhat confident' in response_lower:
                confidence = 0.6
            elif 'uncertain' in response_lower or 'unsure' in response_lower:
                confidence = 0.4
            
            # Apply confidence threshold
            if confidence < self.config['confidence_threshold']:
                prediction = 'error'
            
            result = {
                'prediction': prediction,
                'confidence': confidence
            }
            
            # Add reasoning if requested
            if self.config['include_reasoning']:
                result['reasoning'] = response
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'reasoning': f"Parse error: {e}"
            }