"""Image-Headline Relevancy Checker Module.

This module contains the ImageHeadlineRelevancyChecker class that analyzes
the relationship between news images and headlines to detect potential mismatches.
"""

import os
import math
import time
import logging
from typing import Dict, Any, Tuple, Optional

from core.base import BasePipelineModule
from models.router import ModelRouter


class ImageHeadlineRelevancyChecker(BasePipelineModule):
    """Pipeline module for checking relevancy between images and headlines.
    
    This module analyzes whether an image could reasonably illustrate
    the content described in a news headline.
    """
    
    def __init__(self, 
                 name: str = "relevance_checker",
                 llm_provider: str = "openai", 
                 model_name: str = "gpt-4-vision-preview", 
                 show_workflow: bool = False,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: int = 30,
                 **kwargs):
        """Initialize the relevance checker.
        
        Args:
            name: Name of the module
            llm_provider: The LLM provider to use
            model_name: The specific model to use
            show_workflow: Whether to show workflow steps
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
            **kwargs: Additional configuration parameters
        """
        config = {
            'llm_provider': llm_provider,
            'model_name': model_name,
            'show_workflow': show_workflow,
            'max_retries': max_retries,
            'retry_delay': retry_delay,
            'timeout': timeout,
            **kwargs
        }
        super().__init__(name=name, config=config)
        
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.model_router: Optional[ModelRouter] = None
        self.show_workflow = show_workflow
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Load prompt template
        from core.prompts import get_prompt_manager
        self.prompt_manager = get_prompt_manager()
        self.template_name = "relevance_check"
    
    def initialize(self) -> None:
        """Initialize the module with its configuration."""
        try:
            # Initialize model router if not provided
            if not self.model_router:
                from models.router import ModelRouter
                import os
                
                # Get API key from config or environment
                api_key = (
                    self.config.get('api_key') or
                    self.config.get('openai_api_key') or
                    os.getenv('OPENAI_API_KEY') or
                    os.getenv('GEMINI_API_KEY') or
                    'test-key-for-demo'  # Fallback for testing
                )
                
                self.model_router = ModelRouter(
                    model_name=self.model_name,
                    api_key=api_key,
                    temperature=self.config.get('temperature', 0.2),
                    max_retries=self.max_retries,
                    provider_name=self.llm_provider
                )
            
            self.logger.info(f"Initialized {self.name} with {self.llm_provider}:{self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            raise RuntimeError(f"Module initialization failed: {e}")
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format.
        
        Args:
            data: Input data dictionary
            
        Returns:
            True if input contains required fields, False otherwise
        """
        required_fields = ['image_path', 'headline']
        return all(field in data for field in required_fields)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process image-headline relevancy check.
        
        Args:
            data: Input data containing 'image_path' and 'headline'
            
        Returns:
            Dictionary containing relevancy check results
        """
        image_path = data['image_path']
        headline = data['headline']
        
        response_content, confidence, success = self.check_relevancy(
            image_path, headline, self.max_retries
        )
        
        return {
            **data,
            'relevancy_response': response_content,
            'relevancy_confidence': confidence,
            'relevancy_check_success': success,
            'potential_mismatch': 'Yes, potential mismatch:' in response_content
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema for this module.
        
        Returns:
            Dictionary describing the output schema
        """
        return {
            'relevancy_response': 'str - The model\'s response about image-headline relevancy',
            'relevancy_confidence': 'float - Confidence score (0.0-1.0)',
            'relevancy_check_success': 'bool - Whether the check completed successfully',
            'potential_mismatch': 'bool - Whether a potential mismatch was detected'
        }
    
    def check_relevancy(self, image_path: str, headline: str, max_retries: int = 3) -> Tuple[str, float, bool]:
        """Check relevancy between image and headline.
        
        Args:
            image_path: Path to the image file
            headline: News headline text
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_content, confidence, success)
        """
        if not image_path or not os.path.isfile(image_path):
            self.logger.warning(f"Skipping relevancy check due to missing image file: {image_path}")
            return "Image File Not Found", 0.0, False
        
        if not headline or not headline.strip():
            self.logger.warning(f"Skipping relevancy check due to empty headline for image: {image_path}")
            return "Empty Headline Error", 0.0, False
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Relevancy check attempt {attempt + 1}/{max_retries} for image: {image_path}")
                
                # Render prompt template
                system_prompt = self.prompt_manager.render_template(
                    self.template_name, 
                    {}
                )
                
                # Use centralized router for prompt, image handling, retries, provider quirks
                response = self.model_router.llm_multimodal(
                    system_prompt,
                    f"Headline: {headline}",
                    image_path
                )
                
                if not response:
                    raise RuntimeError("No response from model.")
                
                # Robustly get text content
                response_content = getattr(response, "content", None)
                if not response_content and isinstance(response, dict):
                    response_content = response.get("text")
                if not response_content:
                    raise RuntimeError("No content in model response.")
                response_content = response_content.strip()
                
                # Calculate confidence if available
                confidence = 0.0
                if hasattr(response, 'response_metadata') and "logprobs" in getattr(response, "response_metadata", {}):
                    logprobs = response.response_metadata.get("logprobs", {}).get("content", None)
                    if logprobs:
                        logprob_values = [token["logprob"] for token in logprobs if "logprob" in token]
                        if logprob_values:
                            avg_logprob = sum(logprob_values) / len(logprob_values)
                            confidence = math.exp(avg_logprob)
                
                # Check format
                if not ("No, the image is appropriately used." in response_content or
                        "Yes, potential mismatch:" in response_content):
                    self.logger.warning(f"Relevancy check got unexpected format: {response_content}")
                
                self.logger.info(f"Relevancy check successful on attempt {attempt + 1}")
                return response_content, confidence, True
                
            except Exception as e:
                self.logger.warning(f"Relevancy check attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.debug(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {max_retries} attempts failed for relevancy check")
        
        return f"API Error: Failed after {max_retries} attempts", 0.0, False