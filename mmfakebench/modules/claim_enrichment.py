"""Claim Enrichment Module.

This module contains the ClaimEnrichmentTool class that processes images and headlines
to extract restated claims and detailed image context descriptions.
"""

import os
import re
import time
import logging
from typing import Dict, Any, Tuple, Optional

from core.base import BasePipelineModule
from models.router import ModelRouter


class ClaimEnrichmentTool(BasePipelineModule):
    """Pipeline module for enriching claims with image context.
    
    This module processes news images and headlines to extract:
    1. A restated version of the core claim from the headline
    2. Detailed visual context description of the image
    """
    
    def __init__(self, 
                 name: str = "claim_enrichment",
                 llm_provider: str = "openai", 
                 model_name: str = "gpt-4-vision-preview",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: int = 30,
                 **kwargs):
        """Initialize the claim enrichment tool.
        
        Args:
            name: Name of the module
            llm_provider: The LLM provider to use
            model_name: The specific model to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
            **kwargs: Additional configuration parameters
        """
        config = {
            'llm_provider': llm_provider,
            'model_name': model_name,
            'max_retries': max_retries,
            'retry_delay': retry_delay,
            'timeout': timeout,
            **kwargs
        }
        super().__init__(name=name, config=config)
        
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.model_router: Optional[ModelRouter] = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Load prompt template
        from core.prompts import get_prompt_manager
        self.prompt_manager = get_prompt_manager()
        self.template_name = "claim_enrichment"
    
    def initialize(self) -> None:
        """Initialize the module with its configuration."""
        try:
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
            self.logger.info(f"Initialized {self.name} with model router")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            raise
    
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
        """Process claim enrichment.
        
        Args:
            data: Input data containing 'image_path' and 'headline'
            
        Returns:
            Dictionary containing enriched claim and image context
        """
        image_path = data['image_path']
        headline = data['headline']
        
        restated_claim, image_context, success = self.run(image_path, headline)
        
        return {
            **data,
            'enriched_claim': restated_claim,
            'image_context': image_context,
            'claim_enrichment_success': success
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema for this module.
        
        Returns:
            Dictionary describing the output schema
        """
        return {
            'enriched_claim': 'str - Restated version of the headline claim',
            'image_context': 'str - Detailed description of image visual elements',
            'claim_enrichment_success': 'bool - Whether the enrichment completed successfully'
        }
    
    def run(self, image_path: str, headline: str) -> Tuple[str, str, bool]:
        """Run claim enrichment on image and headline.
        
        Args:
            image_path: Path to the image file
            headline: News headline text
            
        Returns:
            Tuple of (restated_claim, image_context, success)
        """
        if not image_path or not os.path.isfile(image_path):
            self.logger.warning(f"Image file missing: {image_path}")
            return "Image File Not Found", "", False
        
        if not headline or not headline.strip():
            self.logger.warning("Headline is empty.")
            return "Empty Headline", "", False
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Claim enrichment attempt {attempt + 1}/{self.max_retries}")
                
                # Render prompt template
                system_prompt = self.prompt_manager.render_template(
                    self.template_name, 
                    {}
                )
                
                response = self.model_router.llm_multimodal(
                    system_prompt,
                    f"Headline: {headline}",
                    image_path
                )
                
                # Robustly get text content
                text = getattr(response, "content", None)
                if not text and isinstance(response, dict):
                    text = response.get("text")
                if not text:
                    raise RuntimeError("No response content from model.")
                text = text.strip()
                
                # Parse the structured response
                claim_match = re.search(r"RESTATED CLAIM:\s*(.*?)\s*IMAGE CONTEXT:", text, re.DOTALL | re.IGNORECASE)
                context_match = re.search(r"IMAGE CONTEXT:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
                
                restated_claim = claim_match.group(1).strip() if claim_match else ""
                image_context = context_match.group(1).strip() if context_match else ""
                
                if not restated_claim and not image_context:
                    # Fallback: try to extract any useful content
                    self.logger.warning(f"Could not parse structured response, using raw text: {text[:200]}...")
                    return text, "", True
                
                self.logger.info(f"Claim enrichment successful on attempt {attempt + 1}")
                return restated_claim, image_context, True
                
            except Exception as e:
                self.logger.warning(f"Claim enrichment attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    self.logger.debug(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed for claim enrichment")
        
        return f"API Error: Failed after {self.max_retries} attempts", "", False