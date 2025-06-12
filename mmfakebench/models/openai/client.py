"""OpenAI model provider client.

This module provides a client for interacting with OpenAI's GPT models
through the LangChain interface.
"""

import logging
import time
from typing import Optional, Dict, Any, List

from models.base_client import BaseModelClient
from utils.image_processor import encode_image


class OpenAIClient(BaseModelClient):
    """Client for OpenAI GPT models.
    
    This client handles communication with OpenAI's API through LangChain,
    including multimodal capabilities for GPT-4 Vision models.
    """
    
    def _init_client(self):
        """Initialize the OpenAI client.
        
        Returns:
            Initialized ChatOpenAI client
            
        Raises:
            ValueError: If API key is missing or model is not supported
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ValueError("langchain_openai is required for OpenAI models. Install with: pip install langchain-openai")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required for GPT models")
        
        return ChatOpenAI(
            api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature,
            **self.config
        )
    
    def create_multimodal_message(self, 
                                 system_prompt: str, 
                                 text_prompt: str, 
                                 image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create a multimodal message for OpenAI models.
        
        Args:
            system_prompt: System prompt for the model
            text_prompt: User text prompt
            image_path: Optional path to image file
            
        Returns:
            List of LangChain message objects
            
        Raises:
            ValueError: If image encoding fails
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = [SystemMessage(content=system_prompt)]
        
        if image_path:
            base64_image, mime_type = encode_image(image_path)
            if not base64_image:
                raise ValueError(f"Failed to encode image: {image_path}")
            
            messages.append(HumanMessage(content=[
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]))
        else:
            messages.append(HumanMessage(content=text_prompt))
        
        return messages
    
    def _call_model_with_retry(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Call the model with retry logic.
        
        Args:
            messages: List of message objects
            
        Returns:
            Model response text or None if failed
        """
        self.usage_stats['total_calls'] += 1
        
        for attempt in range(self.max_retries):
            try:
                response = self._client.invoke(messages)
                self.usage_stats['successful_calls'] += 1
                
                # Extract text content from response
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limiting
                if any(term in error_msg for term in ["rate limit", "429", "quota", "please try again"]):
                    sleep_time = 2 ** attempt
                    logging.warning(
                        f"[OpenAI] Rate limit error on attempt {attempt + 1}/{self.max_retries}. "
                        f"Sleeping {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                    continue
                
                # Handle image format errors
                elif any(term in error_msg for term in ["image format", "unsupported image"]):
                    logging.error("[OpenAI] Unsupported/corrupt image payload sent to model.")
                    self.usage_stats['failed_calls'] += 1
                    return None
                
                # Handle other errors
                else:
                    logging.error(f"[OpenAI] Model call failed on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:  # Last attempt
                        self.usage_stats['failed_calls'] += 1
                        return None
                    continue
        
        logging.error("[OpenAI] Max retries exhausted.")
        self.usage_stats['failed_calls'] += 1
        return None
    
    def _generate_impl(self, 
                      prompt: str, 
                      image_data: Optional[str] = None,
                      system_prompt: Optional[str] = None,
                      **kwargs) -> Optional[str]:
        """Generate a response from the OpenAI model.
        
        Args:
            prompt: Text prompt for the model
            image_data: Path to image file (for GPT-4 Vision)
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response or None if failed
        """
        try:
            sys_prompt = system_prompt or "You are a helpful AI assistant."
            messages = self.create_multimodal_message(sys_prompt, prompt, image_data)
            return self._call_model_with_retry(messages)
        except Exception as e:
            logging.error(f"[OpenAI] Failed to generate response: {e}")
            self.usage_stats['failed_calls'] += 1
            return None
    
    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """Estimate the cost of an OpenAI request.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Estimated cost in USD
        """
        # Rough token estimation (4 characters per token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4
        
        # OpenAI pricing (approximate, as of 2024)
        if 'gpt-4' in self.model_name.lower():
            if 'vision' in self.model_name.lower() or 'turbo' in self.model_name.lower():
                cost = (input_tokens * 0.00001) + (output_tokens * 0.00003)  # GPT-4 Turbo
            else:
                cost = (input_tokens * 0.00003) + (output_tokens * 0.00006)  # GPT-4
        elif 'gpt-3.5' in self.model_name.lower():
            cost = (input_tokens * 0.0000015) + (output_tokens * 0.000002)  # GPT-3.5 Turbo
        else:
            cost = 0.0  # Unknown model
        
        self.usage_stats['estimated_cost'] += cost
        return cost