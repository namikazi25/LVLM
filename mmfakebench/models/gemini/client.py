"""Google Gemini model provider client.

This module provides a client for interacting with Google's Gemini models
through the LangChain interface.
"""

import logging
import time
from typing import Optional, Dict, Any, List

from models.base_client import BaseModelClient
from utils.image_processor import encode_image


class GeminiClient(BaseModelClient):
    """Client for Google Gemini models.
    
    This client handles communication with Google's Gemini API through LangChain,
    including multimodal capabilities for Gemini Pro Vision models.
    """
    
    def _init_client(self):
        """Initialize the Gemini client.
        
        Returns:
            Initialized ChatGoogleGenerativeAI client
            
        Raises:
            ValueError: If API key is missing or model is not supported
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ValueError("langchain_google_genai is required for Gemini models. Install with: pip install langchain-google-genai")
        
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini models")
        
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            **self.config
        )
    
    def create_multimodal_message(self, 
                                 system_prompt: str, 
                                 text_prompt: str, 
                                 image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create a multimodal message for Gemini models.
        
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
            
            # Gemini uses a different format for multimodal messages
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
                if any(term in error_msg for term in ["rate limit", "429", "quota", "resource_exhausted"]):
                    sleep_time = 2 ** attempt
                    logging.warning(
                        f"[Gemini] Rate limit error on attempt {attempt + 1}/{self.max_retries}. "
                        f"Sleeping {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                    continue
                
                # Handle image format errors
                elif any(term in error_msg for term in ["image format", "unsupported image", "invalid image"]):
                    logging.error("[Gemini] Unsupported/corrupt image payload sent to model.")
                    self.usage_stats['failed_calls'] += 1
                    return None
                
                # Handle safety filter errors
                elif any(term in error_msg for term in ["safety", "blocked", "harmful"]):
                    logging.error("[Gemini] Content blocked by safety filters.")
                    self.usage_stats['failed_calls'] += 1
                    return None
                
                # Handle other errors
                else:
                    logging.error(f"[Gemini] Model call failed on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:  # Last attempt
                        self.usage_stats['failed_calls'] += 1
                        return None
                    continue
        
        logging.error("[Gemini] Max retries exhausted.")
        self.usage_stats['failed_calls'] += 1
        return None
    
    def _generate_impl(self, 
                      prompt: str, 
                      image_data: Optional[str] = None,
                      system_prompt: Optional[str] = None,
                      **kwargs) -> Optional[str]:
        """Generate a response from the Gemini model.
        
        Args:
            prompt: Text prompt for the model
            image_data: Path to image file (for Gemini Pro Vision)
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
            logging.error(f"[Gemini] Failed to generate response: {e}")
            self.usage_stats['failed_calls'] += 1
            return None
    
    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """Estimate the cost of a Gemini request.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Estimated cost in USD
        """
        # Rough token estimation (4 characters per token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4
        
        # Gemini pricing (approximate, as of 2024)
        if 'gemini-pro' in self.model_name.lower():
            # Gemini Pro pricing
            cost = (input_tokens * 0.000000125) + (output_tokens * 0.000000375)
        elif 'gemini-ultra' in self.model_name.lower():
            # Gemini Ultra pricing (estimated)
            cost = (input_tokens * 0.000001) + (output_tokens * 0.000003)
        else:
            cost = 0.0  # Unknown model or free tier
        
        self.usage_stats['estimated_cost'] += cost
        return cost