"""Model router for managing different LLM providers.

This module provides a unified interface for interacting with different
model providers like OpenAI, Google Gemini, etc.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from core.base import BaseClient
from models.registry import get_registry
from models.base_client import BaseModelClient


class ModelRouter:
    """Router for managing and switching between different model providers.
    
    This class provides a unified interface for interacting with different
    LLM providers while handling retries, rate limiting, and error recovery.
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 temperature: float = 0.2,
                 max_retries: int = 5,
                 provider_name: Optional[str] = None,
                 **kwargs):
        """Initialize the model router.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the model provider
            temperature: Temperature for model generation
            max_retries: Maximum number of retries for failed requests
            provider_name: Optional provider name (auto-detected if not provided)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name.lower()
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.provider_name = provider_name
        self.config = kwargs
        self.registry = get_registry()
        
        # Initialize the client using the registry
        self._client = self._init_client()
    
    def _init_client(self) -> BaseModelClient:
        """Initialize the appropriate client based on model name.
        
        Returns:
            Initialized model client
            
        Raises:
            ValueError: If model is not supported
        """
        # Determine provider if not explicitly provided
        if not self.provider_name:
            self.provider_name = self.registry.get_provider_for_model(self.model_name)
            if not self.provider_name:
                raise ValueError(f"Cannot determine provider for model: {self.model_name}")
        
        return self.registry.create_client(
            model_name=self.model_name,
            api_key=self.api_key,
            provider_name=self.provider_name,
            temperature=self.temperature,
            max_retries=self.max_retries,
            **self.config
        )
    
    def get_model(self):
        """Get the underlying model client.
        
        Returns:
            The initialized model client
        """
        return self._client
    
    def get_model_name(self) -> str:
        """Get the current model name.
        
        Returns:
            Current model name
        """
        return self.model_name
    
    def switch_model(self, new_model_name: str, new_api_key: Optional[str] = None, **kwargs):
        """Switch to a different model.
        
        Args:
            new_model_name: Name of the new model
            new_api_key: Optional new API key (uses current if not provided)
            **kwargs: Additional model-specific parameters
        """
        old_model = self.model_name
        old_api_key = self.api_key
        old_config = self.config.copy()
        
        self.model_name = new_model_name.lower()
        if new_api_key:
            self.api_key = new_api_key
        self.config.update(kwargs)
        
        try:
            self._client = self._init_client()
            logging.info(f"Successfully switched from {old_model} to {self.model_name}")
        except Exception as e:
            # Revert to old model if switch fails
            self.model_name = old_model
            self.api_key = old_api_key
            self.config = old_config
            self._client = self._init_client()
            logging.error(f"Failed to switch to {new_model_name}, reverted to {old_model}: {e}")
            raise
    
    def create_multimodal_message(self, 
                                 system_prompt: str, 
                                 text_prompt: str, 
                                 image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create a multimodal message for the model.
        
        Args:
            system_prompt: System prompt for the model
            text_prompt: User text prompt
            image_path: Optional path to image file
            
        Returns:
            List of message dictionaries
            
        Raises:
            ValueError: If image encoding fails
        """
        return self._client.create_multimodal_message(system_prompt, text_prompt, image_path)
    
    def call_model(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Call the model with the provided messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Model response text or None if failed
        """
        # Delegate to the client's internal retry logic
        return self._client._call_model_with_retry(messages)
    
    def llm_multimodal(self, 
                      system_prompt: str, 
                      text: str, 
                      image_path: Optional[str] = None) -> Optional[str]:
        """Main interface for multimodal LLM calls.
        
        Args:
            system_prompt: System prompt for the model
            text: User text input
            image_path: Optional path to image file
            
        Returns:
            Model response text or None if failed
        """
        return self._client.generate(text, image_path, system_prompt)
    
    def generate(self, 
                prompt: str, 
                image_data: Optional[str] = None,
                system_prompt: Optional[str] = None,
                **kwargs) -> Optional[str]:
        """Generate a response from the model (BaseClient interface).
        
        Args:
            prompt: Text prompt for the model
            image_data: Path to image file (for compatibility)
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        return self._client.generate(prompt, image_data, system_prompt, **kwargs)
    
    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """Estimate the cost of a request.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Estimated cost in USD
        """
        return self._client.estimate_cost(prompt, response)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Dictionary containing usage statistics
        """
        return self._client.get_usage_stats()
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self._client.reset_usage_stats()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the router and client.
        
        Returns:
            Dictionary containing router and client information
        """
        info = {
            'router': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_retries': self.max_retries,
                'provider_name': self.provider_name,
                'config': self.config
            },
            'client': self._client.get_info() if hasattr(self._client, 'get_info') else {},
            'usage_stats': self._client.usage_stats if hasattr(self._client, 'usage_stats') else {}
        }
        
        # Add management information if available
        if hasattr(self._client, 'enable_monitoring') and self._client.enable_monitoring:
            info['performance_metrics'] = self.get_performance_metrics()
        
        if hasattr(self._client, 'enable_rate_limiting') and self._client.enable_rate_limiting:
            info['rate_limit_status'] = self.get_rate_limit_status()
        
        if hasattr(self._client, 'enable_quota_management') and self._client.enable_quota_management:
            info['quota_status'] = self.get_quota_status()
        
        return info
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the client.
        
        Returns:
            Dictionary containing performance metrics
        """
        if hasattr(self._client, 'performance_monitor'):
            return self._client.performance_monitor.get_metrics()
        return {}
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limit status from the client.
        
        Returns:
            Dictionary containing rate limit status
        """
        if hasattr(self._client, 'rate_limiter'):
            return self._client.rate_limiter.get_status()
        return {}
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get quota status from the client.
        
        Returns:
            Dictionary containing quota status
        """
        if hasattr(self._client, 'quota_manager'):
            return self._client.quota_manager.get_status()
        return {}
    
    def reset_rate_limits(self) -> bool:
        """Reset rate limits for the client.
        
        Returns:
            True if reset was successful, False otherwise
        """
        if hasattr(self._client, 'rate_limiter'):
            self._client.rate_limiter.reset()
            return True
        return False
    
    def update_quota(self, new_quota_config) -> bool:
        """Update quota configuration for the client.
        
        Args:
            new_quota_config: New quota configuration
            
        Returns:
            True if update was successful, False otherwise
        """
        if hasattr(self._client, 'quota_manager'):
            self._client.quota_manager.update_config(new_quota_config)
            return True
        return False