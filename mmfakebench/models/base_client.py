"""Base client interface for model providers.

This module defines the abstract base class that all model provider clients
must implement to ensure consistent interface across different providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
import logging
from .management import PerformanceMonitor, RateLimiter, QuotaManager, RateLimitConfig, QuotaConfig


class BaseModelClient(ABC):
    """Abstract base class for model provider clients.
    
    All model provider clients must inherit from this class and implement
    the required methods to ensure consistent interface.
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 temperature: float = 0.2,
                 max_retries: int = 5,
                 enable_monitoring: bool = True,
                 enable_rate_limiting: bool = True,
                 enable_quota_management: bool = True,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 quota_config: Optional[QuotaConfig] = None,
                 **kwargs):
        """Initialize the base client.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the provider
            temperature: Temperature for model generation
            max_retries: Maximum number of retries for failed requests
            enable_monitoring: Whether to enable performance monitoring
            enable_rate_limiting: Whether to enable rate limiting
            enable_quota_management: Whether to enable quota management
            rate_limit_config: Custom rate limit configuration
            quota_config: Custom quota configuration
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.config = kwargs
        
        # Track usage statistics
        self.usage_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'estimated_cost': 0.0
        }
        
        # Initialize management components
        self.enable_monitoring = enable_monitoring
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_quota_management = enable_quota_management
        
        if self.enable_monitoring:
            self.performance_monitor = PerformanceMonitor()
        
        if self.enable_rate_limiting:
            self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        
        if self.enable_quota_management:
            self.quota_manager = QuotaManager(quota_config or QuotaConfig())
        
        # Initialize the underlying client
        self._client = self._init_client()
    
    @abstractmethod
    def _init_client(self):
        """Initialize the provider-specific client.
        
        Returns:
            Initialized client instance
            
        Raises:
            ValueError: If initialization fails
        """
        pass
    
    def generate(self, 
                prompt: str, 
                image_data: Optional[str] = None,
                system_prompt: Optional[str] = None,
                **kwargs) -> Optional[str]:
        """Generate a response from the model with management features.
        
        Args:
            prompt: Text prompt for the model
            image_data: Path to image file (for multimodal models)
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response or None if failed
            
        Raises:
            Exception: If rate limits or quotas are exceeded
        """
        # Estimate tokens and cost for pre-checks
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimation
        estimated_cost = self.estimate_cost(prompt)
        
        # Check rate limits
        if self.enable_rate_limiting:
            can_proceed, reason = self.rate_limiter.can_make_request(int(estimated_tokens))
            if not can_proceed:
                raise Exception(f"Rate limit exceeded: {reason}")
        
        # Check quotas
        if self.enable_quota_management:
            can_proceed, reason = self.quota_manager.can_make_request(estimated_cost)
            if not can_proceed:
                raise Exception(f"Quota exceeded: {reason}")
        
        # Perform the actual generation with monitoring
        start_time = time.time()
        success = False
        response = None
        error_type = None
        actual_tokens = 0
        actual_cost = 0.0
        
        try:
            response = self._generate_impl(prompt, image_data, system_prompt, **kwargs)
            success = True
            
            # Calculate actual metrics
            actual_tokens = len(prompt.split()) + len(response.split()) if response else 0
            actual_cost = self.estimate_cost(prompt, response)
            
            # Update usage stats
            self.usage_stats['total_calls'] += 1
            self.usage_stats['successful_calls'] += 1
            self.usage_stats['total_tokens'] += actual_tokens
            self.usage_stats['estimated_cost'] += actual_cost
            
        except Exception as e:
            error_type = type(e).__name__
            self.usage_stats['total_calls'] += 1
            self.usage_stats['failed_calls'] += 1
            raise
        
        finally:
            latency = time.time() - start_time
            
            # Record rate limiter usage
            if self.enable_rate_limiting:
                self.rate_limiter.record_request(actual_tokens)
            
            # Record quota usage
            if self.enable_quota_management and success:
                self.quota_manager.record_usage(actual_cost)
            
            # Record performance metrics
            if self.enable_monitoring:
                self.performance_monitor.record_request(
                    model_name=self.model_name,
                    success=success,
                    latency=latency,
                    tokens_used=actual_tokens,
                    cost=actual_cost,
                    error_type=error_type
                )
        
        return response
    
    def estimate_cost(self, prompt: str, response: str = None) -> float:
        """Estimate the cost of a request based on token usage.
        
        This is a basic implementation that can be overridden by specific providers
        for more accurate cost estimation.
        
        Args:
            prompt: Input prompt
            response: Generated response (if available)
            
        Returns:
            Estimated cost in USD
        """
        # Basic estimation: $0.002 per 1K tokens (GPT-3.5-turbo pricing as baseline)
        prompt_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        response_tokens = len(response.split()) * 1.3 if response else 0
        total_tokens = prompt_tokens + response_tokens
        return (total_tokens / 1000) * 0.002
    
    @abstractmethod
    def _generate_impl(self, 
                      prompt: str, 
                      image_data: Optional[str] = None,
                      system_prompt: Optional[str] = None,
                      **kwargs) -> Optional[str]:
        """Implementation-specific generation method.
        
        This method should be implemented by each provider client.
        
        Args:
            prompt: Text prompt for the model
            image_data: Path to image file (for multimodal models)
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response or None if failed
        """
        pass
    
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """Estimate the cost of a request.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    def get_model_name(self) -> str:
        """Get the current model name.
        
        Returns:
            Current model name
        """
        return self.model_name
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Dictionary containing usage statistics
        """
        return self.usage_stats.copy()
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'estimated_cost': 0.0
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get client information.
        
        Returns:
            Dictionary containing client metadata
        """
        return {
            'provider': self.__class__.__name__,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_retries': self.max_retries,
            'config': self.config,
            'usage_stats': self.get_usage_stats()
        }