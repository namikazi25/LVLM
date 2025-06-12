"""Model registry for managing different model providers.

This module provides a centralized registry for all model providers,
allowing for easy discovery and instantiation of model clients.
"""

import logging
from typing import Dict, Type, List, Optional, Any
from .base_client import BaseModelClient


class ModelRegistry:
    """Registry for managing model providers and their capabilities.
    
    This class maintains a registry of all available model providers
    and provides methods for discovering and instantiating clients.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self._providers: Dict[str, Type[BaseModelClient]] = {}
        self._model_mappings: Dict[str, str] = {}
        self._register_default_providers()
    
    def _register_default_providers(self):
        """Register the default model providers."""
        try:
            from .openai.client import OpenAIClient
            self.register_provider('openai', OpenAIClient)
            
            # Register common OpenAI model patterns
            openai_models = [
                'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-1106',
                'gpt-4', 'gpt-4-turbo', 'gpt-4-vision-preview', 'gpt-4-1106-preview',
                'gpt-4o', 'gpt-4o-mini'
            ]
            for model in openai_models:
                self._model_mappings[model] = 'openai'
                
        except ImportError:
            logging.warning("OpenAI provider not available")
        
        try:
            from .gemini.client import GeminiClient
            self.register_provider('gemini', GeminiClient)
            
            # Register common Gemini model patterns
            gemini_models = [
                'gemini-pro', 'gemini-pro-vision', 'gemini-ultra',
                'gemini-1.0-pro', 'gemini-1.5-pro', 'gemini-1.5-flash'
            ]
            for model in gemini_models:
                self._model_mappings[model] = 'gemini'
                
        except ImportError:
            logging.warning("Gemini provider not available")
        
        # Mock provider removed - using real providers only
    
    def register_provider(self, name: str, client_class: Type[BaseModelClient]):
        """Register a new model provider.
        
        Args:
            name: Name of the provider (e.g., 'openai', 'gemini')
            client_class: Client class that implements BaseModelClient
            
        Raises:
            ValueError: If the client class doesn't inherit from BaseModelClient
        """
        if not issubclass(client_class, BaseModelClient):
            raise ValueError(f"Client class {client_class} must inherit from BaseModelClient")
        
        self._providers[name.lower()] = client_class
        logging.info(f"Registered provider: {name}")
    
    def register_model_mapping(self, model_name: str, provider_name: str):
        """Register a mapping from model name to provider.
        
        Args:
            model_name: Name of the model
            provider_name: Name of the provider
            
        Raises:
            ValueError: If the provider is not registered
        """
        if provider_name.lower() not in self._providers:
            raise ValueError(f"Provider {provider_name} is not registered")
        
        self._model_mappings[model_name.lower()] = provider_name.lower()
    
    def get_provider_for_model(self, model_name: str) -> Optional[str]:
        """Get the provider name for a given model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Provider name or None if not found
        """
        model_name = model_name.lower()
        
        # Direct mapping
        if model_name in self._model_mappings:
            return self._model_mappings[model_name]
        
        # Pattern matching
        for pattern, provider in self._model_mappings.items():
            if pattern in model_name or model_name.startswith(pattern):
                return provider
        
        # Fallback pattern matching
        if 'gpt' in model_name:
            return 'openai'
        elif 'gemini' in model_name:
            return 'gemini'
        elif 'claude' in model_name:
            return 'anthropic'
        
        return None
    
    def create_client(self, 
                     model_name: str,
                     api_key: str,
                     provider_name: Optional[str] = None,
                     **kwargs) -> BaseModelClient:
        """Create a client for the specified model.
        
        Args:
            model_name: Name of the model
            api_key: API key for the provider
            provider_name: Optional provider name (auto-detected if not provided)
            **kwargs: Additional client parameters
            
        Returns:
            Initialized model client
            
        Raises:
            ValueError: If provider cannot be determined or is not supported
        """
        if provider_name:
            provider = provider_name.lower()
        else:
            provider = self.get_provider_for_model(model_name)
        
        if not provider:
            raise ValueError(f"Cannot determine provider for model: {model_name}")
        
        if provider not in self._providers:
            raise ValueError(f"Provider {provider} is not registered")
        
        client_class = self._providers[provider]
        return client_class(
            model_name=model_name,
            api_key=api_key,
            **kwargs
        )
    
    def list_providers(self) -> List[str]:
        """List all registered providers.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def list_models(self, provider_name: Optional[str] = None) -> List[str]:
        """List all registered models.
        
        Args:
            provider_name: Optional provider name to filter by
            
        Returns:
            List of model names
        """
        if provider_name:
            provider = provider_name.lower()
            return [model for model, prov in self._model_mappings.items() if prov == provider]
        else:
            return list(self._model_mappings.keys())
    
    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get information about a provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Dictionary containing provider information
            
        Raises:
            ValueError: If provider is not registered
        """
        provider = provider_name.lower()
        if provider not in self._providers:
            raise ValueError(f"Provider {provider_name} is not registered")
        
        client_class = self._providers[provider]
        models = self.list_models(provider)
        
        return {
            'name': provider,
            'client_class': client_class.__name__,
            'module': client_class.__module__,
            'supported_models': models,
            'model_count': len(models)
        }
    
    def is_model_supported(self, model_name: str) -> bool:
        """Check if a model is supported.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if the model is supported, False otherwise
        """
        return self.get_provider_for_model(model_name) is not None


# Global registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry instance.
    
    Returns:
        Global ModelRegistry instance
    """
    return _registry


def register_provider(name: str, client_class: Type[BaseModelClient]):
    """Register a provider with the global registry.
    
    Args:
        name: Name of the provider
        client_class: Client class that implements BaseModelClient
    """
    _registry.register_provider(name, client_class)


def create_client(model_name: str, api_key: str, **kwargs) -> BaseModelClient:
    """Create a client using the global registry.
    
    Args:
        model_name: Name of the model
        api_key: API key for the provider
        **kwargs: Additional client parameters
        
    Returns:
        Initialized model client
    """
    return _registry.create_client(model_name, api_key, **kwargs)