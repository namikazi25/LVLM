#!/usr/bin/env python3
"""Test suite for Phase 4.1: Refactor Existing Providers.

This module tests the refactored model provider system including:
- Base client interface
- OpenAI and Gemini provider clients
- Model registry system
- Updated ModelRouter functionality
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the components to test
from mmfakebench.models.base_client import BaseModelClient
from mmfakebench.models.registry import ModelRegistry, get_registry
from mmfakebench.models.router import ModelRouter


class MockClient(BaseModelClient):
    """Mock client for testing purposes."""
    
    def _init_client(self):
        """Initialize mock client."""
        return Mock()
    
    def generate(self, prompt: str, image_data=None, system_prompt=None, **kwargs):
        """Mock generate method."""
        self.usage_stats['total_calls'] += 1
        self.usage_stats['successful_calls'] += 1
        return f"Mock response to: {prompt}"
    
    def create_multimodal_message(self, system_prompt: str, text_prompt: str, image_path=None):
        """Mock multimodal message creation."""
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": text_prompt}]
    
    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """Mock cost estimation."""
        cost = len(prompt) * 0.00001 + len(response) * 0.00002
        self.usage_stats['estimated_cost'] += cost
        return cost
    
    def _call_model_with_retry(self, messages):
        """Mock model call with retry."""
        return "Mock response"


def test_base_client_interface():
    """Test that BaseModelClient defines the correct interface."""
    # Test that BaseModelClient is abstract
    with pytest.raises(TypeError):
        BaseModelClient("test-model", "test-key")
    
    # Test that MockClient can be instantiated
    client = MockClient("test-model", "test-key")
    assert client.model_name == "test-model"
    assert client.api_key == "test-key"
    assert client.temperature == 0.2
    assert client.max_retries == 5
    
    # Test usage stats initialization
    stats = client.get_usage_stats()
    assert stats['total_calls'] == 0
    assert stats['successful_calls'] == 0
    assert stats['failed_calls'] == 0
    assert stats['estimated_cost'] == 0.0


def test_model_registry():
    """Test the model registry functionality."""
    registry = ModelRegistry()
    
    # Test provider registration
    registry.register_provider('mock', MockClient)
    assert 'mock' in registry.list_providers()
    
    # Test model mapping
    registry.register_model_mapping('mock-model', 'mock')
    assert registry.get_provider_for_model('mock-model') == 'mock'
    
    # Test client creation
    client = registry.create_client('mock-model', 'test-key')
    assert isinstance(client, MockClient)
    assert client.model_name == 'mock-model'
    assert client.api_key == 'test-key'
    
    # Test provider info
    info = registry.get_provider_info('mock')
    assert info['name'] == 'mock'
    assert info['client_class'] == 'MockClient'
    assert 'mock-model' in info['supported_models']


def test_global_registry():
    """Test the global registry functions."""
    from mmfakebench.models.registry import register_provider, create_client
    
    # Register a mock provider
    register_provider('test-mock', MockClient)
    
    # Test that it's available in the global registry
    global_registry = get_registry()
    assert 'test-mock' in global_registry.list_providers()


@patch('mmfakebench.models.openai.client.ChatOpenAI')
def test_openai_client(mock_chat_openai):
    """Test OpenAI client functionality."""
    try:
        from mmfakebench.models.openai.client import OpenAIClient
        
        # Mock the LangChain client
        mock_instance = Mock()
        mock_chat_openai.return_value = mock_instance
        
        # Create client
        client = OpenAIClient('gpt-4', 'test-key')
        assert client.model_name == 'gpt-4'
        assert client.api_key == 'test-key'
        
        # Test that ChatOpenAI was called with correct parameters
        mock_chat_openai.assert_called_once_with(
            api_key='test-key',
            model='gpt-4',
            temperature=0.2
        )
        
        # Test cost estimation
        cost = client.estimate_cost("test prompt", "test response")
        assert cost > 0
        
    except ImportError:
        pytest.skip("OpenAI dependencies not available")


@patch('mmfakebench.models.gemini.client.ChatGoogleGenerativeAI')
def test_gemini_client(mock_chat_gemini):
    """Test Gemini client functionality."""
    try:
        from mmfakebench.models.gemini.client import GeminiClient
        
        # Mock the LangChain client
        mock_instance = Mock()
        mock_chat_gemini.return_value = mock_instance
        
        # Create client
        client = GeminiClient('gemini-pro', 'test-key')
        assert client.model_name == 'gemini-pro'
        assert client.api_key == 'test-key'
        
        # Test that ChatGoogleGenerativeAI was called with correct parameters
        mock_chat_gemini.assert_called_once_with(
            model='gemini-pro',
            google_api_key='test-key',
            temperature=0.2
        )
        
        # Test cost estimation
        cost = client.estimate_cost("test prompt", "test response")
        assert cost >= 0  # Gemini might be free tier
        
    except ImportError:
        pytest.skip("Gemini dependencies not available")


def test_model_router_with_registry():
    """Test that ModelRouter works with the registry system."""
    # Register mock provider
    registry = get_registry()
    registry.register_provider('test-router', MockClient)
    registry.register_model_mapping('test-router-model', 'test-router')
    
    # Create router
    router = ModelRouter('test-router-model', 'test-key')
    assert router.model_name == 'test-router-model'
    assert router.api_key == 'test-key'
    
    # Test generation
    response = router.generate("Hello, world!")
    assert "Mock response" in response
    
    # Test usage stats
    stats = router.get_usage_stats()
    assert stats['total_calls'] > 0
    assert stats['successful_calls'] > 0
    
    # Test model switching
    registry.register_model_mapping('another-test-model', 'test-router')
    router.switch_model('another-test-model')
    assert router.model_name == 'another-test-model'


def test_model_router_info():
    """Test router information retrieval."""
    registry = get_registry()
    registry.register_provider('info-test', MockClient)
    registry.register_model_mapping('info-test-model', 'info-test')
    
    router = ModelRouter('info-test-model', 'test-key')
    info = router.get_info()
    
    assert 'router' in info
    assert 'client' in info
    assert 'usage_stats' in info
    
    assert info['router']['model_name'] == 'info-test-model'
    assert info['client']['provider'] == 'MockClient'


def test_error_handling():
    """Test error handling in the registry and router."""
    registry = ModelRegistry()
    
    # Test unsupported model
    with pytest.raises(ValueError, match="Cannot determine provider"):
        registry.create_client('unsupported-model', 'test-key')
    
    # Test invalid provider registration
    with pytest.raises(ValueError, match="must inherit from BaseModelClient"):
        registry.register_provider('invalid', str)  # str doesn't inherit from BaseModelClient
    
    # Test router with unsupported model
    with pytest.raises(ValueError):
        ModelRouter('completely-unknown-model', 'test-key')


def test_backward_compatibility():
    """Test that the refactored router maintains backward compatibility."""
    # Register mock provider for testing
    registry = get_registry()
    registry.register_provider('compat-test', MockClient)
    registry.register_model_mapping('compat-model', 'compat-test')
    
    # Test that old interface methods still work
    router = ModelRouter('compat-model', 'test-key')
    
    # Test llm_multimodal method
    response = router.llm_multimodal("You are helpful", "Hello")
    assert response is not None
    
    # Test get_model method
    model = router.get_model()
    assert model is not None
    
    # Test get_model_name method
    assert router.get_model_name() == 'compat-model'


if __name__ == "__main__":
    # Run basic tests
    print("Testing Phase 4.1 refactored providers...")
    
    try:
        test_base_client_interface()
        print("✅ Base client interface test passed")
    except Exception as e:
        print(f"❌ Base client interface test failed: {e}")
    
    try:
        test_model_registry()
        print("✅ Model registry test passed")
    except Exception as e:
        print(f"❌ Model registry test failed: {e}")
    
    try:
        test_global_registry()
        print("✅ Global registry test passed")
    except Exception as e:
        print(f"❌ Global registry test failed: {e}")
    
    try:
        test_model_router_with_registry()
        print("✅ Model router with registry test passed")
    except Exception as e:
        print(f"❌ Model router with registry test failed: {e}")
    
    try:
        test_error_handling()
        print("✅ Error handling test passed")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    try:
        test_backward_compatibility()
        print("✅ Backward compatibility test passed")
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
    
    print("\n🎉 Phase 4.1 refactoring tests completed!")