"""Mock model client for testing purposes.

This module provides a mock implementation of the model client interface
for testing and development when real API keys are not available.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from .base_client import BaseModelClient


class MockClient(BaseModelClient):
    """Mock model client for testing purposes.
    
    This client simulates model responses without making actual API calls,
    useful for testing and development scenarios.
    """
    
    def __init__(self, api_key: str = "mock-key", **kwargs):
        """Initialize the mock client.
        
        Args:
            api_key: Mock API key (ignored)
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(api_key, **kwargs)
        self.model_name = kwargs.get('model_name', 'mock-model')
        self.temperature = kwargs.get('temperature', 0.2)
        
    def generate_response(self, 
                         messages: List[Dict[str, Any]], 
                         **kwargs) -> Dict[str, Any]:
        """Generate a mock response.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional generation parameters
            
        Returns:
            Mock response dictionary
        """
        # Simulate processing time
        time.sleep(0.1)
        
        # Generate mock response based on the last message
        last_message = messages[-1] if messages else {"content": ""}
        content = last_message.get("content", "")
        
        # Simple mock logic for misinformation detection
        if "fake" in content.lower() or "false" in content.lower():
            prediction = "fake"
            confidence = 0.85
        elif "real" in content.lower() or "true" in content.lower():
            prediction = "real"
            confidence = 0.82
        else:
            prediction = "uncertain"
            confidence = 0.65
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": f"Mock analysis of content: {content[:50]}...",
            "model_info": {
                "name": self.model_name,
                "version": "mock-1.0",
                "temperature": self.temperature
            },
            "usage": {
                "prompt_tokens": len(content.split()),
                "completion_tokens": 10,
                "total_tokens": len(content.split()) + 10
            },
            "cost": 0.001  # Mock cost
        }
    
    def validate_connection(self) -> bool:
        """Validate the mock connection.
        
        Returns:
            Always True for mock client
        """
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information.
        
        Returns:
            Mock model information dictionary
        """
        return {
            "name": self.model_name,
            "provider": "mock",
            "version": "mock-1.0",
            "capabilities": ["text", "vision"],
            "max_tokens": 4096
        }
    
    def estimate_cost(self, 
                     prompt_tokens: int, 
                     completion_tokens: int) -> float:
        """Estimate mock cost.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Mock cost estimate
        """
        return (prompt_tokens + completion_tokens) * 0.0001