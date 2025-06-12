"""Model providers and routing functionality."""

from models.router import ModelRouter
from models.registry import ModelRegistry, get_registry, register_provider, create_client
from models.base_client import BaseModelClient

# Import providers to ensure they're registered
try:
    from models.openai.client import OpenAIClient
except ImportError:
    pass

try:
    from models.gemini.client import GeminiClient
except ImportError:
    pass

__all__ = [
    'ModelRouter',
    'ModelRegistry', 
    'BaseModelClient',
    'get_registry',
    'register_provider',
    'create_client'
]