"""Basic smoke tests for the Observ Python SDK."""

import pytest
from observ import Observ


def test_import():
    """Test that we can import the main Observ class."""
    assert Observ is not None


def test_observ_initialization():
    """Test that we can initialize Observ with basic parameters."""
    obs = Observ(
        api_key="test-key",
        project_id="test-project",
        debug=True
    )
    
    assert obs.api_key == "test-key"
    assert obs.project_id == "test-project"
    assert obs.debug is True
    assert obs.jwt_token is None


def test_jwt_token_methods():
    """Test JWT token storage and retrieval."""
    obs = Observ(api_key="test-key")
    
    # Test initial state
    assert obs.jwt_token is None
    assert obs.get_auth_header() == "Bearer test-key"
    
    # Test setting JWT token
    test_jwt = "eyJhbGciOiJIUzI1NiJ9.test"
    obs.set_jwt_token(test_jwt)
    assert obs.jwt_token == test_jwt
    assert obs.get_auth_header() == f"Bearer {test_jwt}"


def test_provider_imports():
    """Test that we can import provider wrappers."""
    from observ.providers.anthropic import AnthropicMessagesWrapper
    from observ.providers.openai import OpenAIChatCompletionsWrapper
    from observ.providers.mistral import MistralChatCompletionsWrapper
    from observ.providers.gemini import GeminiGenerateContentWrapper
    from observ.providers.xai import XAIChatCompletionsWrapper
    from observ.providers.openrouter import OpenRouterChatCompletionsWrapper
    
    assert AnthropicMessagesWrapper is not None
    assert OpenAIChatCompletionsWrapper is not None
    assert MistralChatCompletionsWrapper is not None
    assert GeminiGenerateContentWrapper is not None
    assert XAIChatCompletionsWrapper is not None
    assert OpenRouterChatCompletionsWrapper is not None