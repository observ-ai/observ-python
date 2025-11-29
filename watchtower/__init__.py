"""Observ Python SDK"""

from typing import Any
import httpx
import anthropic

from .providers import (
    AnthropicMessagesWrapper,
    OpenAIChatCompletionsWrapper,
    GeminiGenerateContentWrapper,
    XAIChatCompletionsWrapper,
    MistralChatCompletionsWrapper,
    OpenRouterChatCompletionsWrapper,
)


class Observ:
    """Main Observ SDK class for tracing AI provider calls"""

    def __init__(self, api_key: str, project_id: str = "default", recall: bool = False, environment: str = "production"):
        self.api_key = api_key
        self.project_id = project_id
        self.recall = recall
        self.environment = environment
        self.endpoint = "http://localhost:8080"
        self.http_client = httpx.Client(timeout=30.0)

    def anthropic(self, client: anthropic.Anthropic) -> anthropic.Anthropic:
        """Wrap an Anthropic client to route through Observ gateway"""
        client.messages = AnthropicMessagesWrapper(client.messages, self)
        return client

    def openai(self, client) -> Any:
        """Wrap an OpenAI client to route through Observ gateway"""
        try:
            import openai
            if isinstance(client, openai.OpenAI):
                client.chat.completions = OpenAIChatCompletionsWrapper(client.chat.completions, self)
                return client
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        raise TypeError("Expected openai.OpenAI client")

    def gemini(self, model) -> Any:
        """Wrap a Gemini GenerativeModel to route through Observ gateway"""
        try:
            import google.generativeai as genai
            if isinstance(model, genai.GenerativeModel):
                # Monkey patch the generate_content method
                wrapper = GeminiGenerateContentWrapper(model, self)
                model.generate_content = wrapper.generate_content
                # Preserve with_metadata and with_session_id chaining
                model.with_metadata = wrapper.with_metadata
                model.with_session_id = wrapper.with_session_id
                return model
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
        raise TypeError("Expected google.generativeai.GenerativeModel")

    def xai(self, client) -> Any:
        """Wrap an xAI client (using OpenAI SDK) to route through Observ gateway"""
        try:
            import openai
            if isinstance(client, openai.OpenAI):
                client.chat.completions = XAIChatCompletionsWrapper(client.chat.completions, self)
                return client
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        raise TypeError("Expected openai.OpenAI client configured for xAI")

    def mistral(self, client) -> Any:
        """Wrap a Mistral client to route through Observ gateway"""
        try:
            import mistralai
            if isinstance(client, mistralai.Mistral):
                client.chat.completions = MistralChatCompletionsWrapper(client.chat.completions, self)
                return client
        except ImportError:
            raise ImportError("mistralai package is required. Install with: pip install mistralai")
        raise TypeError("Expected mistralai.Mistral client")

    def openrouter(self, client) -> Any:
        """Wrap an OpenRouter client (using OpenAI SDK) to route through Observ gateway"""
        try:
            import openai
            if isinstance(client, openai.OpenAI):
                client.chat.completions = OpenRouterChatCompletionsWrapper(client.chat.completions, self)
                return client
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        raise TypeError("Expected openai.OpenAI client configured for OpenRouter")

    def _send_callback(self, trace_id: str, response, duration_ms: int):
        """Send completion callback to gateway for Anthropic responses (non-blocking)"""
        try:
            content = ""
            if hasattr(response, 'content') and len(response.content) > 0:
                content = response.content[0].text

            tokens_used = 0
            if hasattr(response, 'usage'):
                tokens_used = getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)

            callback = {
                "trace_id": trace_id,
                "content": content,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used
            }

            self.http_client.post(
                f"{self.endpoint}/api/v1/llm/callback",
                json=callback,
                timeout=5.0
            )
        except Exception as e:
            print(f"Observ callback error: {e}")

    def _send_callback_openai(self, trace_id: str, response, duration_ms: int):
        """Send completion callback to gateway for OpenAI/xAI/OpenRouter responses (non-blocking)"""
        try:
            content = ""
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'content'):
                    content = message.content or ""

            tokens_used = 0
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'total_tokens'):
                    tokens_used = usage.total_tokens
                elif hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                    tokens_used = usage.prompt_tokens + usage.completion_tokens

            callback = {
                "trace_id": trace_id,
                "content": content,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used
            }

            self.http_client.post(
                f"{self.endpoint}/api/v1/llm/callback",
                json=callback,
                timeout=5.0
            )
        except Exception as e:
            print(f"Observ callback error: {e}")

    def _send_callback_gemini(self, trace_id: str, response, duration_ms: int):
        """Send completion callback to gateway for Gemini responses (non-blocking)"""
        try:
            content = ""
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    if len(candidate.content.parts) > 0:
                        content = candidate.content.parts[0].text

            tokens_used = 0
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                if hasattr(usage, 'total_token_count'):
                    tokens_used = usage.total_token_count
                elif hasattr(usage, 'prompt_token_count') and hasattr(usage, 'candidates_token_count'):
                    tokens_used = usage.prompt_token_count + usage.candidates_token_count

            callback = {
                "trace_id": trace_id,
                "content": content,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used
            }

            self.http_client.post(
                f"{self.endpoint}/api/v1/llm/callback",
                json=callback,
                timeout=5.0
            )
        except Exception as e:
            print(f"Observ callback error: {e}")

    def _send_callback_mistral(self, trace_id: str, response, duration_ms: int):
        """Send completion callback to gateway for Mistral responses (non-blocking)"""
        try:
            content = ""
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'content'):
                    content = message.content or ""

            tokens_used = 0
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'total_tokens'):
                    tokens_used = usage.total_tokens
                elif hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                    tokens_used = usage.prompt_tokens + usage.completion_tokens

            callback = {
                "trace_id": trace_id,
                "content": content,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used
            }

            self.http_client.post(
                f"{self.endpoint}/api/v1/llm/callback",
                json=callback,
                timeout=5.0
            )
        except Exception as e:
            print(f"Observ callback error: {e}")


__all__ = ["Observ"]
