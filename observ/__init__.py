"""Observ Python SDK - AI tracing and semantic caching"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from .providers import (
    AnthropicMessagesWrapper,
    GeminiGenerateContentWrapper,
    MistralChatCompletionsWrapper,
    OpenAIChatCompletionsWrapper,
    OpenRouterChatCompletionsWrapper,
    XAIChatCompletionsWrapper,
)

if TYPE_CHECKING:
    import anthropic
    import mistralai
    import openai


class Observ:
    """Main Observ SDK class for tracing AI provider calls."""

    def __init__(
        self,
        api_key: str,
        project_id: str = "default",
        recall: bool = False,
        environment: str = "production",
        endpoint: str = "https://api.observ.dev",
        debug: bool = False,
    ) -> None:
        self.api_key = api_key
        self.project_id = project_id
        self.recall = recall
        self.environment = environment
        self.endpoint = endpoint
        self.debug = debug
        self.http_client = httpx.Client(timeout=30.0)

    def log(self, message: str) -> None:
        """Log a message if debug mode is enabled."""
        if self.debug:
            print(f"[Observ] {message}")

    def anthropic(self, client: anthropic.Anthropic) -> anthropic.Anthropic:
        """Wrap an Anthropic client to route through Observ gateway."""
        client.messages = AnthropicMessagesWrapper(client.messages, self)  # type: ignore[assignment]
        return client

    def openai(self, client: openai.OpenAI) -> openai.OpenAI:
        """Wrap an OpenAI client to route through Observ gateway."""
        client.chat.completions = OpenAIChatCompletionsWrapper(  # type: ignore[assignment]
            client.chat.completions, self
        )
        return client

    def gemini(self, model: Any) -> Any:
        """Wrap a Gemini GenerativeModel to route through Observ gateway."""
        wrapper = GeminiGenerateContentWrapper(model, self)
        model.generate_content = wrapper.generate_content
        model.with_metadata = wrapper.with_metadata
        model.with_session_id = wrapper.with_session_id
        return model

    def xai(self, client: openai.OpenAI) -> openai.OpenAI:
        """Wrap an xAI client (using OpenAI SDK) to route through Observ gateway."""
        client.chat.completions = XAIChatCompletionsWrapper(  # type: ignore[assignment]
            client.chat.completions, self
        )
        return client

    def mistral(self, client: mistralai.Mistral) -> mistralai.Mistral:
        """Wrap a Mistral client to route through Observ gateway."""
        client.chat.completions = MistralChatCompletionsWrapper(  # type: ignore[assignment]
            client.chat.completions, self
        )
        return client

    def openrouter(self, client: openai.OpenAI) -> openai.OpenAI:
        """Wrap an OpenRouter client (using OpenAI SDK) to route through Observ gateway."""
        client.chat.completions = OpenRouterChatCompletionsWrapper(  # type: ignore[assignment]
            client.chat.completions, self
        )
        return client

    def _send_callback(self, trace_id: str, response: Any, duration_ms: int) -> None:
        """Send completion callback to gateway for Anthropic responses."""
        try:
            content = ""
            if hasattr(response, "content") and len(response.content) > 0:
                content = response.content[0].text

            tokens_used = 0
            if hasattr(response, "usage"):
                tokens_used = getattr(response.usage, "input_tokens", 0) + getattr(
                    response.usage, "output_tokens", 0
                )

            callback = {
                "trace_id": trace_id,
                "content": content,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
            }

            self.http_client.post(
                f"{self.endpoint}/v1/llm/callback",
                json=callback,
                timeout=5.0,
            )
        except Exception as e:
            self.log(f"Callback error: {e}")

    def _send_callback_openai(self, trace_id: str, response: Any, duration_ms: int) -> None:
        """Send completion callback to gateway for OpenAI/xAI/OpenRouter responses."""
        try:
            content = ""
            if hasattr(response, "choices") and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, "content"):
                    content = message.content or ""

            tokens_used = 0
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "total_tokens"):
                    tokens_used = usage.total_tokens
                elif hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
                    tokens_used = usage.prompt_tokens + usage.completion_tokens

            callback = {
                "trace_id": trace_id,
                "content": content,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
            }

            self.http_client.post(
                f"{self.endpoint}/v1/llm/callback",
                json=callback,
                timeout=5.0,
            )
        except Exception as e:
            self.log(f"Callback error: {e}")

    def _send_callback_gemini(self, trace_id: str, response: Any, duration_ms: int) -> None:
        """Send completion callback to gateway for Gemini responses."""
        try:
            content = ""
            if hasattr(response, "text"):
                content = response.text
            elif hasattr(response, "candidates") and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    if len(candidate.content.parts) > 0:
                        content = candidate.content.parts[0].text

            tokens_used = 0
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if hasattr(usage, "total_token_count"):
                    tokens_used = usage.total_token_count
                elif hasattr(usage, "prompt_token_count") and hasattr(
                    usage, "candidates_token_count"
                ):
                    tokens_used = usage.prompt_token_count + usage.candidates_token_count

            callback = {
                "trace_id": trace_id,
                "content": content,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
            }

            self.http_client.post(
                f"{self.endpoint}/v1/llm/callback",
                json=callback,
                timeout=5.0,
            )
        except Exception as e:
            self.log(f"Callback error: {e}")

    def _send_callback_mistral(self, trace_id: str, response: Any, duration_ms: int) -> None:
        """Send completion callback to gateway for Mistral responses."""
        try:
            content = ""
            if hasattr(response, "choices") and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, "content"):
                    content = message.content or ""

            tokens_used = 0
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "total_tokens"):
                    tokens_used = usage.total_tokens
                elif hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
                    tokens_used = usage.prompt_tokens + usage.completion_tokens

            callback = {
                "trace_id": trace_id,
                "content": content,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
            }

            self.http_client.post(
                f"{self.endpoint}/v1/llm/callback",
                json=callback,
                timeout=5.0,
            )
        except Exception as e:
            self.log(f"Callback error: {e}")


__all__ = ["Observ"]
