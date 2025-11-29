"""Anthropic provider wrapper"""

from typing import Any
import time
from .base import convert_messages_to_gateway_format, build_completion_request


class AnthropicMessagesWrapper:
    """Wrapper for Anthropic messages that supports .with_metadata() and .with_session_id() chaining"""

    def __init__(self, original_messages, observ_instance):
        self._original_messages = original_messages
        self._wt = observ_instance
        self._metadata = {}
        self._session_id = None

    def with_metadata(self, metadata: dict[str, Any]):
        """Set metadata for the next API call"""
        self._metadata = metadata
        return self

    def with_session_id(self, session_id: str):
        """Set session ID for the next API call"""
        self._session_id = session_id
        return self

    def create(self, *args, **kwargs):
        """Create method that routes through Observ gateway"""
        metadata = self._metadata
        session_id = self._session_id
        self._metadata = {}
        self._session_id = None

        # Extract messages from kwargs or args
        messages = kwargs.get("messages", args[0] if args else [])
        model = kwargs.get("model", args[1] if len(args) > 1 else "claude-3-5-sonnet-20241022")

        # Convert messages to gateway format
        gateway_messages = convert_messages_to_gateway_format(messages)

        # Build completion request
        completion_request = build_completion_request(
            "anthropic", model, gateway_messages, self._wt, metadata, session_id
        )

        # Send to gateway
        try:
            response = self._wt.http_client.post(
                f"{self._wt.endpoint}/api/v1/llm/complete",
                json=completion_request,
                headers={
                    "Authorization": f"Bearer {self._wt.api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            gateway_response = response.json()

            # Handle cache hit
            if gateway_response.get("action") == "cache_hit":
                cached_content = gateway_response.get("content", "")
                return type('Message', (), {
                    'content': [type('TextBlock', (), {
                        'text': cached_content,
                        'type': 'text'
                    })()],
                    'role': 'assistant',
                    'model': model,
                    'stop_reason': 'end_turn',
                    'usage': type('Usage', (), {
                        'input_tokens': 0,
                        'output_tokens': 0
                    })()
                })()

            # Proceed with actual API call
            trace_id = gateway_response.get("trace_id")
            start_time = time.time()

            actual_response = self._original_messages.create(*args, **kwargs)

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Send completion callback (non-blocking)
            self._wt._send_callback(trace_id, actual_response, duration_ms)

            return actual_response

        except Exception as e:
            print(f"Observ gateway error: {e}")
            # Fallback to direct API call on error
            return self._original_messages.create(*args, **kwargs)

