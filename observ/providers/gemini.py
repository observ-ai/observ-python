"""Gemini provider wrapper"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import build_completion_request

if TYPE_CHECKING:
    from observ import Observ


class GeminiGenerateContentWrapper:
    """Wrapper for Gemini generate_content that supports .with_metadata() and .with_session_id() chaining."""

    def __init__(self, original_model: Any, observ_instance: Observ) -> None:
        self._original_model = original_model
        self._wt = observ_instance
        self._metadata: Dict[str, Any] = {}
        self._session_id: Optional[str] = None

    def with_metadata(self, metadata: Dict[str, Any]) -> GeminiGenerateContentWrapper:
        """Set metadata for the next API call."""
        self._metadata = metadata
        return self

    def with_session_id(self, session_id: str) -> GeminiGenerateContentWrapper:
        """Set session ID for the next API call."""
        self._session_id = session_id
        return self

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        """Generate content method that routes through Observ gateway."""
        metadata = self._metadata
        session_id = self._session_id
        self._metadata = {}
        self._session_id = None

        prompt = kwargs.get("prompt") or kwargs.get("contents") or (args[0] if args else "")
        model = getattr(self._original_model, "_model_name", "gemini-pro")

        gateway_messages: List[Dict[str, str]]
        if isinstance(prompt, str):
            gateway_messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            gateway_messages = []
            for item in prompt:
                if isinstance(item, dict):
                    parts = item.get("parts", [""])
                    content = parts[0] if parts else ""
                    gateway_messages.append({
                        "role": item.get("role", "user"),
                        "content": item.get("content", str(content)),
                    })
                elif hasattr(item, "parts"):
                    gateway_messages.append({
                        "role": "user",
                        "content": item.parts[0].text if item.parts else "",
                    })
                else:
                    gateway_messages.append({"role": "user", "content": str(item)})
        else:
            gateway_messages = [{"role": "user", "content": str(prompt)}]

        completion_request = build_completion_request(
            "gemini", model, gateway_messages, self._wt, metadata, session_id
        )

        try:
            response = self._wt.http_client.post(
                f"{self._wt.endpoint}/v1/llm/complete",
                json=completion_request,
                headers={
                    "Authorization": f"Bearer {self._wt.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            gateway_response = response.json()

            if gateway_response.get("action") == "cache_hit":
                cached_content = gateway_response.get("content", "")
                return type(
                    "GenerateContentResponse",
                    (),
                    {
                        "text": cached_content,
                        "candidates": [
                            type(
                                "Candidate",
                                (),
                                {
                                    "content": type(
                                        "Content",
                                        (),
                                        {
                                            "parts": [
                                                type("Part", (), {"text": cached_content})()
                                            ],
                                            "role": "model",
                                        },
                                    )()
                                },
                            )()
                        ],
                        "usage_metadata": type(
                            "UsageMetadata",
                            (),
                            {
                                "prompt_token_count": 0,
                                "candidates_token_count": 0,
                                "total_token_count": 0,
                            },
                        )(),
                    },
                )()

            trace_id = gateway_response.get("trace_id")
            start_time = time.time()

            actual_response = self._original_model.generate_content(*args, **kwargs)

            duration_ms = int((time.time() - start_time) * 1000)
            self._wt._send_callback_gemini(trace_id, actual_response, duration_ms)

            return actual_response

        except Exception as e:
            self._wt.log(f"Gateway error: {e}")
            return self._original_model.generate_content(*args, **kwargs)
