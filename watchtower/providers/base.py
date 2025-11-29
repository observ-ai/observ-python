"""Base utilities for provider wrappers"""

from typing import Any


def convert_messages_to_gateway_format(messages):
    """Convert provider messages to gateway format"""
    gateway_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            gateway_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        else:
            gateway_messages.append({
                "role": getattr(msg, "role", "user"),
                "content": getattr(msg, "content", "")
            })
    return gateway_messages


def build_completion_request(provider: str, model: str, gateway_messages: list, observ_instance, metadata: dict[str, Any], session_id: str | None = None):
    """Build completion request for gateway"""
    request = {
        "provider": provider,
        "model": model,
        "messages": gateway_messages,
        "features": {
            "trace": True,
            "recall": observ_instance.recall,
            "resilience": False,
            "adapt": False
        },
        "environment": observ_instance.environment,
        "metadata": metadata
    }
    if session_id:
        request["external_session_id"] = session_id
    return request

