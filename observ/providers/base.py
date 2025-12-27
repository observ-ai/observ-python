"""Base utilities for provider wrappers"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from observ import Observ


def convert_messages_to_gateway_format(
    messages: List[Any],
) -> List[Dict[str, str]]:
    """Convert provider messages to gateway format."""
    gateway_messages: List[Dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, dict):
            gateway_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        else:
            gateway_messages.append({
                "role": getattr(msg, "role", "user"),
                "content": getattr(msg, "content", ""),
            })
    return gateway_messages


def build_completion_request(
    provider: str,
    model: str,
    gateway_messages: List[Dict[str, str]],
    observ_instance: Observ,
    metadata: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build completion request for gateway."""
    request: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "messages": gateway_messages,
        "features": {
            "trace": True,
            "recall": observ_instance.recall,
            "resilience": False,
            "adapt": False,
        },
        "environment": observ_instance.environment,
        "metadata": metadata,
    }
    if session_id:
        request["external_session_id"] = session_id
    return request
