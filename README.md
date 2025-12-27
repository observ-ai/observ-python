# Observ Python SDK

AI tracing and semantic caching SDK for [Observ](https://observ.dev).

## Installation

```bash
pip install observ-sdk
```

Install provider-specific SDKs as needed:

```bash
# For Anthropic
pip install observ-sdk[anthropic]

# For OpenAI (also used by xAI and OpenRouter)
pip install observ-sdk[openai]

# For Google Gemini
pip install observ-sdk[gemini]

# For Mistral
pip install observ-sdk[mistral]

# Install all providers
pip install observ-sdk[all]
```

## Quick Start

### Anthropic

```python
import anthropic
from observ import Observ

ob = Observ(
    api_key="your-observ-api-key",
    recall=True,  # Enable semantic caching
)

client = anthropic.Anthropic(api_key="your-anthropic-key")
client = ob.anthropic(client)

# Use normally - all calls are automatically traced
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

### OpenAI

```python
import openai
from observ import Observ

ob = Observ(
    api_key="your-observ-api-key",
    recall=True,
)

client = openai.OpenAI(api_key="your-openai-key")
client = ob.openai(client)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Google Gemini

```python
import google.generativeai as genai
from observ import Observ

ob = Observ(
    api_key="your-observ-api-key",
    recall=True,
)

genai.configure(api_key="your-gemini-key")
model = genai.GenerativeModel("gemini-pro")
model = ob.gemini(model)

response = model.generate_content("Hello!")
print(response.text)
```

### Mistral

```python
from mistralai import Mistral
from observ import Observ

ob = Observ(
    api_key="your-observ-api-key",
    recall=True,
)

client = Mistral(api_key="your-mistral-key")
client = ob.mistral(client)

response = client.chat.completions.create(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### xAI (Grok)

```python
import openai
from observ import Observ

ob = Observ(
    api_key="your-observ-api-key",
    recall=True,
)

client = openai.OpenAI(
    api_key="your-xai-key",
    base_url="https://api.x.ai/v1"
)
client = ob.xai(client)

response = client.chat.completions.create(
    model="grok-beta",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### OpenRouter

```python
import openai
from observ import Observ

ob = Observ(
    api_key="your-observ-api-key",
    recall=True,
)

client = openai.OpenAI(
    api_key="your-openrouter-key",
    base_url="https://openrouter.ai/api/v1"
)
client = ob.openrouter(client)

response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Configuration

```python
ob = Observ(
    api_key="your-observ-api-key",  # Required
    recall=True,                     # Enable semantic caching (default: False)
    environment="production",        # Environment tag (default: "production")
    endpoint="https://api.observ.dev",  # Custom endpoint (optional)
    debug=False,                     # Enable debug logging (default: False)
)
```

## Features

- **Automatic Tracing**: All LLM calls are automatically traced
- **Semantic Caching**: Cache similar prompts to reduce costs and latency
- **Multi-Provider**: Support for Anthropic, OpenAI, Gemini, Mistral, xAI, and OpenRouter
- **Session Tracking**: Group related calls with session IDs
- **Metadata**: Attach custom metadata to traces

## Metadata and Sessions

All wrapped clients support metadata and session ID chaining:

```python
# Add metadata to a request
response = client.messages.with_metadata({"user_id": "123", "feature": "chat"}).create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Track conversation sessions
response = client.messages.with_session_id("conversation-abc").create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Combine both
response = client.messages.with_metadata({"user_id": "123"}).with_session_id("session-abc").create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## License

MIT
