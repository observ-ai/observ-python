# Observ Python SDK

AI tracing SDK for Observ supporting multiple AI providers.

## Installation

```bash
pip install observ
```

Install provider-specific SDKs as needed:

```bash
# For Anthropic
pip install anthropic

# For OpenAI
pip install openai

# For Google Gemini
pip install google-generativeai

# For Mistral
pip install mistralai

# For xAI and OpenRouter (use OpenAI SDK)
pip install openai
```

## Usage

### Anthropic

```python
import anthropic
from observ import Observ

# Initialize Observ
ob = Observ(api_key="your-observ-api-key", recall=True)

# Initialize Anthropic client
client = anthropic.Anthropic(api_key="your-anthropic-key")

# Wrap the client to enable tracing
client = ob.anthropic(client)

# Use normally - all calls are automatically traced
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### OpenAI

```python
import openai
from observ import Observ

# Initialize Observ
ob = Observ(api_key="your-observ-api-key", recall=True)

# Initialize OpenAI client
client = openai.OpenAI(api_key="your-openai-key")

# Wrap the client to enable tracing
client = ob.openai(client)

# Use normally - all calls are automatically traced
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

# Initialize Observ
ob = Observ(api_key="your-observ-api-key", recall=True)

# Configure Gemini API key
genai.configure(api_key="your-gemini-key")

# Initialize Gemini model
model = genai.GenerativeModel("gemini-pro")

# Wrap the model to enable tracing
model = ob.gemini(model)

# Use normally - all calls are automatically traced
response = model.generate_content("Hello!")
print(response.text)
```

### Mistral

```python
from mistralai import Mistral
from observ import Observ

# Initialize Observ
ob = Observ(api_key="your-observ-api-key", recall=True)

# Initialize Mistral client
client = Mistral(api_key="your-mistral-key")

# Wrap the client to enable tracing
client = ob.mistral(client)

# Use normally - all calls are automatically traced
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

# Initialize Observ
ob = Observ(api_key="your-observ-api-key", recall=True)

# Initialize xAI client using OpenAI SDK with xAI endpoint
client = openai.OpenAI(
    api_key="your-xai-key",
    base_url="https://api.x.ai/v1"
)

# Wrap the client to enable tracing
client = ob.xai(client)

# Use normally - all calls are automatically traced
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

# Initialize Observ
ob = Observ(api_key="your-observ-api-key", recall=True)

# Initialize OpenRouter client using OpenAI SDK
client = openai.OpenAI(
    api_key="your-openrouter-key",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://your-app-url.com",  # Optional: for OpenRouter analytics
        "X-Title": "Your App Name"  # Optional: for OpenRouter analytics
    }
)

# Wrap the client to enable tracing
client = ob.openrouter(client)

# Use normally - all calls are automatically traced
response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Features

- Automatic trace creation before API calls
- Duration tracking in milliseconds
- Error handling and logging
- Non-blocking async trace updates
- Support for semantic caching (recall) when enabled
- Metadata support via `.with_metadata()` chaining

## Metadata Support

All wrapped clients support metadata chaining:

```python
# Anthropic example
response = client.messages.with_metadata({"user_id": "123"}).create(
    model="claude-3-haiku-20240307",
    messages=[{"role": "user", "content": "Hello!"}]
)

# OpenAI example
response = client.chat.completions.with_metadata({"user_id": "123"}).create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```
