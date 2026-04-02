from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter

async def create_adapter(provider: str):
    if provider == 'openai':
        return OpenAIAdapter()
    if provider == 'anthropic':
        return AnthropicAdapter()
    raise ValueError(f"Unknown provider: {provider}")
