"""Provider router — dispatches to the correct provider adapter."""
from __future__ import annotations
import logging
from typing import Generator

from .openai_provider import stream_chat as openai_stream, fetch_models as openai_models
from .anthropic_provider import stream_chat as anthropic_stream, fetch_models as anthropic_models
from .google_provider import stream_chat as google_stream, fetch_models as google_models

log = logging.getLogger(__name__)

# Bluesminds uses OpenAI-compatible API
BLUESMINDS_BASE_URL = "https://api.bluesminds.com/v1"


def stream_chat(
    provider: str,
    api_key: str,
    model: str,
    messages: list[dict],
    base_url: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """Route to the correct provider and stream response."""
    if provider == "anthropic":
        yield from anthropic_stream(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "google":
        yield from google_stream(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "bluesminds":
        yield from openai_stream(
            api_key=api_key,
            base_url=BLUESMINDS_BASE_URL,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        yield from openai_stream(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "custom":
        if not base_url:
            raise ValueError("base_url is required for custom provider")
        yield from openai_stream(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def fetch_models(
    provider: str,
    api_key: str,
    base_url: str | None = None,
) -> list[dict]:
    """Fetch available models for a provider."""
    if provider == "anthropic":
        return anthropic_models(api_key=api_key)
    elif provider == "google":
        return google_models(api_key=api_key)
    elif provider == "bluesminds":
        return openai_models(api_key=api_key, base_url=BLUESMINDS_BASE_URL)
    elif provider == "openai":
        return openai_models(api_key=api_key, base_url=base_url or "https://api.openai.com/v1")
    elif provider == "custom":
        if not base_url:
            raise ValueError("base_url is required for custom provider")
        return openai_models(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}")
