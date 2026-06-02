"""OpenAI provider — also handles OpenAI-compatible endpoints (Bluesminds, custom)."""
from __future__ import annotations
import logging
from typing import Generator

from openai import OpenAI

log = logging.getLogger(__name__)


def stream_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """Stream chat completion chunks from an OpenAI-compatible API."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for ev in stream:
        if not ev.choices:
            continue
        delta = ev.choices[0].delta
        txt = getattr(delta, "content", None)
        if txt:
            yield txt


def fetch_models(api_key: str, base_url: str) -> list[dict]:
    """Fetch available models from an OpenAI-compatible API."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.models.list()
    return [{"id": m.id, "name": m.id} for m in resp.data]
