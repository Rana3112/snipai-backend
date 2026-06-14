"""Tier 2 free-model whitelist — per-provider curated lists.

This is the fallback for providers that don't expose pricing in their /models
response (OpenAI, Anthropic, Google, Groq, NVIDIA, Bluesminds, OpenCode Zen).

For OpenRouter, pricing comes live from https://openrouter.ai/api/v1/models
(see backend/providers/router.py), so this table intentionally omits it.

When a provider rotates its free lineup, edit this file. It is the single
source of truth for Tier 2 detection.
"""
from __future__ import annotations

# Provider -> list of {id, vision} entries that are currently free.
PROVIDER_FREE_MODELS: dict[str, list[dict]] = {
    "google": [
        {"id": "gemini-2.0-flash", "vision": True},
        {"id": "gemini-2.0-flash-lite", "vision": True},
    ],
    "groq": [
        {"id": "llama-3.2-90b-vision-preview", "vision": True},
    ],
    "nvidia": [
        {"id": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1", "vision": True},
    ],
    "bluesminds": [
        {"id": "meta/llama-3.2-11b-vision-instruct", "vision": True},
    ],
    "openai": [],
    "anthropic": [],
    "opencode_zen": [],
    "openrouter": [],
    "custom": [],
}


def is_free(provider: str, model_id: str) -> bool | None:
    """Return True/False if the model is whitelisted as free for the provider.
    Returns None if no entry exists (caller decides default).
    """
    for entry in PROVIDER_FREE_MODELS.get(provider, []):
        if entry["id"] == model_id:
            return True
    return None


def vision_for(provider: str, model_id: str) -> bool | None:
    """Return vision flag from whitelist, or None if not in list."""
    for entry in PROVIDER_FREE_MODELS.get(provider, []):
        if entry["id"] == model_id:
            return entry.get("vision", False)
    return None


def any_free_vision(provider: str) -> str | None:
    """Return the first free+vision model id for a provider, or None."""
    for entry in PROVIDER_FREE_MODELS.get(provider, []):
        if entry.get("vision"):
            return entry["id"]
    return None
