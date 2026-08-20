"""Shared LLM client supporting a local endpoint and OpenRouter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    local_endpoint: str
    local_model: str
    local_authorization: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it:free"
    timeout_seconds: int = 300


def normalize_provider(value: str) -> str:
    provider = str(value or "local").strip().lower()
    if provider == "openroute":
        provider = "openrouter"
    if provider not in {"local", "openrouter"}:
        raise ValueError("LLM_PROVIDER must be 'local' or 'openrouter'")
    return provider


def chat_completion(
    messages: list[dict[str, Any]],
    settings: LLMSettings,
    *,
    reasoning: bool | None = None,
) -> dict[str, Any]:
    """Return the assistant message, including reasoning_details when present."""
    provider = normalize_provider(settings.provider)
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        url = OPENROUTER_ENDPOINT
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": settings.openrouter_model,
            "messages": messages,
            "reasoning": {"enabled": True if reasoning is None else reasoning},
        }
    else:
        url = settings.local_endpoint
        headers = {"Content-Type": "application/json"}
        if settings.local_authorization:
            headers["Authorization"] = settings.local_authorization
        payload = {
            "model": settings.local_model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 30,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    response = requests.post(url, headers=headers, json=payload, timeout=settings.timeout_seconds)
    response.raise_for_status()
    document = response.json()
    try:
        message = document["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Unexpected LLM response: {document}") from error
    if not isinstance(message, dict):
        raise ValueError(f"Unexpected assistant message: {message}")
    return message


def complete_prompt(prompt: str, settings: LLMSettings) -> str:
    message = chat_completion([{"role": "user", "content": prompt}], settings)
    return str(message.get("content") or "")


def continue_with_reasoning(
    first_prompt: str,
    follow_up_prompt: str,
    settings: LLMSettings,
) -> dict[str, Any]:
    """Make two OpenRouter calls while preserving reasoning_details unmodified."""
    first_message = chat_completion(
        [{"role": "user", "content": first_prompt}], settings, reasoning=True
    )
    messages = [
        {"role": "user", "content": first_prompt},
        {
            "role": "assistant",
            "content": first_message.get("content"),
            "reasoning_details": first_message.get("reasoning_details"),
        },
        {"role": "user", "content": follow_up_prompt},
    ]
    return chat_completion(messages, settings, reasoning=True)
