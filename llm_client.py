"""Shared LLM client supporting a local endpoint and OpenRouter."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any

import requests


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 10
MAX_RETRY_DELAY_SECONDS = 60.0
_cancel_event = Event()


class LLMUnauthorizedError(RuntimeError):
    """Fatal authentication error that must stop every LLM workflow."""


class LLMCancelledError(RuntimeError):
    """An LLM call cancelled because another call was unauthorized."""


def _raise_if_cancelled() -> None:
    if _cancel_event.is_set():
        raise LLMCancelledError("LLM execution cancelled after an HTTP 401 response")


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

    for attempt in range(MAX_RETRIES + 1):
        _raise_if_cancelled()
        response: requests.Response | None = None
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=settings.timeout_seconds
            )
            if response.status_code == 401:
                _cancel_event.set()
                raise LLMUnauthorizedError(
                    f"LLM authentication failed with HTTP 401 for {url}; cancelling all LLM work"
                )
            _raise_if_cancelled()
            response.raise_for_status()
            document = response.json()
            try:
                message = document["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError(f"Unexpected LLM response: {document}") from error
            if not isinstance(message, dict):
                raise ValueError(f"Unexpected assistant message: {message}")
            return message
        except (LLMUnauthorizedError, LLMCancelledError):
            raise
        except (requests.RequestException, ValueError) as error:
            if attempt >= MAX_RETRIES:
                raise
            retry_number = attempt + 1
            retry_after = response.headers.get("Retry-After") if response is not None else None
            try:
                delay = float(retry_after) if retry_after is not None else 2 ** (retry_number - 1)
            except (TypeError, ValueError):
                delay = 2 ** (retry_number - 1)
            delay = max(0.0, min(delay, MAX_RETRY_DELAY_SECONDS))
            print(
                f"LLM retry {retry_number}/{MAX_RETRIES} after error: {error}; "
                f"waiting {delay:g}s",
                flush=True,
            )
            if _cancel_event.wait(delay):
                _raise_if_cancelled()

    raise RuntimeError("Unreachable LLM retry state")


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
