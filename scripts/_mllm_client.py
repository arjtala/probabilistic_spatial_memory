"""Thin OpenAI-compat MLLM client for Gemini 3.1 Pro + Claude 4.6 Opus.

Both models are served behind the same internal OpenAI-compatible proxy
at api.llama.com — only the `model` field and the env-var name for the
API key differ. This keeps the client model-agnostic: pass `Mllm.GEMINI`
or `Mllm.CLAUDE` and the rest of the request shape is identical.

The proxy quirk we care about: Gemini is a reasoning model whose response
may lack a `message` field if all `max_tokens` are spent on hidden thinking
tokens. We surface that as a clear ValueError rather than KeyError so the
caller can bump max_tokens and retry; Claude doesn't have this failure
mode but the same response-shape check is harmless for it.

This module is deliberately small — request building, retry, parse — so
the eval harness can own the per-question loop, partial-result caching,
and any prompt-templating logic.
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import requests
from PIL import Image


# Internal OpenAI-compat proxy serving both Gemini and Claude.
DEFAULT_API_BASE = "https://api.llama.com/experimental/compat/openai/v1"

# Default response budget. Gemini's thinking tokens come out of this, so
# it has to be generous; Claude rarely needs more than 64 for short-form
# answers but pays no cost for unused budget.
DEFAULT_MAX_TOKENS = 1024

# Default per-call timeout — payloads with 5-10 base64 frames take 20-60s.
DEFAULT_TIMEOUT_S = 120


class Mllm(Enum):
    """Supported proprietary MLLMs.

    Values are (model_id, env_var). Model ids match what the
    api.llama.com proxy expects — keep in sync with the EPIC probe
    scripts in experimental_video_retrieval/.
    """
    GEMINI = ("gemini-3-1-pro-preview-genai", "GEMINI_API_KEY")
    CLAUDE = ("claude-4-6-opus-genai", "CLAUDE_API_KEY")

    @property
    def model_id(self) -> str:
        return self.value[0]

    @property
    def env_var(self) -> str:
        return self.value[1]


@dataclass
class MllmCall:
    """A single MLLM call: frames + text prompt, returned response.

    Stored together for cache-key construction and replay debugging.
    """
    frames_b64: list[str]
    prompt: str
    model: str
    response: str


class MllmError(RuntimeError):
    """Raised when the MLLM returns no usable content after retries."""


def encode_frame(frame: Image.Image | Path, *, max_size: int = 768) -> str:
    """Encode a PIL image or JPEG path to a base64 JPEG string.

    Resizes the long edge to `max_size` (default 768) before encoding.
    CLIP-L already downsamples to 224 and the MLLM's vision encoder
    won't benefit from much more — bigger payloads just slow the API
    call and cost more tokens.

    Accepts both PIL.Image and pathlib.Path so callers can pass JPEG
    frame paths directly from PSM's --search output.
    """
    if isinstance(frame, (str, Path)):
        img = Image.open(frame).convert("RGB")
    else:
        img = frame
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_mllm(
    *,
    model: Mllm,
    frames_b64: list[str],
    prompt: str,
    api_base: str = DEFAULT_API_BASE,
    api_key: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_retries: int = 3,
) -> str:
    """One MLLM call. Returns the response text; raises MllmError on failure.

    `api_key` defaults to `os.environ[model.env_var]`. Pass an explicit
    string to override (with or without a `Bearer ` prefix; this function
    adds the prefix when missing).

    Retries with exponential backoff on transport / 5xx errors. A reasoning
    model returning an empty content slot (thinking-token exhaustion) is
    NOT retried — bumping max_tokens is the caller's call, not ours.
    """
    if api_key is None:
        api_key = os.environ.get(model.env_var)
    if not api_key:
        raise MllmError(
            f"no API key for {model.name}: set ${model.env_var} or pass api_key=..."
        )
    if not api_key.startswith("Bearer "):
        api_key = f"Bearer {api_key}"

    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {"Content-Type": "application/json", "Authorization": api_key}
    url = f"{api_base}/chat/completions"

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message")
            if message is None or "content" not in message or not message["content"]:
                # Reasoning-model thinking-token exhaustion. Surface
                # finish_reason so the caller knows whether to bump tokens
                # vs the model legitimately producing no answer.
                raise MllmError(
                    f"empty response from {model.name} "
                    f"(finish_reason={choice.get('finish_reason')}; "
                    "consider raising max_tokens)"
                )
            return message["content"].strip()
        except MllmError:
            # Don't retry empty-content errors — same input will yield same output.
            raise
        except Exception as exc:  # noqa: BLE001 — proxy can throw a variety of errors.
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
    raise MllmError(f"all {max_retries} retries failed: {last_exc}") from last_exc


def smoke_test(model: Mllm, *, api_base: str = DEFAULT_API_BASE) -> str:
    """One-line health check — no images, just `Say OK.`

    Used by the eval harness on startup so a misconfigured key / proxy
    fails fast at second 0 rather than per-question.
    """
    return call_mllm(
        model=model,
        frames_b64=[],
        prompt="Say OK.",
        api_base=api_base,
        max_tokens=16,
    )
