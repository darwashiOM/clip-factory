"""
llm_client.py — Shared LLM provider abstraction for clip-factory.

Supported text providers (TEXT_MODEL_PROVIDER):
  deepseek  (default)  — DeepSeek Chat via OpenAI-compatible API
  openai               — OpenAI Chat API
  gemini    (legacy)   — Google Gemini via google-genai SDK

Supported vision providers (VISION_MODEL_PROVIDER):
  disabled  (default)  — no vision model; vision callers skip or use technical checks only
  openai               — OpenAI GPT-4o vision
  gemini               — Google Gemini vision (legacy)

Environment variables
─────────────────────
  TEXT_MODEL_PROVIDER   deepseek | openai | gemini        default: deepseek
  TEXT_MODEL_NAME       model identifier                  default: per-provider
                          deepseek → deepseek-chat
                          openai   → gpt-4o-mini
                          gemini   → gemini-2.5-flash
  VISION_MODEL_PROVIDER disabled | openai | gemini        default: disabled
  VISION_MODEL_NAME     model identifier                  default: per-provider
                          openai   → gpt-4o-mini
                          gemini   → gemini-2.5-pro

  DEEPSEEK_API_KEY      required when TEXT_MODEL_PROVIDER=deepseek
  OPENAI_API_KEY        required when TEXT_MODEL_PROVIDER=openai
                        required when VISION_MODEL_PROVIDER=openai
  GEMINI_API_KEY        required when TEXT_MODEL_PROVIDER=gemini
                        required when VISION_MODEL_PROVIDER=gemini

Feature switches (per-server — see individual servers for defaults)
────────────────────────────────────────────────────────────────────
  CLIP_FINDER_USE_LLM         true | false    default: true
  TRANSCRIPT_REFINER_USE_LLM  true | false    default: true
  SCENE_DIRECTOR_USE_LLM      true | false    default: false
  ASSET_GUARD_USE_VISION      true | false    default: false
"""

from __future__ import annotations

import json
import os
from typing import Optional


# ─── Defaults ────────────────────────────────────────────────────────────────

_PROVIDER_DEFAULTS: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}

_VISION_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-pro",
}

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


# ─── Provider config helpers ─────────────────────────────────────────────────

def _text_provider() -> str:
    return os.environ.get("TEXT_MODEL_PROVIDER", "deepseek").strip().lower()


def _text_model_name() -> str:
    provider = _text_provider()
    default = _PROVIDER_DEFAULTS.get(provider, "deepseek-chat")
    return (os.environ.get("TEXT_MODEL_NAME") or "").strip() or default


def _vision_provider() -> str:
    return os.environ.get("VISION_MODEL_PROVIDER", "disabled").strip().lower()


def _vision_model_name() -> str:
    provider = _vision_provider()
    default = _VISION_PROVIDER_DEFAULTS.get(provider, "")
    return (os.environ.get("VISION_MODEL_NAME") or "").strip() or default


def vision_enabled() -> bool:
    return _vision_provider() not in ("disabled", "none", "off", "false", "")


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


# ─── TextLLM class ───────────────────────────────────────────────────────────

class TextLLM:
    """
    Thin wrapper that normalises text generation across providers.

    Usage:
        llm = get_text_llm()
        json_str = llm.generate_json(prompt)
        data = json.loads(json_str)
    """

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_json(self, prompt: str, system: str = "") -> str:
        """
        Call the configured provider and return a raw JSON string.
        The caller is responsible for parsing and validating the result.
        Raises RuntimeError on missing API keys or provider failures.
        """
        if self.provider == "gemini":
            return self._gemini_json(prompt)
        return self._openai_compat_json(prompt, system)

    # ── Provider implementations ──────────────────────────────────────────────

    def _openai_compat_json(self, prompt: str, system: str = "") -> str:
        """Call DeepSeek or OpenAI (both use the openai SDK)."""
        from openai import OpenAI

        if self.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY is not set. "
                    "Add it to .env or set TEXT_MODEL_PROVIDER=openai to use OpenAI instead."
                )
            client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in .env")
            client = OpenAI(api_key=api_key)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return (resp.choices[0].message.content or "{}").strip()

    def _gemini_json(self, prompt: str) -> str:
        """Call Google Gemini (legacy path)."""
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai package is required for TEXT_MODEL_PROVIDER=gemini. "
                "Install it or switch to TEXT_MODEL_PROVIDER=deepseek."
            ) from e

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return (getattr(resp, "text", None) or "{}").strip()

    def __repr__(self) -> str:
        return f"TextLLM(provider={self.provider!r}, model={self.model!r})"


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_text_llm() -> TextLLM:
    """
    Return a TextLLM configured from environment variables.

    TEXT_MODEL_PROVIDER selects the backend:
      deepseek (default) → DEEPSEEK_API_KEY
      openai             → OPENAI_API_KEY
      gemini             → GEMINI_API_KEY

    TEXT_MODEL_NAME overrides the model name. When unset, sensible
    per-provider defaults are used (see module docstring).
    """
    provider = _text_provider()
    model = _text_model_name()
    valid = {"deepseek", "openai", "gemini"}
    if provider not in valid:
        raise RuntimeError(
            f"Unknown TEXT_MODEL_PROVIDER={provider!r}. Valid: {sorted(valid)}"
        )
    return TextLLM(provider=provider, model=model)


# ─── Feature-switch helpers (used by individual servers) ─────────────────────

def clip_finder_use_llm() -> bool:
    return _env_bool("CLIP_FINDER_USE_LLM", default=True)


def transcript_refiner_use_llm() -> bool:
    return _env_bool("TRANSCRIPT_REFINER_USE_LLM", default=True)


def scene_director_use_llm() -> bool:
    return _env_bool("SCENE_DIRECTOR_USE_LLM", default=False)


def asset_guard_use_vision() -> bool:
    return _env_bool("ASSET_GUARD_USE_VISION", default=False)


# ─── Config summary (for healthcheck tools) ──────────────────────────────────

def provider_summary() -> dict:
    """Return a dict describing the current provider configuration."""
    tp = _text_provider()
    vp = _vision_provider()
    return {
        "text_provider": tp,
        "text_model": _text_model_name(),
        "vision_provider": vp,
        "vision_model": _vision_model_name() if vision_enabled() else "disabled",
        "text_key_present": _has_key(tp),
        "vision_key_present": _has_key(vp) if vision_enabled() else False,
        "feature_switches": {
            "CLIP_FINDER_USE_LLM": clip_finder_use_llm(),
            "TRANSCRIPT_REFINER_USE_LLM": transcript_refiner_use_llm(),
            "SCENE_DIRECTOR_USE_LLM": scene_director_use_llm(),
            "ASSET_GUARD_USE_VISION": asset_guard_use_vision(),
        },
    }


def _has_key(provider: str) -> bool:
    key_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    key_name = key_map.get(provider)
    return bool(key_name and os.environ.get(key_name, "").strip())
