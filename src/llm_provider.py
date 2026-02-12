"""
Round-robin LLM provider that rotates across Groq models AND
Vercel AI Gateway models to avoid rate limits and maximize uptime.

Supports:
  - Multiple Groq models (via groq SDK)
  - Vercel AI Gateway models (via openai SDK with base_url override)
  - Automatic failover: if one model/provider is rate-limited, skip to next
  - Cooldown tracking per model so rate-limited models recover
"""

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelEndpoint:
    """A single model endpoint (either Groq or Vercel AI Gateway)."""

    provider: str  # "groq" or "vercel"
    model_id: str  # e.g. "llama-3.3-70b-versatile"
    max_completion_tokens: int = 8000
    context_window: int = 131072
    cooldown_until: float = 0.0  # timestamp until which this model is on cooldown
    failure_count: int = 0
    success_count: int = 0


class LLMProvider:
    """
    Round-robin LLM provider that rotates across multiple models and providers.

    Usage:
        provider = LLMProvider(groq_api_key="...", gateway_api_key="...")
        response = provider.chat(system_prompt, user_prompt)
    """

    # Good Groq models (production + strong preview), ordered by quality
    GROQ_MODELS = [
        ModelEndpoint(
            provider="groq",
            model_id="meta-llama/llama-4-maverick-17b-128e-instruct",
            max_completion_tokens=8192,
        ),
        ModelEndpoint(
            provider="groq",
            model_id="llama-3.3-70b-versatile",
            max_completion_tokens=32768,
        ),
        ModelEndpoint(
            provider="groq",
            model_id="qwen/qwen3-32b",
            max_completion_tokens=40960,
        ),
        ModelEndpoint(
            provider="groq",
            model_id="meta-llama/llama-4-scout-17b-16e-instruct",
            max_completion_tokens=8192,
        ),
        ModelEndpoint(
            provider="groq",
            model_id="openai/gpt-oss-120b",
            max_completion_tokens=65536,
        ),
        ModelEndpoint(
            provider="groq",
            model_id="openai/gpt-oss-20b",
            max_completion_tokens=65536,
        ),
        ModelEndpoint(
            provider="groq",
            model_id="llama-3.1-8b-instant",
            max_completion_tokens=131072,
        ),
    ]

    # Vercel AI Gateway models (free $5/mo credit, diverse providers)
    VERCEL_MODELS = [
        ModelEndpoint(
            provider="vercel",
            model_id="meta-llama/llama-3.3-70b-versatile",
            max_completion_tokens=32768,
        ),
        ModelEndpoint(
            provider="vercel",
            model_id="google/gemini-2.5-flash",
            max_completion_tokens=65536,
            context_window=1000000,
        ),
        ModelEndpoint(
            provider="vercel",
            model_id="mistralai/mistral-small-latest",
            max_completion_tokens=32768,
        ),
    ]

    # Cooldown time in seconds after a rate limit (escalates with failures)
    BASE_COOLDOWN = 15
    MAX_COOLDOWN = 120

    def __init__(
        self,
        groq_api_key: str = "",
        gateway_api_key: str = "",
    ):
        self._groq_client = None
        self._vercel_client = None
        self._lock = threading.Lock()
        self._current_index = 0

        # Build endpoint list from available providers
        self._endpoints: list[ModelEndpoint] = []

        # --- Groq ---
        if groq_api_key:
            try:
                from groq import Groq

                self._groq_client = Groq(api_key=groq_api_key)
                self._endpoints.extend(self.GROQ_MODELS)
                logger.info(
                    "Groq provider enabled: %d models",
                    len(self.GROQ_MODELS),
                )
            except ImportError:
                logger.warning("groq package not installed — skipping Groq provider")

        # --- Vercel AI Gateway ---
        gw_key = gateway_api_key or os.getenv("VERCEL_OIDC_TOKEN", "")
        if gw_key:
            try:
                from openai import OpenAI

                self._vercel_client = OpenAI(
                    api_key=gw_key,
                    base_url="https://ai-gateway.vercel.sh/v1",
                )
                self._endpoints.extend(self.VERCEL_MODELS)
                logger.info(
                    "Vercel AI Gateway enabled: %d models",
                    len(self.VERCEL_MODELS),
                )
            except ImportError:
                logger.warning(
                    "openai package not installed — skipping Vercel AI Gateway"
                )

        if not self._endpoints:
            raise ValueError(
                "No LLM providers configured. Set GROQ_API_KEY and/or AI_GATEWAY_API_KEY."
            )

        logger.info(
            "LLMProvider ready: %d total endpoints across %s",
            len(self._endpoints),
            list({e.provider for e in self._endpoints}),
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 8000,
        max_retries: int = None,
    ) -> str:
        """
        Send a chat completion request, rotating through providers/models.

        On rate limit or failure, automatically moves to the next endpoint.
        Returns empty string only if ALL endpoints fail.
        """
        if max_retries is None:
            max_retries = len(self._endpoints)

        tried = set()

        for attempt in range(max_retries):
            endpoint = self._next_endpoint(tried)
            if endpoint is None:
                # All endpoints tried or on cooldown — wait for shortest cooldown
                wait = self._shortest_cooldown()
                if wait > 0 and attempt < max_retries - 1:
                    logger.info(
                        "All endpoints on cooldown. Waiting %.1fs...", wait
                    )
                    time.sleep(wait)
                    tried.clear()
                    continue
                break

            try:
                result = self._call_endpoint(
                    endpoint, system_prompt, user_prompt, temperature, max_tokens
                )
                endpoint.success_count += 1
                endpoint.failure_count = max(0, endpoint.failure_count - 1)
                return result

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "rate_limit" in error_str or "429" in error_str
                is_overloaded = "503" in error_str or "overloaded" in error_str

                endpoint.failure_count += 1
                tried.add(id(endpoint))

                if is_rate_limit or is_overloaded:
                    cooldown = min(
                        self.BASE_COOLDOWN * (2 ** (endpoint.failure_count - 1)),
                        self.MAX_COOLDOWN,
                    )
                    endpoint.cooldown_until = time.time() + cooldown
                    logger.warning(
                        "Rate limited on %s/%s — cooldown %.0fs (attempt %d/%d)",
                        endpoint.provider,
                        endpoint.model_id,
                        cooldown,
                        attempt + 1,
                        max_retries,
                    )
                else:
                    logger.warning(
                        "Error on %s/%s: %s (attempt %d/%d)",
                        endpoint.provider,
                        endpoint.model_id,
                        str(e)[:200],
                        attempt + 1,
                        max_retries,
                    )
                    # Short pause before trying next model
                    time.sleep(1)

        logger.error("All LLM endpoints exhausted after %d attempts", max_retries)
        return ""

    @property
    def current_model(self) -> str:
        """Return the model_id of the current endpoint (for logging)."""
        if self._endpoints:
            idx = self._current_index % len(self._endpoints)
            return self._endpoints[idx].model_id
        return "none"

    @property
    def endpoint_count(self) -> int:
        return len(self._endpoints)

    def get_status(self) -> list[dict]:
        """Return status of all endpoints (for debugging/monitoring)."""
        now = time.time()
        return [
            {
                "provider": e.provider,
                "model": e.model_id,
                "successes": e.success_count,
                "failures": e.failure_count,
                "on_cooldown": e.cooldown_until > now,
                "cooldown_remaining": max(0, e.cooldown_until - now),
            }
            for e in self._endpoints
        ]

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _next_endpoint(self, tried: set) -> Optional[ModelEndpoint]:
        """Pick the next available endpoint using round-robin, skipping cooldowns."""
        now = time.time()
        n = len(self._endpoints)

        with self._lock:
            for _ in range(n):
                idx = self._current_index % n
                self._current_index += 1
                ep = self._endpoints[idx]

                if id(ep) in tried:
                    continue
                if ep.cooldown_until > now:
                    continue

                return ep

        return None

    def _shortest_cooldown(self) -> float:
        """Return seconds until the next endpoint comes off cooldown."""
        now = time.time()
        remaining = [
            ep.cooldown_until - now
            for ep in self._endpoints
            if ep.cooldown_until > now
        ]
        return min(remaining) if remaining else 0

    def _call_endpoint(
        self,
        endpoint: ModelEndpoint,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Dispatch a chat completion to the appropriate provider."""
        capped_tokens = min(max_tokens, endpoint.max_completion_tokens)

        logger.debug(
            "Calling %s/%s (max_tokens=%d)",
            endpoint.provider,
            endpoint.model_id,
            capped_tokens,
        )

        if endpoint.provider == "groq":
            return self._call_groq(
                endpoint, system_prompt, user_prompt, temperature, capped_tokens
            )
        elif endpoint.provider == "vercel":
            return self._call_vercel(
                endpoint, system_prompt, user_prompt, temperature, capped_tokens
            )
        else:
            raise ValueError(f"Unknown provider: {endpoint.provider}")

    def _call_groq(
        self,
        endpoint: ModelEndpoint,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call via the Groq SDK."""
        if not self._groq_client:
            raise RuntimeError("Groq client not initialized")

        response = self._groq_client.chat.completions.create(
            model=endpoint.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content if content else ""

    def _call_vercel(
        self,
        endpoint: ModelEndpoint,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call via the Vercel AI Gateway (OpenAI-compatible)."""
        if not self._vercel_client:
            raise RuntimeError("Vercel AI Gateway client not initialized")

        response = self._vercel_client.chat.completions.create(
            model=endpoint.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content if content else ""
