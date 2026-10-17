"""Gemini provider wrapper: structured JSON output, retries, pacing, caching, accounting.

The API key is read from the environment (``GEMINI_API_KEY``, loaded from a
local ``.env`` that is never committed). Without a key, or when a call fails
after retries, ``generate_json`` returns ``None`` and every agent falls back
to a deterministic policy, so the pipeline always runs end to end.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from ..config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Tried in order when the current model's free-tier quota is used up.
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3-flash-preview"]
CACHE_DIR = REPO_ROOT / "results" / "llm_cache"


@dataclass
class CallLog:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lat = sorted(self.latency_s)
        return {
            "calls": self.calls, "cache_hits": self.cache_hits, "failures": self.failures,
            "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
            "mean_latency_s": sum(lat) / len(lat) if lat else 0.0,
            "p95_latency_s": lat[int(0.95 * (len(lat) - 1))] if lat else 0.0,
        }


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0, min_interval_s: float = 4.5,
                 max_retries: int = 3, use_cache: bool = True, api_key: str | None = None):
        self.model = model
        self.models = [model] + [m for m in FALLBACK_MODELS if m != model]
        self._exhausted: set[str] = set()
        self.temperature = temperature
        self.min_interval_s = min_interval_s       # free tier is rate limited per minute
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.log = CallLog()
        self._last_call = 0.0
        self._lock = threading.Lock()
        key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = None
        if key:
            from google import genai
            self._client = genai.Client(api_key=key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def _cache_path(self, prompt: str, system: str, schema_name: str) -> Path:
        h = hashlib.sha256(f"{self.model}|{self.temperature}|{schema_name}|{system}|{prompt}".encode()).hexdigest()
        return CACHE_DIR / f"{h[:32]}.json"

    def generate_json(self, prompt: str, schema: type[BaseModel], system: str = "") -> BaseModel | None:
        """Ask for JSON matching ``schema``; returns a validated instance or None."""
        path = self._cache_path(prompt, system, schema.__name__)
        if self.use_cache and path.exists():
            self.log.cache_hits += 1
            try:
                return schema.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                path.unlink()
        if not self.available:
            return None

        from google.genai import types
        for attempt in range(self.max_retries):
            live = [m for m in self.models if m not in self._exhausted]
            if not live:
                return None
            model = live[0]
            config = types.GenerateContentConfig(
                temperature=self.temperature, response_mime_type="application/json",
                response_json_schema=gemini_schema(schema), system_instruction=system or None,
                # 2.5 models think by default; switch it off to keep decisions fast.
                thinking_config=types.ThinkingConfig(thinking_budget=0) if "2.5" in model else None)
            with self._lock:
                wait = self.min_interval_s - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.time()
            t0 = time.perf_counter()
            try:
                resp = self._client.models.generate_content(model=model, contents=prompt, config=config)
                self.log.calls += 1
                self.log.latency_s.append(time.perf_counter() - t0)
                usage = resp.usage_metadata
                if usage:
                    self.log.prompt_tokens += usage.prompt_token_count or 0
                    self.log.output_tokens += usage.candidates_token_count or 0
                parsed = schema.model_validate_json(resp.text)
                if self.use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    path.write_text(parsed.model_dump_json(), encoding="utf-8")
                return parsed
            except Exception as exc:  # network, quota, or schema errors
                self.log.failures += 1
                msg = str(exc)
                if "exceeded your current quota" in msg:
                    self._exhausted.add(model)        # daily quota gone: move to the next model
                    continue
                backoff = 20.0 if "429" in msg or "RESOURCE_EXHAUSTED" in msg else 2.0 * (attempt + 1)
                time.sleep(backoff)
        return None


_UNSUPPORTED = {"exclusiveMinimum", "exclusiveMaximum", "maxLength", "minLength", "title", "default"}


def gemini_schema(schema: type[BaseModel]) -> dict:
    """Pydantic JSON schema without keywords the Gemini API rejects.

    Constraints removed here are still enforced: the reply is validated
    against the full pydantic model afterwards.
    """
    def clean(node):
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k not in _UNSUPPORTED}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node
    return clean(schema.model_json_schema())


def to_prompt_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)
