"""Model / agent backends. Every provider returns the agent's raw text output.

  mock    deterministic keyword rules; no model, runs in milliseconds (unit tests, smoke runs)
  ollama  free local open-source model (default in CI)
  gemini  Google Gemini free tier via REST (optional; needs GEMINI_API_KEY)
  http    your own deployed agent, e.g. the Project 2 FastAPI service
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from evalgate.schema import GoldenCase


class Provider(ABC):
    name: str = "base"

    @property
    def model_id(self) -> str:
        return self.name

    @abstractmethod
    def generate(self, prompt: str, case: GoldenCase) -> str:
        """Return raw agent output for one ticket."""


class MockProvider(Provider):
    """Keyword rules that imitate a reasonable triage agent. Fully deterministic."""

    name = "mock"

    CATEGORY_RULES = [
        ("Cancellation request", ["cancel", "close my account", "auto-renew", "auto renew", "ending our contract"]),
        ("Refund request", ["refund", "money back", "return it", "return for"]),
        ("Billing inquiry", ["charge", "charged", "invoice", "bill", "payment", "price", "card on file", "tax"]),
        ("Technical issue", ["crash", "won't", "wont", "error", "not responding", "stuck", "fails", "drops",
                             "blank", "black screen", "stopped", "sync", "boot"]),
        ("Product inquiry", ["does", "is the", "what", "how many", "which", "compatib", "stock", "warranty", "support"]),
    ]
    CRITICAL_WORDS = ["locked out", "hacked", "fire hazard", "burning", "work has stopped", "can't access",
                      "never made", "black screen", "presentation in"]
    HIGH_WORDS = ["twice", "crash", "suspension", "cracked", "before it ships", "cancelled and", "$600"]
    LOW_WORDS = ["before i buy", "can you tell me", "how do i", "could you email", "explain", "restock",
                 "not urgent", "what's the process", "any tips", "trying to decide", "how many hours",
                 "will the", "does your", "what does", "how do i return", "please cancel my monthly",
                 "please cancel the technician"]

    def generate(self, prompt: str, case: GoldenCase) -> str:
        text = f"{case.subject} {case.description}".lower()
        category = "Product inquiry"
        for label, words in self.CATEGORY_RULES:
            if any(w in text for w in words):
                category = label
                break
        if any(w in text for w in self.CRITICAL_WORDS):
            priority = "Critical"
        elif any(w in text for w in self.HIGH_WORDS):
            priority = "High"
        elif any(w in text for w in self.LOW_WORDS):
            priority = "Low"
        else:
            priority = "Medium"
        return json.dumps({"category": category, "priority": priority, "reason": "keyword rules"})


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:1.5b",
                 timeout_s: float = 120, temperature: float = 0.0, seed: int = 42, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.seed = seed
        self.retries = retries

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    def generate(self, prompt: str, case: GoldenCase) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",  # ask Ollama to constrain output to valid JSON
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Ollama call failed after {self.retries + 1} attempts: {last_exc}")


class GeminiProvider(Provider):
    """Google Gemini via REST (free tier). Throttled to respect free-tier rate limits."""

    name = "gemini"
    API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str = "gemini-2.5-flash", api_key_env: str = "GEMINI_API_KEY",
                 timeout_s: float = 60, min_interval_s: float = 4.5, retries: int = 3):
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"Gemini provider needs an API key in the {api_key_env} environment variable")
        self.timeout_s = timeout_s
        self.min_interval_s = min_interval_s
        self.retries = retries
        self._last_call = 0.0

    @property
    def model_id(self) -> str:
        return f"gemini/{self.model}"

    def generate(self, prompt: str, case: GoldenCase) -> str:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        last_err = ""
        for attempt in range(self.retries + 1):
            self._last_call = time.monotonic()
            resp = requests.post(self.API.format(model=self.model), json=body, headers=headers, timeout=self.timeout_s)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")


class HttpAgentProvider(Provider):
    """Calls a deployed agent. Contract: POST {"subject", "description"} -> {"category", "priority", ...}."""

    name = "http"

    def __init__(self, url: str = "http://localhost:8000/triage", timeout_s: float = 60):
        self.url = url
        self.timeout_s = timeout_s

    @property
    def model_id(self) -> str:
        return f"http/{self.url}"

    def generate(self, prompt: str, case: GoldenCase) -> str:
        resp = requests.post(self.url, json={"subject": case.subject, "description": case.description},
                             timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.text


def make_provider(provider_cfg: dict[str, Any], name_override: str | None = None) -> Provider:
    name = (name_override or provider_cfg.get("name") or "mock").lower()
    opts = provider_cfg.get(name, {}) or {}
    if name == "mock":
        return MockProvider()
    if name == "ollama":
        return OllamaProvider(**opts)
    if name == "gemini":
        return GeminiProvider(**opts)
    if name == "http":
        return HttpAgentProvider(**opts)
    raise ValueError(f"Unknown provider '{name}'. Use one of: mock, ollama, gemini, http")
