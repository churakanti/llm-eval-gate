"""Load eval_config.yaml and apply environment-variable overrides.

Env overrides (handy in Docker / CI, no file edits needed):
  EVAL_PROVIDER      mock | ollama | gemini | http
  OLLAMA_BASE_URL    e.g. http://ollama:11434 inside docker-compose
  OLLAMA_MODEL       e.g. qwen2.5:1.5b
  GEMINI_MODEL       e.g. gemini-2.5-flash
  HTTP_AGENT_URL     e.g. http://localhost:8000/triage (your Project 2 FastAPI service)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/eval_config.yaml") -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    provider = cfg.setdefault("provider", {})

    env_map = {
        ("name",): "EVAL_PROVIDER",
        ("ollama", "base_url"): "OLLAMA_BASE_URL",
        ("ollama", "model"): "OLLAMA_MODEL",
        ("gemini", "model"): "GEMINI_MODEL",
        ("http", "url"): "HTTP_AGENT_URL",
    }
    for keys, env_var in env_map.items():
        value = os.environ.get(env_var)
        if not value:
            continue
        target = provider
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    for key in ("dataset", "prompt", "labels", "thresholds"):
        if key not in cfg:
            raise ValueError(f"eval config is missing required key: '{key}'")
    labels = cfg["labels"]
    if not labels.get("categories") or not labels.get("priorities"):
        raise ValueError("labels.categories and labels.priorities must be non-empty lists")
