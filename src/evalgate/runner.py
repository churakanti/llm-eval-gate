"""Run the golden set through a provider and collect per-case results."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone

from evalgate.providers import Provider
from evalgate.schema import CaseResult, GoldenCase
from evalgate.triage import build_prompt, load_prompt, parse_triage_output


def git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a git repo / git missing is fine
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cases(cases: list[GoldenCase], provider: Provider, prompt_path: str,
              categories: list[str], priorities: list[str], verbose: bool = True) -> list[CaseResult]:
    template = load_prompt(prompt_path)
    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        prompt = build_prompt(template, case, categories, priorities)
        start = time.perf_counter()
        raw, error = "", None
        try:
            raw = provider.generate(prompt, case)
        except Exception as exc:  # noqa: BLE001 - a failed call is a failed case, not a crashed run
            error = f"provider error: {exc}"
        latency = time.perf_counter() - start

        parsed = None
        if error is None:
            parsed, error = parse_triage_output(raw, categories, priorities)
        result = CaseResult(
            id=case.id,
            gold_category=case.category,
            gold_priority=case.priority,
            pred_category=parsed.category if parsed else None,
            pred_priority=parsed.priority if parsed else None,
            valid_output=parsed is not None,
            latency_s=latency,
            raw_output=raw,
            error=error,
        )
        results.append(result)
        if verbose:
            mark = "ok " if result.category_correct and result.priority_correct else "MISS"
            print(f"[{i:>3}/{len(cases)}] {mark} {case.id:<6} {case.category:>20} -> "
                  f"{(result.pred_category or 'INVALID'):<20} {case.priority:>8} -> "
                  f"{(result.pred_priority or 'INVALID'):<8} {latency:6.2f}s", flush=True)
    return results
