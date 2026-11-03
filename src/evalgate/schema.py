"""Data models for test cases, predictions and per-case results."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    """One labeled ticket from the golden set."""

    id: str
    subject: str
    description: str
    category: str  # gold label
    priority: str  # gold label
    tags: list[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    """What the agent predicted, after parsing + label normalization."""

    category: str
    priority: str
    reason: str | None = None


class CaseResult(BaseModel):
    """Everything we log for one test case."""

    id: str
    gold_category: str
    gold_priority: str
    pred_category: str | None
    pred_priority: str | None
    valid_output: bool
    latency_s: float
    raw_output: str
    error: str | None = None

    @property
    def category_correct(self) -> bool:
        return self.valid_output and self.pred_category == self.gold_category

    @property
    def priority_correct(self) -> bool:
        return self.valid_output and self.pred_priority == self.gold_priority


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    """Load a JSONL file (one JSON object per line). Blank lines and # comments are skipped."""
    cases: list[GoldenCase] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(GoldenCase(**json.loads(line)))
        except Exception as exc:  # noqa: BLE001 - surface the bad line clearly
            raise ValueError(f"{path}:{line_no}: invalid test case: {exc}") from exc
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate test case ids in {path}: {sorted(dupes)}")
    return cases
