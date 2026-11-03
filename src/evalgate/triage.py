"""Prompt building and robust parsing of the triage agent's output.

Small local models often wrap JSON in ```code fences```, add chatter, or use
slightly different label spellings ("billing" vs "Billing inquiry"). The parser
handles those cases so we measure the model's *judgment*, while still counting
genuinely unusable output as invalid (valid_output_rate metric).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template

from evalgate.schema import GoldenCase, TriageResult

# Extra spellings a model might produce -> canonical label (matched case-insensitively).
CATEGORY_SYNONYMS = {
    "technical": "Technical issue",
    "technical support": "Technical issue",
    "tech issue": "Technical issue",
    "bug": "Technical issue",
    "billing": "Billing inquiry",
    "billing issue": "Billing inquiry",
    "payment": "Billing inquiry",
    "product": "Product inquiry",
    "product question": "Product inquiry",
    "pre-sales": "Product inquiry",
    "refund": "Refund request",
    "return": "Refund request",
    "cancellation": "Cancellation request",
    "cancel": "Cancellation request",
}
PRIORITY_SYNONYMS = {
    "p1": "Critical",
    "urgent": "Critical",
    "p2": "High",
    "p3": "Medium",
    "normal": "Medium",
    "p4": "Low",
}


def load_prompt(path: str | Path) -> Template:
    return Template(Path(path).read_text(encoding="utf-8"))


def build_prompt(template: Template, case: GoldenCase, categories: list[str], priorities: list[str]) -> str:
    """Fill the prompt. Templates use $subject, $description, $categories, $priorities."""
    return template.safe_substitute(
        subject=case.subject,
        description=case.description,
        categories=", ".join(categories),
        priorities=", ".join(priorities),
    )


def extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of model output (handles fences and surrounding text)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to scanning for a balanced {...} block.
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = cleaned.find("{", start + 1)
    return None


def normalize_label(value: object, allowed: list[str], synonyms: dict[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().strip(".").lower()
    for label in allowed:
        if v == label.lower():
            return label
    if v in synonyms and synonyms[v] in allowed:
        return synonyms[v]
    return None


def parse_triage_output(raw: str, categories: list[str], priorities: list[str]) -> tuple[TriageResult | None, str | None]:
    """Return (result, error). result is None when the output is unusable."""
    obj = extract_json_object(raw)
    if obj is None:
        return None, "no JSON object found in output"
    # Accept common key variants.
    lowered = {str(k).lower(): v for k, v in obj.items()}
    cat_raw = lowered.get("category", lowered.get("ticket_category"))
    pri_raw = lowered.get("priority", lowered.get("urgency"))
    category = normalize_label(cat_raw, categories, CATEGORY_SYNONYMS)
    priority = normalize_label(pri_raw, priorities, PRIORITY_SYNONYMS)
    if category is None or priority is None:
        return None, f"label not in allowed set (category={cat_raw!r}, priority={pri_raw!r})"
    reason = lowered.get("reason")
    return TriageResult(category=category, priority=priority, reason=reason if isinstance(reason, str) else None), None
