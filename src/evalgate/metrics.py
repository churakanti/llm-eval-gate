"""Scoring. Pure functions, no dependencies, so they're easy to unit-test and explain."""

from __future__ import annotations

import math
from collections import Counter

from evalgate.schema import CaseResult

URGENT = {"High", "Critical"}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def per_class_f1(results: list[CaseResult], labels: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for r in results if r.pred_category == label and r.gold_category == label)
        fp = sum(1 for r in results if r.pred_category == label and r.gold_category != label)
        fn = sum(1 for r in results if r.gold_category == label and r.pred_category != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                      "support": tp + fn}
    return out


def confusion_matrix(results: list[CaseResult], labels: list[str]) -> dict[str, dict[str, int]]:
    """rows = gold label, columns = predicted label (+ 'INVALID')."""
    cols = labels + ["INVALID"]
    matrix = {g: {p: 0 for p in cols} for g in labels}
    for r in results:
        if r.gold_category not in matrix:
            continue
        pred = r.pred_category if r.valid_output and r.pred_category in labels else "INVALID"
        matrix[r.gold_category][pred] += 1
    return matrix


def compute_metrics(results: list[CaseResult], categories: list[str], priorities: list[str]) -> dict:
    n = len(results)
    latencies = [r.latency_s for r in results]
    per_class = per_class_f1(results, categories)
    present = [c for c in categories if per_class[c]["support"] > 0]

    pri_index = {p: i for i, p in enumerate(priorities)}
    within_one = sum(
        1 for r in results
        if r.valid_output and r.pred_priority in pri_index
        and abs(pri_index[r.pred_priority] - pri_index[r.gold_priority]) <= 1
    )
    urgent_gold = [r for r in results if r.gold_priority in URGENT]
    urgent_hit = sum(1 for r in urgent_gold if r.valid_output and r.pred_priority in URGENT)

    return {
        "num_cases": n,
        "category_accuracy": round(_safe_div(sum(r.category_correct for r in results), n), 4),
        "category_macro_f1": round(_safe_div(sum(per_class[c]["f1"] for c in present), len(present)), 4),
        "priority_accuracy": round(_safe_div(sum(r.priority_correct for r in results), n), 4),
        "priority_within_one": round(_safe_div(within_one, n), 4),
        "urgent_recall": round(_safe_div(urgent_hit, len(urgent_gold)), 4),
        "valid_output_rate": round(_safe_div(sum(r.valid_output for r in results), n), 4),
        "latency_p50_s": round(percentile(latencies, 50), 3),
        "latency_p95_s": round(percentile(latencies, 95), 3),
        "errors": sum(1 for r in results if r.error),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(results, categories),
        "predicted_category_counts": dict(Counter(r.pred_category or "INVALID" for r in results)),
    }
