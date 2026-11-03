"""The quality gate: absolute thresholds + regression check against a committed baseline.

A check fails the gate when:
  * a metric is below its minimum threshold (or above it, for latency), or
  * a tracked metric dropped more than `max_drop` versus baseline/baseline_metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

# Metrics where *lower* is better; their thresholds are maximums.
LOWER_IS_BETTER = {"latency_p50_s", "latency_p95_s"}


def check_thresholds(metrics: dict, thresholds: dict) -> list[dict]:
    checks = []
    for name, limit in thresholds.items():
        if name not in metrics:
            checks.append({"check": name, "type": "threshold", "value": None, "limit": limit,
                           "passed": False, "detail": "metric not computed"})
            continue
        value = metrics[name]
        if name in LOWER_IS_BETTER:
            passed, op = value <= limit, "<="
        else:
            passed, op = value >= limit, ">="
        checks.append({"check": name, "type": "threshold", "value": value, "limit": limit,
                       "passed": passed, "detail": f"needs {op} {limit}"})
    return checks


def load_baseline(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def check_regression(metrics: dict, baseline: dict | None, tracked: list[str], max_drop: float) -> list[dict]:
    if baseline is None:
        return []
    base_metrics = baseline.get("metrics", baseline)
    checks = []
    for name in tracked:
        if name not in base_metrics or name not in metrics:
            continue
        base, value = base_metrics[name], metrics[name]
        drop = round(base - value, 4)
        checks.append({"check": f"{name} vs baseline", "type": "regression", "value": value, "limit": base,
                       "passed": drop <= max_drop,
                       "detail": f"baseline {base}, change {-drop:+.4f} (max allowed drop {max_drop})"})
    return checks


def evaluate_gate(metrics: dict, cfg: dict, model_id: str | None = None) -> dict:
    reg_cfg = cfg.get("regression", {}) or {}
    baseline = load_baseline(reg_cfg.get("baseline_path", "baseline/baseline_metrics.json"))
    note = None
    if baseline is None:
        note = "No baseline committed yet - only absolute thresholds were checked. Run with --update-baseline."
    elif model_id and baseline.get("model") and baseline["model"] != model_id:
        # Comparing different models is apples-to-oranges, so skip the regression check.
        note = f"Baseline was recorded with {baseline['model']}, this run used {model_id}; regression check skipped."
        baseline = None
    checks = check_thresholds(metrics, cfg.get("thresholds", {}))
    checks += check_regression(metrics, baseline, reg_cfg.get("metrics", []), float(reg_cfg.get("max_drop", 0.05)))
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "regression_checked": baseline is not None,
        "note": note,
    }
