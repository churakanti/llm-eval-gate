import json

from evalgate.gate import check_regression, check_thresholds, evaluate_gate


def test_thresholds_min_and_max():
    metrics = {"category_accuracy": 0.8, "latency_p95_s": 40.0}
    checks = {c["check"]: c["passed"] for c in check_thresholds(metrics, {"category_accuracy": 0.7,
                                                                           "latency_p95_s": 30.0})}
    assert checks == {"category_accuracy": True, "latency_p95_s": False}


def test_missing_metric_fails():
    assert check_thresholds({}, {"category_accuracy": 0.7})[0]["passed"] is False


def test_regression_allows_small_drop_blocks_large_drop():
    base = {"metrics": {"category_accuracy": 0.90, "urgent_recall": 0.90}}
    now = {"category_accuracy": 0.87, "urgent_recall": 0.80}
    checks = {c["check"]: c["passed"] for c in check_regression(now, base, ["category_accuracy", "urgent_recall"], 0.05)}
    assert checks["category_accuracy vs baseline"] is True
    assert checks["urgent_recall vs baseline"] is False


def test_improvement_passes_regression():
    checks = check_regression({"category_accuracy": 0.95}, {"metrics": {"category_accuracy": 0.8}},
                              ["category_accuracy"], 0.05)
    assert checks[0]["passed"] is True


def _cfg(tmp_path, baseline):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    return {"thresholds": {"category_accuracy": 0.5},
            "regression": {"baseline_path": str(path), "max_drop": 0.05, "metrics": ["category_accuracy"]}}


def test_gate_fails_on_regression(tmp_path):
    cfg = _cfg(tmp_path, {"model": "ollama/m", "metrics": {"category_accuracy": 0.9}})
    gate = evaluate_gate({"category_accuracy": 0.7}, cfg, "ollama/m")
    assert gate["passed"] is False and gate["regression_checked"] is True


def test_gate_skips_regression_for_different_model(tmp_path):
    cfg = _cfg(tmp_path, {"model": "ollama/m", "metrics": {"category_accuracy": 0.9}})
    gate = evaluate_gate({"category_accuracy": 0.7}, cfg, "mock")
    assert gate["passed"] is True and gate["regression_checked"] is False and "skipped" in gate["note"]


def test_gate_without_baseline(tmp_path):
    cfg = {"thresholds": {"category_accuracy": 0.5},
           "regression": {"baseline_path": str(tmp_path / "missing.json"), "metrics": ["category_accuracy"]}}
    gate = evaluate_gate({"category_accuracy": 0.7}, cfg, "mock")
    assert gate["passed"] is True and "No baseline" in gate["note"]
