"""End-to-end tests using the deterministic mock provider (no model, no network)."""

import json
from pathlib import Path

import yaml

from evalgate.cli import main
from evalgate.config import load_config
from evalgate.schema import load_golden_set

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "eval_config.yaml"


def test_golden_set_is_valid_and_uses_configured_labels():
    cfg = load_config(CONFIG)
    cases = load_golden_set(ROOT / cfg["dataset"])
    assert len(cases) >= 40
    assert {c.category for c in cases} <= set(cfg["labels"]["categories"])
    assert {c.priority for c in cases} <= set(cfg["labels"]["priorities"])
    # every category should be covered by at least 5 cases so per-class F1 means something
    for cat in cfg["labels"]["categories"]:
        assert sum(c.category == cat for c in cases) >= 5, cat


def _write_cfg(tmp_path, **threshold_overrides):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["dataset"] = str(ROOT / cfg["dataset"])
    cfg["prompt"] = str(ROOT / cfg["prompt"])
    cfg["regression"]["baseline_path"] = str(tmp_path / "baseline.json")
    cfg["thresholds"].update(threshold_overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_mock_run_passes_and_writes_reports(tmp_path):
    out = tmp_path / "reports"
    code = main(["run", "--config", str(_write_cfg(tmp_path)), "--provider", "mock",
                 "--output-dir", str(out), "--quiet", "--update-baseline"])
    assert code == 0
    for name in ("metrics.json", "predictions.csv", "summary.md", "report.html", "history.jsonl"):
        assert (out / name).exists(), name
    run = json.loads((out / "metrics.json").read_text())
    assert run["gate"]["passed"] is True
    assert json.loads((tmp_path / "baseline.json").read_text())["model"] == "mock"


def test_gate_blocks_when_threshold_not_met(tmp_path):
    code = main(["run", "--config", str(_write_cfg(tmp_path, category_accuracy=0.99)), "--provider", "mock",
                 "--output-dir", str(tmp_path / "reports"), "--quiet"])
    assert code == 1


def test_no_fail_flag_reports_only(tmp_path):
    code = main(["run", "--config", str(_write_cfg(tmp_path, category_accuracy=0.99)), "--provider", "mock",
                 "--output-dir", str(tmp_path / "reports"), "--quiet", "--no-fail"])
    assert code == 0


def test_promote_writes_baseline_from_run(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    out = tmp_path / "reports"
    assert main(["run", "--config", str(cfg_path), "--provider", "mock", "--output-dir", str(out), "--quiet"]) == 0
    assert main(["promote", str(out / "metrics.json"), "--config", str(cfg_path)]) == 0
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["model"] == "mock" and "category_accuracy" in baseline["metrics"]
