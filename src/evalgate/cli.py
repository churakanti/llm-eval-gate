"""Command line entry point.

  python -m evalgate run                       # run with config/eval_config.yaml (provider from config)
  python -m evalgate run --provider mock       # fast deterministic smoke run, no model needed
  python -m evalgate run --update-baseline     # record current metrics as the new baseline
  python -m evalgate run --prompt prompts/triage_v0_bad.txt   # watch the gate fail
  python -m evalgate promote path/to/metrics.json             # baseline from a CI artifact

Exit codes: 0 = gate passed, 1 = gate failed, 2 = setup/config error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalgate.config import load_config
from evalgate.gate import evaluate_gate
from evalgate.metrics import compute_metrics
from evalgate.providers import make_provider
from evalgate.report import write_reports
from evalgate.runner import git_sha, run_cases, utc_now
from evalgate.schema import load_golden_set


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalgate", description="LLM regression tests + CI quality gate")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the eval suite and apply the gate")
    run.add_argument("--config", default="config/eval_config.yaml")
    run.add_argument("--provider", choices=["mock", "ollama", "gemini", "http"], help="override provider")
    run.add_argument("--prompt", help="override prompt template path")
    run.add_argument("--dataset", help="override golden set path")
    run.add_argument("--limit", type=int, help="only run the first N cases (quick local checks)")
    run.add_argument("--output-dir", default="reports")
    run.add_argument("--update-baseline", action="store_true", help="save this run's metrics as the baseline")
    run.add_argument("--no-fail", action="store_true", help="always exit 0 (report only)")
    run.add_argument("--quiet", action="store_true")
    promote = sub.add_parser("promote", help="turn a run's metrics.json (e.g. from the CI artifact) into the baseline")
    promote.add_argument("metrics_json", help="path to a metrics.json written by `evalgate run`")
    promote.add_argument("--config", default="config/eval_config.yaml")
    return p


def write_baseline(run: dict, cfg: dict) -> Path:
    baseline_path = Path(cfg.get("regression", {}).get("baseline_path", "baseline/baseline_metrics.json"))
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    tracked = {k: v for k, v in run["metrics"].items() if isinstance(v, (int, float))}
    meta = {k: run.get(k) for k in ("created_at", "git_sha", "model", "prompt")}
    baseline_path.write_text(json.dumps(meta | {"metrics": tracked}, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline updated: {baseline_path}  (commit this file)")
    return baseline_path


def cmd_promote(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
        run = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
        if "metrics" not in run:
            raise ValueError("file has no 'metrics' key - pass the metrics.json from an evalgate run")
    except Exception as exc:  # noqa: BLE001
        print(f"Setup error: {exc}", file=sys.stderr)
        return 2
    write_baseline(run, cfg)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
        if args.prompt:
            cfg["prompt"] = args.prompt
        if args.dataset:
            cfg["dataset"] = args.dataset
        cases = load_golden_set(cfg["dataset"])
        if args.limit:
            cases = cases[: args.limit]
        provider = make_provider(cfg["provider"], args.provider)
    except Exception as exc:  # noqa: BLE001
        print(f"Setup error: {exc}", file=sys.stderr)
        return 2

    categories, priorities = cfg["labels"]["categories"], cfg["labels"]["priorities"]
    unknown = {c.category for c in cases} - set(categories) | {c.priority for c in cases} - set(priorities)
    if unknown:
        print(f"Setup error: golden set uses labels not in config: {sorted(unknown)}", file=sys.stderr)
        return 2

    print(f"Running {len(cases)} cases with {provider.model_id} and prompt {cfg['prompt']}\n")
    results = run_cases(cases, provider, cfg["prompt"], categories, priorities, verbose=not args.quiet)
    metrics = compute_metrics(results, categories, priorities)

    run = {
        "created_at": utc_now(),
        "git_sha": git_sha(),
        "model": provider.model_id,
        "prompt": cfg["prompt"],
        "dataset": cfg["dataset"],
        "metrics": metrics,
    }
    run["gate"] = evaluate_gate(metrics, cfg, provider.model_id)
    paths = write_reports(run, results, args.output_dir)

    print("\n" + "=" * 60)
    for c in run["gate"]["checks"]:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['check']:<34} {c['value']!s:<8} {c['detail']}")
    if run["gate"].get("note"):
        print(f"  note: {run['gate']['note']}")
    print("=" * 60)
    print(f"GATE {'PASSED' if run['gate']['passed'] else 'FAILED'}  ·  report: {paths['html']}")

    if args.update_baseline:
        write_baseline(run, cfg)

    if args.no_fail:
        return 0
    return 0 if run["gate"]["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "promote":
        return cmd_promote(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
