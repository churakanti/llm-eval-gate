"""Write run outputs: metrics.json, predictions.csv, summary.md (PR comment), report.html, history.jsonl."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from evalgate.schema import CaseResult

COMMENT_MARKER = "<!-- evalgate-report -->"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return "-" if v is None else str(v)


def write_predictions_csv(results: list[CaseResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "gold_category", "pred_category", "category_correct", "gold_priority", "pred_priority",
                    "priority_correct", "valid_output", "latency_s", "error", "raw_output"])
        for r in results:
            w.writerow([r.id, r.gold_category, r.pred_category, r.category_correct, r.gold_priority,
                        r.pred_priority, r.priority_correct, r.valid_output, round(r.latency_s, 3), r.error or "",
                        r.raw_output[:500]])


def build_summary_md(run: dict, results: list[CaseResult], max_failures: int = 10) -> str:
    gate = run["gate"]
    m = run["metrics"]
    status = "✅ PASSED" if gate["passed"] else "❌ FAILED"
    lines = [
        COMMENT_MARKER,
        f"## LLM Eval Gate: {status}",
        "",
        f"**Model:** `{run['model']}` · **Prompt:** `{run['prompt']}` · **Cases:** {m['num_cases']} · "
        f"**Commit:** `{(run.get('git_sha') or 'local')[:7]}`",
        "",
        "| Check | Value | Requirement | Result |",
        "|---|---|---|---|",
    ]
    for c in gate["checks"]:
        lines.append(f"| {c['check']} | {_fmt(c['value'])} | {c['detail']} | {'✅' if c['passed'] else '❌'} |")
    if gate.get("note"):
        lines += ["", f"> ℹ️ {gate['note']}"]

    lines += ["", "<details><summary>Per-category F1</summary>", "", "| Category | Precision | Recall | F1 | Support |",
              "|---|---|---|---|---|"]
    for cat, s in m["per_class"].items():
        lines.append(f"| {cat} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {s['support']} |")
    lines += ["", "</details>"]

    failures = [r for r in results if not r.category_correct or not r.priority_correct]
    if failures:
        lines += ["", f"<details><summary>Misclassified cases ({len(failures)})</summary>", "",
                  "| id | gold category | predicted | gold priority | predicted |", "|---|---|---|---|---|"]
        for r in failures[:max_failures]:
            lines.append(f"| {r.id} | {r.gold_category} | {r.pred_category or 'INVALID'} | {r.gold_priority} | "
                         f"{r.pred_priority or 'INVALID'} |")
        if len(failures) > max_failures:
            lines.append(f"| … | {len(failures) - max_failures} more in predictions.csv | | | |")
        lines += ["", "</details>"]
    lines += ["", f"Latency p50 {m['latency_p50_s']}s · p95 {m['latency_p95_s']}s · "
                  f"full report in the `eval-report` workflow artifact."]
    return "\n".join(lines) + "\n"


def build_html(run: dict, results: list[CaseResult]) -> str:
    gate, m = run["gate"], run["metrics"]
    esc = html.escape
    ok = gate["passed"]
    check_rows = "".join(
        f"<tr><td>{esc(c['check'])}</td><td>{esc(_fmt(c['value']))}</td><td>{esc(c['detail'])}</td>"
        f"<td class='{'pass' if c['passed'] else 'fail'}'>{'PASS' if c['passed'] else 'FAIL'}</td></tr>"
        for c in gate["checks"])
    cats = list(m["confusion_matrix"].keys())
    cols = cats + ["INVALID"]
    cm_head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    cm_rows = ""
    for g in cats:
        cells = ""
        for p in cols:
            v = m["confusion_matrix"][g][p]
            cls = "diag" if g == p and v else ("off" if v else "")
            cells += f"<td class='{cls}'>{v}</td>"
        cm_rows += f"<tr><th>{esc(g)}</th>{cells}</tr>"
    case_rows = "".join(
        f"<tr class='{'' if (r.category_correct and r.priority_correct) else 'miss'}'><td>{esc(r.id)}</td>"
        f"<td>{esc(r.gold_category)}</td><td>{esc(r.pred_category or 'INVALID')}</td>"
        f"<td>{esc(r.gold_priority)}</td><td>{esc(r.pred_priority or 'INVALID')}</td>"
        f"<td>{r.latency_s:.2f}s</td><td>{esc(r.error or '')}</td></tr>"
        for r in results)
    note = f"<p class='note'>{esc(gate['note'])}</p>" if gate.get("note") else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval Gate Report</title>
<style>
:root{{--bg:#fff;--fg:#1d2330;--muted:#5d6675;--line:#e3e6ec;--pass:#1a7f4b;--fail:#c0362c;--tint:#f5f7fa}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171c;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2a2f38;--pass:#4cc38a;--fail:#ff6b5e;--tint:#1c2028}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1000px;margin-inline:auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}}
.meta{{color:var(--muted);font-size:13px}} .badge{{display:inline-block;padding:2px 10px;border-radius:99px;color:#fff;
background:{'var(--pass)' if ok else 'var(--fail)'};font-weight:600;font-size:13px;margin-left:8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}}
.card{{background:var(--tint);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.card b{{display:block;font-size:22px}} .card span{{color:var(--muted);font-size:12px}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}} th{{color:var(--muted);font-weight:600}}
.pass{{color:var(--pass);font-weight:700}} .fail{{color:var(--fail);font-weight:700}}
td.diag{{background:color-mix(in srgb,var(--pass) 18%,transparent)}} td.off{{background:color-mix(in srgb,var(--fail) 18%,transparent)}}
tr.miss td{{background:color-mix(in srgb,var(--fail) 8%,transparent)}} .note{{color:var(--muted);font-style:italic}}
</style></head><body>
<h1>LLM Eval Gate <span class="badge">{'PASSED' if ok else 'FAILED'}</span></h1>
<div class="meta">Model {esc(run['model'])} · Prompt {esc(run['prompt'])} · {esc(run['created_at'])} · commit {esc((run.get('git_sha') or 'local')[:7])}</div>
<div class="cards">
<div class="card"><b>{m['category_accuracy']:.0%}</b><span>Category accuracy</span></div>
<div class="card"><b>{m['category_macro_f1']:.2f}</b><span>Macro F1</span></div>
<div class="card"><b>{m['priority_accuracy']:.0%}</b><span>Priority accuracy</span></div>
<div class="card"><b>{m['urgent_recall']:.0%}</b><span>Urgent recall</span></div>
<div class="card"><b>{m['valid_output_rate']:.0%}</b><span>Valid output</span></div>
<div class="card"><b>{m['latency_p95_s']:.1f}s</b><span>Latency p95</span></div>
</div>
<h2>Gate checks</h2>{note}
<div class="wrap"><table><tr><th>Check</th><th>Value</th><th>Requirement</th><th>Result</th></tr>{check_rows}</table></div>
<h2>Confusion matrix (rows = gold, columns = predicted)</h2>
<div class="wrap"><table><tr><th></th>{cm_head}</tr>{cm_rows}</table></div>
<h2>All cases</h2>
<div class="wrap"><table><tr><th>id</th><th>Gold category</th><th>Predicted</th><th>Gold priority</th><th>Predicted</th><th>Latency</th><th>Error</th></tr>{case_rows}</table></div>
</body></html>
"""


def write_reports(run: dict, results: list[CaseResult], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out / "metrics.json",
        "predictions": out / "predictions.csv",
        "summary": out / "summary.md",
        "html": out / "report.html",
        "history": out / "history.jsonl",
    }
    paths["metrics"].write_text(json.dumps(run, indent=2), encoding="utf-8")
    write_predictions_csv(results, paths["predictions"])
    paths["summary"].write_text(build_summary_md(run, results), encoding="utf-8")
    paths["html"].write_text(build_html(run, results), encoding="utf-8")
    history_row = {k: run[k] for k in ("created_at", "git_sha", "model", "prompt")}
    history_row["passed"] = run["gate"]["passed"]
    history_row.update({k: v for k, v in run["metrics"].items() if isinstance(v, (int, float))})
    with paths["history"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(history_row) + "\n")
    return paths
