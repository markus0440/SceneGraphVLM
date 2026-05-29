#!/usr/bin/env python3
"""
Aggregate speed_test_infer JSON files into CSV / pivot JSON; print ASCII comparison table.

Example:
  python metrics/statistics/speed-test-infer/py_scripts/aggregate_speed_results.py \\
    --results-dir metrics/results/speed_test_infer
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_STATS_DIR = Path(__file__).resolve().parent
if str(_STATS_DIR) not in sys.path:
    sys.path.insert(0, str(_STATS_DIR))

from speed_test_utils import SUMMARY_METRIC_META  # noqa: E402

_SKIP_JSON = {"pivot_tokens_per_second.json"}


def _block_val(aggregate: Dict[str, Any], key: str, field: str) -> Optional[float]:
    block = aggregate.get(key) or {}
    v = block.get(field)
    return float(v) if v is not None else None


def _block_phys(aggregate: Dict[str, Any], key: str) -> Optional[str]:
    block = aggregate.get(key) or {}
    return block.get("phys")


def load_result(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_run_json(path: Path) -> bool:
    if path.name.endswith(".tmp") or path.name in _SKIP_JSON:
        return False
    if path.name.startswith("pivot_"):
        return False
    return path.suffix == ".json"


def build_run_row(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    agg = data.get("aggregate") or {}
    model = data.get("model_display_name") or data.get("model") or path.stem
    backend = data.get("accelerator") or data.get("infer_backend") or "?"
    row: Dict[str, Any] = {
        "file": path.name,
        "gpu": data.get("gpu"),
        "model": model,
        "accelerator": backend,
        "num_samples": data.get("num_measured_samples"),
        "status": data.get("status"),
    }
    for key in SUMMARY_METRIC_META:
        row[f"{key}_mean"] = _block_val(agg, key, "mean")
        row[f"{key}_std"] = _block_val(agg, key, "std")
        row[f"{key}_median"] = _block_val(agg, key, "median")
        row[f"{key}_phys"] = _block_phys(agg, key)
    return row


def format_all_runs_ascii(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "Speed test — no runs\n"

    lines = [
        "Speed test — all runs",
        "",
        "| GPU | Model | Accelerator | N | tok/s (mean ± σ) | TTFT s | Input tok | Visual tok | Text tok |",
        "|-----|-------|-------------|---|------------------|--------|-----------|------------|----------|",
    ]

    for r in rows:
        def phys(metric: str) -> str:
            p = r.get(f"{metric}_phys")
            if p:
                return p
            m, s = r.get(f"{metric}_mean"), r.get(f"{metric}_std")
            if m is None:
                return "—"
            if s is not None and (r.get("num_samples") or 0) >= 2:
                return f"({m:.4g} ± {s:.4g})"
            return f"{m:.4g}"

        lines.append(
            f"| {r.get('gpu') or '—'} | {r.get('model') or '—'} | {r.get('accelerator') or '—'} | "
            f"{r.get('num_samples') or '—'} | {phys('tokens_per_second')} | {phys('time_to_first_token_sec')} | "
            f"{phys('input_tokens_total')} | {phys('visual_tokens')} | {phys('text_prompt_tokens_total')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate speed-test JSON → CSV; print ASCII table.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("metrics/results/speed_test_infer"),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-pivot-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="Do not print table to stdout.")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    json_files = sorted(
        p for p in results_dir.rglob("*.json")
        if _is_run_json(p) and "logs" not in p.parts
    )
    rows: List[Dict[str, Any]] = []
    pivot: Dict[str, Dict[str, Optional[float]]] = {}

    for path in json_files:
        data = load_result(path)
        if data.get("status") not in ("ok", None) and not data.get("aggregate"):
            continue
        row = build_run_row(data, path)
        rows.append(row)
        model = row["model"]
        backend = row["accelerator"]
        pivot.setdefault(str(model), {})[str(backend)] = row.get("tokens_per_second_mean")

    out_csv = args.output_csv or results_dir / "summary.csv"
    out_pivot = args.output_pivot_json or results_dir / "pivot_tokens_per_second.json"

    if rows:
        fieldnames: List[str] = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    out_pivot.write_text(json.dumps(pivot, ensure_ascii=False, indent=2), encoding="utf-8")

    table = format_all_runs_ascii(rows)
    print(f"[done] {len(rows)} runs → {out_csv}")
    print(f"[done] pivot → {out_pivot}")
    if not args.quiet and table:
        print()
        print(table)


if __name__ == "__main__":
    main()
