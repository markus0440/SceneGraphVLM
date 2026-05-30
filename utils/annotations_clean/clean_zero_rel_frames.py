#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REL_HEADER_RE = re.compile(r"rel\[(\d+)\]\{\s*subj\s*,\s*pred\s*,\s*obj\s*\}\s*:", re.IGNORECASE)
END_ANSWER_RE = re.compile(r"</answer>", re.IGNORECASE)


def find_jsonl_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.jsonl"):
        if "__MACOSX" in path.parts:
            continue
        if path.stem.endswith("_clean"):
            continue
        yield path


def get_assistant_text(record: dict) -> str:
    messages = record.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "") or ""
    return ""


def parse_relation_block(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (declared_rel_count, actual_relation_lines).
    If no rel[...] block is found, returns (None, None).
    """
    header = REL_HEADER_RE.search(text)
    if not header:
        return None, None

    declared = int(header.group(1))
    lines = text.splitlines()
    rel_idx = None
    for i, line in enumerate(lines):
        if REL_HEADER_RE.search(line):
            rel_idx = i
            break

    actual = 0
    if rel_idx is not None:
        for line in lines[rel_idx + 1 :]:
            if END_ANSWER_RE.search(line):
                break
            if line.strip():
                actual += 1

    return declared, actual


def is_zero_rel_record(record: dict) -> bool:
    text = get_assistant_text(record)
    declared, actual = parse_relation_block(text)
    if declared is None:
        return False
    return declared == 0 or actual == 0


def clean_file(src: Path, dst: Path) -> Dict[str, int]:
    stats = {
        "total": 0,
        "removed_zero_rel": 0,
        "kept": 0,
        "invalid_json": 0,
        "missing_rel_block": 0,
    }

    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            stats["total"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue

            text = get_assistant_text(record)
            declared, actual = parse_relation_block(text)
            if declared is None:
                stats["missing_rel_block"] += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                continue

            if declared == 0 or actual == 0:
                stats["removed_zero_rel"] += 1
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


def build_clean_path(src: Path) -> Path:
    return src.with_name(f"{src.stem}_clean{src.suffix}")


def process_tree(input_root: Path) -> Dict[str, Dict[str, int]]:
    report: Dict[str, Dict[str, int]] = {}
    for src in sorted(find_jsonl_files(input_root)):
        rel = src.relative_to(input_root)
        dst = build_clean_path(src)
        report[str(rel)] = clean_file(src, dst)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy PSG/PVSG helper: remove zero-relation frames from JSONL files "
            "and write *_clean.jsonl next to the sources. Prefer "
            "datasets/tools/build_annotation_variants.py for final annotations."
        )
    )
    parser.add_argument(
        "input_root",
        help="Path to folder containing annotation files (relative or absolute).",
    )
    parser.add_argument(
        "--report-name",
        default="zero_rel_cleanup_report.json",
        help="Report filename to save under input_root.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")

    report = process_tree(input_root)

    totals = {
        "files_processed": len(report),
        "records_total": sum(v["total"] for v in report.values()),
        "records_removed_zero_rel": sum(v["removed_zero_rel"] for v in report.values()),
        "records_kept": sum(v["kept"] for v in report.values()),
        "invalid_json": sum(v["invalid_json"] for v in report.values()),
        "missing_rel_block": sum(v["missing_rel_block"] for v in report.values()),
    }

    report_payload = {
        "input_root": str(input_root),
        "output_mode": "in-place with *_clean suffix",
        "totals": totals,
        "files": report,
    }

    report_path = input_root / args.report_name
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(report_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
