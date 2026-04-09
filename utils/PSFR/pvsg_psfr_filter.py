#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVSG focus-graph (PSFR) key-frame filter on TOON JSON.

Reads ``train_annotations_toon_sft.json`` / ``test_annotations_toon_sft.json`` from ``--input_dir``,
runs per-video **Shi–Tomasi + Lucas–Kanade** patch retention selection (see ``src/key_frame_selection.py``),
and writes filtered arrays to ``--output_dir`` with the **same filenames**. ``image_path`` fields are
**relative to the SceneGraphVLM repo root**.

**Run** (from repo root; needs OpenCV + numpy + tqdm):

  cd /path/to/SceneGraphVLM
  python utils/PSFR/pvsg_psfr_filter.py

  python utils/PSFR/pvsg_psfr_filter.py \\
    --input_dir datasets/annotations/PVSG_annot/data_sft_original \\
    --output_dir datasets/annotations/PVSG_annot/data_sft_psfr \\
    --config utils/PSFR/config_pvsg.json

If output JSON files already exist, skips the heavy pass and prints statistics only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

# utils/PSFR/<this>.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _SCRIPT_DIR.parents[1]
_SRC_DIR = _SCRIPT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from key_frame_selection import select_keyframes_from_frames

DEFAULT_INPUT_DIR = "datasets/annotations/PVSG_annot/data_sft_original"
DEFAULT_OUTPUT_DIR = "datasets/annotations/PVSG_annot/data_sft_psfr"
DEFAULT_TRAIN_JSON = "train_annotations_toon_sft.json"
DEFAULT_TEST_JSON = "test_annotations_toon_sft.json"
DEFAULT_CONFIG = str(_SCRIPT_DIR / "config_pvsg.json")

VIDEO_ID_PATTERN = re.compile(
    r"frames/([^/]+)/\d+\.(?:png|jpg|jpeg)", re.IGNORECASE
)


def resolve_under_repo(repo_root: Path, path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / path_arg).resolve()


def path_for_json(repo_root: Path, file_path: Path) -> str:
    try:
        return str(file_path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(file_path.resolve()).replace("\\", "/")


def normalize_image_path_repo_relative(image_path: str, repo_root: Path) -> str:
    p = Path(image_path)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p.resolve()).replace("\\", "/")
    return str(Path(image_path).as_posix())


def get_video_id(image_path: str) -> str | None:
    m = VIDEO_ID_PATTERN.search(image_path.replace("\\", "/"))
    return m.group(1) if m else None


def load_toon_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def save_toon_json(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def group_records_by_video(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    video_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    video_order: list[str] = []
    for rec in records:
        ip = rec.get("image_path") or rec.get("image") or ""
        vid = get_video_id(str(ip))
        if vid is None:
            continue
        if vid not in video_to_records:
            video_order.append(vid)
        video_to_records[vid].append(rec)
    return [(vid, video_to_records[vid]) for vid in video_order]


def parse_scene_graph_from_toon(text: str) -> tuple[set[str], set[tuple[str, str, str]]]:
    objects: set[str] = set()
    relations: set[tuple[str, str, str]] = set()
    in_obj = False
    in_rel = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith("obj[") and "{" in line:
            in_obj = True
            in_rel = False
            continue
        if line.startswith("rel[") and "{" in line:
            in_rel = True
            in_obj = False
            continue
        if in_obj and line and line[0].isdigit():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                objects.add(parts[1])
        if in_rel and line and line[0].isdigit():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                relations.add((parts[0], parts[1], parts[2]))
    return objects, relations


def parse_scene_graph_from_record(rec: dict[str, Any]) -> tuple[set[str], set[tuple[str, str, str]]]:
    if rec.get("answer_toon"):
        return parse_scene_graph_from_toon(str(rec["answer_toon"]))
    for conv in rec.get("conversations", []):
        if conv.get("from") == "gpt":
            return parse_scene_graph_from_toon(conv.get("value", ""))
    return set(), set()


def _preds_from_triples(rels: set[tuple[str, str, str]]) -> set[str]:
    return {p for (_s, p, _o) in rels}


def aggregate_scene_graph_sets(records: list[dict[str, Any]]) -> tuple[set[str], set[tuple[str, str, str]]]:
    all_objs: set[str] = set()
    all_rels: set[tuple[str, str, str]] = set()
    for rec in records:
        objs, rels = parse_scene_graph_from_record(rec)
        all_objs |= objs
        all_rels |= rels
    return all_objs, all_rels


def process_split_json(
    name: str,
    input_json: Path,
    output_json: Path,
    repo_root: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    records = load_toon_json(input_json)
    for rec in records:
        if "image_path" in rec:
            rec["_rel_path"] = normalize_image_path_repo_relative(str(rec["image_path"]), repo_root)
        elif "image" in rec:
            rec["_rel_path"] = normalize_image_path_repo_relative(str(rec["image"]), repo_root)

    video_groups = group_records_by_video(records)
    total_before = len(records)
    objs_before_set, rels_before_set = aggregate_scene_graph_sets(records)
    pred_before_set = _preds_from_triples(rels_before_set)
    objs_before = len(objs_before_set)
    rels_before = len(pred_before_set)

    dummy = Path(".")
    patching = cfg["patching"]
    shi = cfg["shi_tomasi"]
    lk = cfg["lucas_kanade"]
    selection = cfg["selection"]
    preprocess = cfg.get("preprocess", {})
    visualization = cfg.get("visualization", {}) or {"enabled": False}

    out_records: list[dict[str, Any]] = []
    for _video_id, vrecs in tqdm(video_groups, desc=f"PSFR {name}", leave=True):
        rel_paths = [r.get("_rel_path") or r.get("image_path") or r.get("image", "") for r in vrecs]
        frame_paths = [resolve_under_repo(repo_root, rp) for rp in rel_paths]
        report = select_keyframes_from_frames(
            frames_dir=dummy,
            out_dir=dummy,
            key_frames_dir=dummy,
            patching=patching,
            shi=shi,
            lk=lk,
            selection=selection,
            preprocess=preprocess,
            visualization=visualization,
            frame_paths=frame_paths,
        )
        keyframe_names = set(report["keyframes"])
        for r in vrecs:
            ip = r.get("image_path") or r.get("image", "")
            if Path(str(ip)).name in keyframe_names:
                out_records.append(r)

    for s in out_records:
        s.pop("_rel_path", None)

    cleaned: list[dict[str, Any]] = []
    for s in out_records:
        item = {k: v for k, v in s.items() if not k.startswith("_")}
        if "image_path" in item:
            item["image_path"] = normalize_image_path_repo_relative(str(item["image_path"]), repo_root)
        cleaned.append(item)

    save_toon_json(output_json, cleaned)

    total_after = len(cleaned)
    objs_after_set, rels_after_set = aggregate_scene_graph_sets(cleaned)
    pred_after_set = _preds_from_triples(rels_after_set)
    pct = (100.0 * total_after / total_before) if total_before else 0.0
    return {
        "name": name,
        "total_before": total_before,
        "total_after": total_after,
        "pct_remaining": pct,
        "objs_before": objs_before,
        "objs_after": len(objs_after_set),
        "rels_before": rels_before,
        "rels_after": len(pred_after_set),
        "objs_before_set": objs_before_set,
        "objs_after_set": objs_after_set,
        "pred_before_set": pred_before_set,
        "pred_after_set": pred_after_set,
        "output_path": str(output_json),
    }


def stats_from_existing_json(
    name: str, input_json: Path, output_json: Path
) -> dict[str, Any] | None:
    if not output_json.is_file():
        return None
    records_before = load_toon_json(input_json)
    records_after = load_toon_json(output_json)
    total_before = len(records_before)
    total_after = len(records_after)
    objs_before_set, rels_before_set = aggregate_scene_graph_sets(records_before)
    objs_after_set, rels_after_set = aggregate_scene_graph_sets(records_after)
    pred_before_set = _preds_from_triples(rels_before_set)
    pred_after_set = _preds_from_triples(rels_after_set)
    pct = (100.0 * total_after / total_before) if total_before else 0.0
    return {
        "name": name,
        "total_before": total_before,
        "total_after": total_after,
        "pct_remaining": pct,
        "objs_before": len(objs_before_set),
        "objs_after": len(objs_after_set),
        "rels_before": len(pred_before_set),
        "rels_after": len(pred_after_set),
        "objs_before_set": objs_before_set,
        "objs_after_set": objs_after_set,
        "pred_before_set": pred_before_set,
        "pred_after_set": pred_after_set,
        "output_path": str(output_json),
    }


def print_stats(stats_list: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("PSFR (focus-graph) filtering statistics")
    print("=" * 60)
    for s in stats_list:
        print(f"\n[{s['name']}]")
        print(
            f"  Samples: {s['total_before']} -> {s['total_after']}  ({s['pct_remaining']:.1f}% remaining)"
        )
        print(f"  Unique object categories:  {s['objs_before']} -> {s['objs_after']}")
        print(f"  Unique relation types (predicates):  {s['rels_before']} -> {s['rels_after']}")
        print(f"  Output: {s['output_path']}")

    by_name = {s["name"]: s for s in stats_list}
    if "train" in by_name and "test" in by_name:
        tr, te = by_name["train"], by_name["test"]
        obj_ob = len(tr["objs_before_set"] & te["objs_before_set"])
        obj_oa = len(tr["objs_after_set"] & te["objs_after_set"])
        pred_ob = len(tr["pred_before_set"] & te["pred_before_set"])
        pred_oa = len(tr["pred_after_set"] & te["pred_after_set"])
        print("\n[Train–Test overlap]")
        print(f"  Object categories in both splits:  {obj_ob} (before) -> {obj_oa} (after)")
        print(f"  Relation types in both splits:  {pred_ob} (before) -> {pred_oa} (after)")
    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PVSG PSFR key-frame filter: TOON JSON folder -> data_sft_psfr (same filenames).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo_root", default=str(_REPO_ROOT_DEFAULT), help="SceneGraphVLM repo root")
    p.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Folder with train/test TOON JSON")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output folder (same JSON basenames)")
    p.add_argument("--train-json", default=DEFAULT_TRAIN_JSON, help="Train JSON basename")
    p.add_argument("--test-json", default=DEFAULT_TEST_JSON, help="Test JSON basename")
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Process train JSON only",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="PSFR JSON config (patching, LK, selection, …)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    input_dir = resolve_under_repo(repo_root, args.input_dir)
    output_dir = resolve_under_repo(repo_root, args.output_dir)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    else:
        config_path = config_path.resolve()
    if not config_path.is_file():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_in = input_dir / args.train_json
    test_in = input_dir / args.test_json
    train_out = output_dir / args.train_json
    test_out = output_dir / args.test_json

    if not input_dir.is_dir():
        print(f"Error: input_dir not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not train_in.is_file():
        print(f"Error: missing {train_in}", file=sys.stderr)
        sys.exit(1)

    do_test = not args.train_only and test_in.is_file()
    if not args.train_only and not test_in.is_file():
        print(f"Note: {test_in} not found; train only.", file=sys.stderr)

    need_train = not train_out.is_file()
    need_test = bool(do_test and not test_out.is_file())
    need_run = need_train or need_test

    print(f"repo_root={repo_root}")
    print(f"input_dir={path_for_json(repo_root, input_dir)}")
    print(f"output_dir={path_for_json(repo_root, output_dir)}")
    print(f"config={path_for_json(repo_root, config_path)}")

    if not need_run:
        print("Output JSON already exists; metrics only.")
        stats_list = []
        st = stats_from_existing_json("train", train_in, train_out)
        if st is None:
            print("Error: train output missing.", file=sys.stderr)
            sys.exit(1)
        stats_list.append(st)
        if do_test:
            stt = stats_from_existing_json("test", test_in, test_out)
            if stt is None:
                print("Error: test output missing.", file=sys.stderr)
                sys.exit(1)
            stats_list.append(stt)
        print_stats(stats_list)
        return

    stats_list: list[dict[str, Any]] = []
    if need_train:
        stats_list.append(
            process_split_json("train", train_in, train_out, repo_root, cfg)
        )
    else:
        st = stats_from_existing_json("train", train_in, train_out)
        if st is not None:
            stats_list.append(st)

    if need_test:
        stats_list.append(
            process_split_json("test", test_in, test_out, repo_root, cfg)
        )
    elif do_test:
        stt = stats_from_existing_json("test", test_in, test_out)
        if stt is not None:
            stats_list.append(stt)

    print_stats(stats_list)


if __name__ == "__main__":
    main()
