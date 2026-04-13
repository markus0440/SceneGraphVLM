#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVSG key-frame selection: group frames by video, run Shi–Tomasi + Lucas–Kanade selection per video,
output *_focus_graph.jsonl and print frame + scene-graph statistics.

**Deprecated for SceneGraphVLM layout:** use ``utils/PSFR/pvsg_psfr_filter.py`` with TOON JSON folders
(``--input_dir`` / ``--output_dir``, repo-relative ``image_path``). This module keeps the legacy
InternVL jsonl + meta JSON interface.

# ============ Пример запуска (из корня репозитория, conda activate maxinfo; нужны opencv-python, numpy, tqdm) ============
# Train + test:
# python key-frame-selection/src/run.py --train-jsonl data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_train.jsonl --train-meta InternVL-SFT-Finetune/meta_data/pvsg/pvsg_sgg_prev_train_meta.json --test-jsonl data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_test.jsonl --test-meta InternVL-SFT-Finetune/meta_data/pvsg/pvsg_sgg_prev_test_meta.json --output-dir data_playground/pvsg_sgg
#
# Только train:
#   python key-frame-selection/src/run_pvsg.py --train-jsonl data_playground/pvsg_sgg/pvsg_sgg_train.jsonl --train-meta InternVL-SFT-Finetune/internvl_chat/shell/data/pvsg_sgg_train_meta.json --output-dir data_playground/pvsg_sgg
# =============================================================================================================
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Repo layout: this file is in key-frame-selection/src/
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from key_frame_selection import select_keyframes_from_frames

# --- Video ID from image path: e.g. "train_images_scaled/vidor/frames/0001_4164158586/0000.png" -> "0001_4164158586"
VIDEO_ID_PATTERN = re.compile(r"frames/([^/]+)/\d+\.(?:png|jpg|jpeg)", re.IGNORECASE)


def get_video_id(image_path: str):
    m = VIDEO_ID_PATTERN.search(image_path)
    return m.group(1) if m else None


def load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_meta(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for v in data.values():
        if isinstance(v, dict) and "root" in v:
            return v
    raise ValueError(f"No key with 'root' found in {meta_path}")


def group_records_by_video(records: list):
    """Return list of (video_id, list of records) in order of first appearance of video."""
    video_to_records = defaultdict(list)
    video_order = []
    for rec in records:
        vid = get_video_id(rec.get("image", ""))
        if vid is None:
            continue
        if vid not in video_to_records:
            video_order.append(vid)
        video_to_records[vid].append(rec)
    return [(vid, video_to_records[vid]) for vid in video_order]


def parse_scene_graph_from_record(rec: dict):
    """Extract unique object names and (subj, pred, obj) relation triples from gpt value."""
    objects = set()
    relations = set()
    for conv in rec.get("conversations", []):
        if conv.get("from") != "gpt":
            continue
        text = conv.get("value", "")
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


def _preds_from_triples(rels: set):
    return {p for (_s, p, _o) in rels}


def aggregate_scene_graph_sets(records: list):
    all_objs = set()
    all_rels = set()
    for rec in records:
        objs, rels = parse_scene_graph_from_record(rec)
        all_objs |= objs
        all_rels |= rels
    return all_objs, all_rels


def process_split(
    name: str,
    jsonl_path: str,
    meta_path: str,
    output_dir: str,
    cfg: dict,
):
    """Process one split: group by video, run keyframe selection per video, write _focus_graph.jsonl; return stats dict."""
    meta = load_meta(meta_path)
    root = meta["root"].rstrip("/")
    records = load_jsonl(jsonl_path)
    video_groups = group_records_by_video(records)

    total_before = len(records)
    objs_before_set, rels_before_set = aggregate_scene_graph_sets(records)
    pred_before_set = _preds_from_triples(rels_before_set)
    objs_before = len(objs_before_set)
    rels_before = len(pred_before_set)

    out_records = []
    dummy = Path(os.getcwd())
    patching = cfg["patching"]
    shi = cfg["shi_tomasi"]
    lk = cfg["lucas_kanade"]
    selection = cfg["selection"]
    preprocess = cfg.get("preprocess", {})
    visualization = cfg.get("visualization", {})
    if not visualization:
        visualization = {"enabled": False}

    from tqdm import tqdm
    for video_id, vrecs in tqdm(video_groups, desc=f"Focus-graph {name}", leave=True):
        frame_paths = [Path(root) / r["image"] for r in vrecs]
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
            if os.path.basename(r["image"]) in keyframe_names:
                out_records.append(r)

    total_after = len(out_records)
    objs_after_set, rels_after_set = aggregate_scene_graph_sets(out_records)
    pred_after_set = _preds_from_triples(rels_after_set)
    objs_after = len(objs_after_set)
    rels_after = len(pred_after_set)

    base = os.path.splitext(os.path.basename(jsonl_path))[0]
    if base.endswith(".jsonl"):
        base = os.path.splitext(base)[0]
    out_path = os.path.join(output_dir, f"{base}_focus_graph.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pct = (100.0 * total_after / total_before) if total_before else 0.0
    return {
        "name": name,
        "total_before": total_before,
        "total_after": total_after,
        "pct_remaining": pct,
        "objs_before": objs_before,
        "objs_after": objs_after,
        "rels_before": rels_before,
        "rels_after": rels_after,
        "objs_before_set": objs_before_set,
        "objs_after_set": objs_after_set,
        "pred_before_set": pred_before_set,
        "pred_after_set": pred_after_set,
        "output_path": out_path,
    }


def stats_from_existing_files(name: str, jsonl_path: str, focus_path: str):
    """Load original and _focus_graph jsonl, compute stats. Returns None if focus_path missing."""
    if not os.path.isfile(focus_path):
        return None
    records_before = load_jsonl(jsonl_path)
    records_after = load_jsonl(focus_path)
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
        "output_path": focus_path,
    }


def print_stats(stats_list: list):
    print("\n" + "=" * 60)
    print("Key-frame selection (focus_graph) statistics")
    print("=" * 60)
    for s in stats_list:
        print(f"\n[{s['name']}]")
        print(f"  Frames:  {s['total_before']} -> {s['total_after']}  ({s['pct_remaining']:.1f}% remaining)")
        print(f"  Unique object categories:  {s['objs_before']} -> {s['objs_after']}")
        print(f"  Unique relation types (predicates):  {s['rels_before']} -> {s['rels_after']}")
        print(f"  Output: {s['output_path']}")

    by_name = {s["name"]: s for s in stats_list}
    if "train" in by_name and "test" in by_name:
        tr, te = by_name["train"], by_name["test"]
        obj_overlap_before = len(tr["objs_before_set"] & te["objs_before_set"])
        obj_overlap_after = len(tr["objs_after_set"] & te["objs_after_set"])
        pred_overlap_before = len(tr["pred_before_set"] & te["pred_before_set"])
        pred_overlap_after = len(tr["pred_after_set"] & te["pred_after_set"])
        print("\n[Train–Test overlap]")
        print(f"  Object categories in both train & test:  {obj_overlap_before} (before) -> {obj_overlap_after} (after)")
        print(f"  Relation types (predicates) in both train & test:  {pred_overlap_before} (before) -> {pred_overlap_after} (after)")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="PVSG key-frame selection (Shi–Tomasi + Lucas–Kanade) per video, output *_focus_graph.jsonl."
    )
    parser.add_argument("--train-jsonl", required=True, help="Path to train jsonl")
    parser.add_argument("--train-meta", required=True, help="Path to train meta json (for root)")
    parser.add_argument("--test-jsonl", default="", help="Path to test jsonl (optional)")
    parser.add_argument("--test-meta", default="", help="Path to test meta json (optional)")
    parser.add_argument("--output-dir", required=True, help="Directory for *_focus_graph.jsonl outputs")
    parser.add_argument("--config", default=None, help="Path to config JSON (default: key-frame-selection/config_pvsg.json)")
    args = parser.parse_args()

    if bool(args.test_jsonl) != bool(args.test_meta):
        print("Error: --test-jsonl and --test-meta must be both set or both omitted.", file=sys.stderr)
        sys.exit(1)

    config_path = args.config or (BASE_DIR / "config_pvsg.json")
    if not os.path.isfile(config_path):
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    def _focus_path(jsonl_path: str) -> str:
        base = os.path.splitext(os.path.basename(jsonl_path))[0]
        if base.endswith(".jsonl"):
            base = os.path.splitext(base)[0]
        return os.path.join(args.output_dir, f"{base}_focus_graph.jsonl")

    train_focus_path = _focus_path(args.train_jsonl)
    need_run = not os.path.isfile(train_focus_path)
    if args.test_jsonl and args.test_meta:
        test_focus_path = _focus_path(args.test_jsonl)
        need_run = need_run or not os.path.isfile(test_focus_path)
    else:
        test_focus_path = None

    if not need_run:
        print("Output _focus_graph files already exist. Computing metrics only.")
        stats_list = []
        s_train = stats_from_existing_files("train", args.train_jsonl, train_focus_path)
        if s_train is None:
            print("Error: train focus_graph file missing.", file=sys.stderr)
            sys.exit(1)
        stats_list.append(s_train)
        if args.test_jsonl and args.test_meta:
            s_test = stats_from_existing_files("test", args.test_jsonl, test_focus_path)
            if s_test is None:
                print("Error: test focus_graph file missing.", file=sys.stderr)
                sys.exit(1)
            stats_list.append(s_test)
        print_stats(stats_list)
        return

    stats_list = []
    stats_list.append(
        process_split("train", args.train_jsonl, args.train_meta, args.output_dir, cfg)
    )
    if args.test_jsonl and args.test_meta:
        stats_list.append(
            process_split("test", args.test_jsonl, args.test_meta, args.output_dir, cfg)
        )
    print_stats(stats_list)


if __name__ == "__main__":
    main()
