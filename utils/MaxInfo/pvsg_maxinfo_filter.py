#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaxInfo key-frame selection for PVSG TOON JSON (CLIP -> SVD(r) -> rect_maxvol).

Reads ``train_annotations_toon_sft.json`` and ``test_annotations_toon_sft.json`` from ``--input_dir``
(same schema as ``prepare_original_pvsg_sft.py`` / ``prepare_filtered_pvsg_sft.py``). For each video,
embeds frames with CLIP, reduces with SVD, selects diverse frames via ``rect_maxvol``, and writes
the filtered arrays to ``--output_dir`` under the **same filenames**. ``image_path`` in the output
stays **relative to the repository root** (POSIX ``/``).

**Run** (repository root = SceneGraphVLM; GPU recommended for CLIP):

  cd /path/to/SceneGraphVLM
  python utils/MaxInfo/pvsg_maxinfo_filter.py --fp16

  python utils/MaxInfo/pvsg_maxinfo_filter.py \\
    --input_dir datasets/annotations/PVSG_annot/data_sft_original \\
    --output_dir datasets/annotations/PVSG_annot/data_sft_maxinfo \\
    --repo_root /path/to/SceneGraphVLM \\
    --r 8 --tol 0.23 --fp16 --batch-size 64

If output JSON files already exist, the script skips CLIP and only prints statistics.

**maxvol**: SVD of features then ``rect_maxvol(M, tol)`` (maxvolpy). Here **larger ``tol`` keeps
fewer frames** (looser coefficient bound → smaller pivot set); **smaller ``tol`` keeps more**.
Empirically on full PVSG train JSON (same CLIP path): ``tol≈0.10`` ~76% retained,
``tol≈0.42`` ~11.6% retained, ``tol≈1.0`` ~3% retained. Default ``--tol 0.23`` is calibrated to sit
close to ``datasets/data_playground/PVSG_json/pvsg_maxinfo_gt_prompt`` (~48.4k train / ~9.1k test
lines, ~38–40% of pre-filter rows on ``data_sft_base_annot``). If you retain too **few** frames,
**decrease** ``tol``; too **many** → **increase** ``tol``. maxvolpy’s default ``tol=1.0`` is far too
aggressive here. CLIP:
``openai/clip-vit-large-patch14-336``; pooler output for image features. If the number of selected
frames is odd and >1, the last index is dropped (even count).

Author alignment notes: see legacy notebooks in this folder; differences are handling of current
``transformers`` CLIP outputs and processing all per-video frames from JSON (no uniform 64/128
presampling).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

try:
    from maxvolpy.maxvol import rect_maxvol
except ImportError:
    from maxvolpy.maxvolpy.maxvol import rect_maxvol

# utils/MaxInfo/<this>.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = "datasets/annotations/PVSG_annot/data_sft_base_annot"
DEFAULT_OUTPUT_DIR = "datasets/annotations/PVSG_annot/data_sft_maxinfo"
DEFAULT_TRAIN_JSON = "train_annotations_toon_sft.json"
DEFAULT_TEST_JSON = "test_annotations_toon_sft.json"

# video id from path: .../frames/<video_id>/<frame>.png
VIDEO_ID_PATTERN = re.compile(r"frames/([^/]+)/\d+\.(?:png|jpg|jpeg)", re.IGNORECASE)


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


def normalize_image_path_repo_relative(image_path: str, repo_root: Path) -> str:
    p = Path(image_path)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p.resolve()).replace("\\", "/")
    return str(Path(image_path).as_posix())


def group_records_by_video(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """(video_id, records) in order of first appearance."""
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


def svd_reduce_dim(features: np.ndarray, r: int) -> np.ndarray:
    """features: (n_frames, dim) -> (n_frames, r_use)."""
    n, dim = features.shape
    if n < 2:
        return features[:, : min(r, dim)] if dim else features
    r_use = min(r, dim, n - 1)
    if r_use < 1:
        return features[:, :1]
    u, _s, _v = np.linalg.svd(features.astype(np.float64), full_matrices=False)
    return u[:, :r_use].astype(np.float64)


def maxvol_select_indices(M: np.ndarray, tol: float) -> np.ndarray:
    n, r = M.shape
    if n <= r:
        return np.arange(n)
    piv, _ = rect_maxvol(M, tol=tol)
    return np.array(piv, dtype=np.int64)


def select_key_frames_for_video(
    records: list[dict[str, Any]],
    image_paths: list[str],
    vision_model: Any,
    vision_processor: Any,
    device: torch.device,
    r: int,
    tol: float,
    batch_size: int = 32,
    use_fp16: bool = False,
) -> list[dict[str, Any]]:
    n = len(records)
    if n == 0:
        return []
    if n == 1:
        return list(records)

    all_feats = []
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (336, 336), (0, 0, 0)))
        with torch.no_grad():
            inputs = vision_processor(images=images, return_tensors="pt", padding=True).to(device)
            if use_fp16 and inputs["pixel_values"].dtype == torch.float32:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            out = vision_model.get_image_features(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
        all_feats.append(feats.float().cpu().numpy())
    features = np.concatenate(all_feats, axis=0).astype(np.float64)

    r_actual = min(r, features.shape[1], features.shape[0] - 1)
    if r_actual < 1:
        return list(records)
    M = svd_reduce_dim(features, r_actual)
    if M.shape[0] < M.shape[1]:
        return list(records)
    indices = maxvol_select_indices(M, tol)
    indices = np.sort(indices)
    if len(indices) > 1 and len(indices) % 2 != 0:
        indices = indices[:-1]
    return [records[int(i)] for i in indices]


def parse_scene_graph_from_toon(text: str) -> tuple[set[str], set[tuple[str, str, str]]]:
    """Parse TOON text for object names and (subj, pred, obj) triples."""
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
    """Support TOON JSON (answer_toon) or legacy jsonl (conversations gpt)."""
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
    vision_model: Any,
    vision_processor: Any,
    device: torch.device,
    r: int,
    tol: float,
    batch_size: int,
    use_fp16: bool,
) -> dict[str, Any]:
    records = load_toon_json(input_json)
    # Normalize paths for grouping / disk
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

    out_records: list[dict[str, Any]] = []
    for _video_id, vrecs in tqdm(video_groups, desc=f"MaxInfo {name}", leave=True):
        rel_paths = [r.get("_rel_path") or r.get("image_path") or r.get("image", "") for r in vrecs]
        image_paths = [str(resolve_under_repo(repo_root, rp)) for rp in rel_paths]
        selected = select_key_frames_for_video(
            vrecs,
            image_paths,
            vision_model,
            vision_processor,
            device,
            r,
            tol,
            batch_size,
            use_fp16,
        )
        for s in selected:
            s.pop("_rel_path", None)
        out_records.extend(selected)

    # Repo-relative image_path in output
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
    print("MaxInfo filtering statistics")
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
        obj_overlap_before = len(tr["objs_before_set"] & te["objs_before_set"])
        obj_overlap_after = len(tr["objs_after_set"] & te["objs_after_set"])
        pred_overlap_before = len(tr["pred_before_set"] & te["pred_before_set"])
        pred_overlap_after = len(tr["pred_after_set"] & te["pred_after_set"])
        print("\n[Train–Test overlap]")
        print(
            f"  Object categories in both splits:  {obj_overlap_before} (before) -> {obj_overlap_after} (after)"
        )
        print(
            f"  Relation types in both splits:  {pred_overlap_before} (before) -> {pred_overlap_after} (after)"
        )
    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MaxInfo key-frame filter: TOON JSON folder -> data_sft_maxinfo (same filenames).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo_root",
        default=str(_REPO_ROOT_DEFAULT),
        help="SceneGraphVLM repo root (resolves paths and image_path)",
    )
    p.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help="Folder with train/test TOON JSON",
    )
    p.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder (same JSON basenames as input)",
    )
    p.add_argument("--train-json", default=DEFAULT_TRAIN_JSON, help="Train JSON basename")
    p.add_argument("--test-json", default=DEFAULT_TEST_JSON, help="Test JSON basename")
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Process only the train JSON (ignore test split even if present)",
    )
    p.add_argument("--r", type=int, default=8, help="SVD rank")
    p.add_argument(
        "--tol",
        type=float,
        default=0.23,
        help="rect_maxvol tol: smaller → more frames kept, larger → fewer (see module docstring).",
    )
    p.add_argument("--batch-size", type=int, default=None, help="CLIP batch size (default: 64 GPU / 32 CPU)")
    p.add_argument("--fp16", action="store_true", help="FP16 CLIP on CUDA")
    p.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14-336",
        help="HuggingFace CLIP model id",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    input_dir = resolve_under_repo(repo_root, args.input_dir)
    output_dir = resolve_under_repo(repo_root, args.output_dir)

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
    need_clip = need_train or need_test

    if not need_clip:
        print("Output JSON already exists; metrics only (no CLIP run).")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size if args.batch_size is not None else (64 if device.type == "cuda" else 32)
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and device.type != "cuda":
        print("Note: --fp16 ignored (no CUDA).", file=sys.stderr)

    print(f"repo_root={repo_root}")
    print(f"input_dir={path_for_json(repo_root, input_dir)}")
    print(f"output_dir={path_for_json(repo_root, output_dir)}")
    print(f"Loading CLIP {args.clip_model} on {device} (batch={batch_size}, fp16={use_fp16})")
    vision_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    if use_fp16:
        vision_model = vision_model.half()
    vision_processor = AutoProcessor.from_pretrained(args.clip_model)

    stats_list: list[dict[str, Any]] = []
    if need_train:
        stats_list.append(
            process_split_json(
                "train",
                train_in,
                train_out,
                repo_root,
                vision_model,
                vision_processor,
                device,
                args.r,
                args.tol,
                batch_size,
                use_fp16,
            )
        )
    else:
        st = stats_from_existing_json("train", train_in, train_out)
        if st is not None:
            stats_list.append(st)

    if need_test:
        stats_list.append(
            process_split_json(
                "test",
                test_in,
                test_out,
                repo_root,
                vision_model,
                vision_processor,
                device,
                args.r,
                args.tol,
                batch_size,
                use_fp16,
            )
        )
    elif do_test:
        stt = stats_from_existing_json("test", test_in, test_out)
        if stt is not None:
            stats_list.append(stt)

    print_stats(stats_list)


if __name__ == "__main__":
    main()
