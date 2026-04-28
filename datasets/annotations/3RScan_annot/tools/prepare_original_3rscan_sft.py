#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3RScan ==> TOON SFT JSON + resized frames (640 x 480).

Reads per-frame GT scene graphs from
``datasets/annotations/3RScan_annot/annotations/{train,test}_annotations.json``
and, for each frame, locates the source RGB image inside a previously extracted
3RScan ``sequence.zip`` (default: ``datasets/frames/3RScan_frames/<scan_id>/frame-XXXXXX.color.jpg``).

Each frame is resized to 640 x 480 and saved under
``datasets/frames/3RScan_frames/{train_images,test_images}/<scan_id>/frame-XXXXXX.jpg``.
The ``image_path`` in the output JSON is **relative to the SceneGraphVLM repository root**.

The TOON answer is the standard 3RScan-extended format with per-object color and
material (each missing value is rendered as ``none``)::

    obj[N]{id,name,color,material,x1,y1,x2,y2}:
      id,name,color,material,x1,y1,x2,y2
      ...
    rel[M]{subj,pred,obj}:
      subj,pred,obj
      ...

Bounding boxes in the source JSON are normalized [0..1] (over original 3RScan
resolution); we scale them directly to (640, 480) integer pixels.

**Run** (from the SceneGraphVLM repository root):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/3RScan_annot/tools/prepare_original_3rscan_sft.py

Optional flags: ``--repo_root``, ``--frames_root``, ``--annotation_dir``,
``--export_root``, ``--images_out``, ``--splits`` (train,test), ``--limit``,
``--num_workers``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

# <repo>/datasets/annotations/3RScan_annot/tools/<this>.py
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]

ANNOTATION_DIR_REL = "datasets/annotations/3RScan_annot/annotations"
FRAMES_ROOT_REL = "datasets/frames/3RScan_frames"
EXPORT_ROOT_REL = "datasets/annotations/3RScan_annot/data_sft_original"
IMAGES_OUT_REL = "datasets/frames/3RScan_frames"

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# Frames inside an extracted 3RScan sequence.zip are named "frame-XXXXXX.color.jpg".
SOURCE_FRAME_SUFFIX = ".color.jpg"


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


def split_images_subdir(split_name: str) -> str:
    if split_name == "train":
        return "train_images"
    if split_name == "test":
        return "test_images"
    return f"{split_name}_images"


def normalize_predicate(pred: str) -> str:
    """``standing on`` -> ``standing-on`` (matches PVSG/PSG TOON convention)."""
    pred = (pred or "").strip()
    pred = " ".join(pred.split())
    return pred.replace(" ", "-")


def _sanitize_token(value: str) -> str:
    """
    Sanitize a single TOON column value: replace separators and whitespace so
    the line never contains stray commas / newlines / leading-trailing spaces.
    """
    value = (value or "").strip()
    if not value:
        return "none"
    value = value.replace(",", ";")
    value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    value = " ".join(value.split())
    value = value.replace(" ", "_")
    return value


def first_attr(attrs: Dict[str, List[str]], key: str) -> str:
    """Pick the first value for an attribute key (empty -> 'none')."""
    if not isinstance(attrs, dict):
        return "none"
    seq = attrs.get(key)
    if not seq or not isinstance(seq, list):
        return "none"
    for item in seq:
        token = _sanitize_token(str(item))
        if token != "none":
            return token
    return "none"


def parse_bbox_xyxy(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        seq = bbox.get("xyxy")
    else:
        seq = bbox
    if not isinstance(seq, (list, tuple)) or len(seq) < 4:
        return None
    try:
        x1, y1, x2, y2 = float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def to_pixels(coord: float, dim: int) -> int:
    """Map a normalized [0..1] coord to integer pixel in [0..dim-1]."""
    if coord < 0.0:
        coord = 0.0
    elif coord > 1.0:
        coord = 1.0
    px = int(round(coord * dim))
    if px < 0:
        return 0
    if px > dim:
        return dim
    return px


def render_toon(
    objects: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"obj[{len(objects)}]{{id,name,color,material,x1,y1,x2,y2}}:")
    for obj in objects:
        lines.append(
            "  {oid},{name},{color},{material},{x1},{y1},{x2},{y2}".format(
                oid=obj["obj_id"],
                name=_sanitize_token(obj["name"]),
                color=obj.get("color") or "none",
                material=obj.get("material") or "none",
                x1=obj["bbox"][0],
                y1=obj["bbox"][1],
                x2=obj["bbox"][2],
                y2=obj["bbox"][3],
            )
        )
    lines.append(f"rel[{len(relationships)}]{{subj,pred,obj}}:")
    for rel in relationships:
        lines.append(
            "  {s},{p},{o}".format(
                s=rel["sub_id"],
                p=normalize_predicate(rel["pred"]),
                o=rel["obj_id"],
            )
        )
    return "\n".join(lines)


def split_image_id(image_id: str) -> Tuple[str, str]:
    """``<scan_id>_frame-XXXXXX`` -> (scan_id, 'frame-XXXXXX')."""
    sep = "_frame-"
    idx = image_id.rfind(sep)
    if idx == -1:
        raise ValueError(f"Unexpected image_id format: {image_id!r}")
    return image_id[:idx], image_id[idx + 1 :]


def locate_source_frame(frames_root: Path, scan_id: str, frame_stem: str) -> Optional[Path]:
    """Try a couple of common 3RScan layouts for one frame."""
    candidates = [
        frames_root / scan_id / f"{frame_stem}{SOURCE_FRAME_SUFFIX}",
        frames_root / scan_id / "sequence" / f"{frame_stem}{SOURCE_FRAME_SUFFIX}",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def process_one_sample(task: Tuple[Dict[str, Any], str, str, str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Worker entry: returns (record_or_None, status_tag).
    status_tag in {'ok', 'no_frame', 'no_objects', 'image_error'}.
    """
    item, frames_root_s, images_out_s, repo_root_s = task
    frames_root = Path(frames_root_s)
    images_out_root = Path(images_out_s)
    repo_root = Path(repo_root_s)

    image_id = item.get("image_id") or ""
    try:
        scan_id, frame_stem = split_image_id(image_id)
    except ValueError:
        return None, "no_frame"

    src = locate_source_frame(frames_root, scan_id, frame_stem)
    if src is None:
        return None, "no_frame"

    nodes = item.get("nodes") or []
    if not nodes:
        return None, "no_objects"

    # Collect objects with valid bbox; renumber to local 1..N (preserve order).
    local_objects: List[Dict[str, Any]] = []
    src_to_local: Dict[int, int] = {}

    for node in nodes:
        try:
            src_id = int(node.get("id"))
        except (TypeError, ValueError):
            continue
        data = node.get("data") or {}
        bbox = parse_bbox_xyxy(data.get("bbox_2d"))
        if bbox is None:
            continue
        x1n, y1n, x2n, y2n = bbox
        if x2n <= x1n or y2n <= y1n:
            continue

        attrs = data.get("attributes") or {}
        name = data.get("class_name") or ""
        color = first_attr(attrs, "color")
        material = first_attr(attrs, "material")

        local_id = len(local_objects) + 1
        src_to_local[src_id] = local_id

        x1 = to_pixels(x1n, TARGET_WIDTH)
        y1 = to_pixels(y1n, TARGET_HEIGHT)
        x2 = to_pixels(x2n, TARGET_WIDTH)
        y2 = to_pixels(y2n, TARGET_HEIGHT)
        if x2 <= x1:
            x2 = min(TARGET_WIDTH, x1 + 1)
        if y2 <= y1:
            y2 = min(TARGET_HEIGHT, y1 + 1)

        local_objects.append(
            {
                "obj_id": local_id,
                "name": name,
                "color": color,
                "material": material,
                "bbox": [x1, y1, x2, y2],
            }
        )

    if not local_objects:
        return None, "no_objects"

    # Relationships -> local ids.
    rels: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for link in item.get("links") or []:
        try:
            src_a = int(link.get("source"))
            src_b = int(link.get("target"))
        except (TypeError, ValueError):
            continue
        pred = link.get("label")
        if not pred:
            continue
        la = src_to_local.get(src_a)
        lb = src_to_local.get(src_b)
        if la is None or lb is None or la == lb:
            continue
        key = (la, lb, pred)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rels.append({"sub_id": la, "pred": pred, "obj_id": lb})

    # Resize + save image.
    out_dir = images_out_root / scan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{frame_stem}.jpg"

    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
                img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)
            img.save(out_path, format="JPEG", quality=92)
    except Exception:
        return None, "image_error"

    toon = render_toon(local_objects, rels)
    return (
        {
            "image_id": image_id,
            "image_path": path_for_json(repo_root, out_path),
            "answer_toon": toon,
        },
        "ok",
    )


def process_split(
    split_name: str,
    annotation_path: Path,
    frames_root: Path,
    images_out_root: Path,
    export_root: Path,
    repo_root: Path,
    limit: int,
    num_workers: int,
) -> None:
    if not annotation_path.is_file():
        print(f"[warn] {split_name}: annotation file not found: {annotation_path}", file=sys.stderr)
        return

    print(f"[{split_name}] loading annotations: {annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as fh:
        items: List[Dict[str, Any]] = json.load(fh)

    if limit > 0:
        items = items[:limit]

    images_out_dir = images_out_root / split_images_subdir(split_name)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (item, str(frames_root), str(images_out_dir), str(repo_root))
        for item in items
    ]

    out_records: List[Dict[str, Any]] = []
    counters = {"ok": 0, "no_frame": 0, "no_objects": 0, "image_error": 0}

    if num_workers > 1 and len(tasks) > 1:
        with Pool(processes=min(num_workers, len(tasks))) as pool:
            for record, status in tqdm(
                pool.imap_unordered(process_one_sample, tasks, chunksize=8),
                total=len(tasks),
                desc=f"frames ({split_name})",
                ncols=80,
            ):
                counters[status] = counters.get(status, 0) + 1
                if record is not None:
                    out_records.append(record)
    else:
        for task in tqdm(tasks, desc=f"frames ({split_name})", ncols=80):
            record, status = process_one_sample(task)
            counters[status] = counters.get(status, 0) + 1
            if record is not None:
                out_records.append(record)

    # Stable order for deterministic JSON / training shards.
    out_records.sort(key=lambda x: x["image_id"])

    export_root.mkdir(parents=True, exist_ok=True)
    out_json = export_root / f"{split_name}_annotations_toon_sft.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(out_records, fh, ensure_ascii=False, indent=2)

    print(
        f"[{split_name}] done: ok={counters['ok']}, "
        f"no_frame={counters['no_frame']}, "
        f"no_objects={counters['no_objects']}, "
        f"image_error={counters['image_error']} "
        f"-> {path_for_json(repo_root, out_json)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3RScan ==> TOON SFT JSON + 640x480 frames (image_path is repo-relative).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_root", default=str(_REPO_ROOT_DEFAULT))
    parser.add_argument("--annotation_dir", default=ANNOTATION_DIR_REL)
    parser.add_argument(
        "--frames_root",
        default=FRAMES_ROOT_REL,
        help="Root of extracted 3RScan sequences (one folder per scan_id).",
    )
    parser.add_argument(
        "--export_root",
        default=EXPORT_ROOT_REL,
        help="Output directory for {train,test}_annotations_toon_sft.json.",
    )
    parser.add_argument(
        "--images_out",
        default=IMAGES_OUT_REL,
        help="Output root for resized 640x480 frames (train_images/, test_images/).",
    )
    parser.add_argument("--splits", default="train,test", help="Comma-separated list of splits to process.")
    parser.add_argument("--limit", type=int, default=0, help="Cap items per split (0 = all).")
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    annotation_dir = resolve_under_repo(repo_root, args.annotation_dir)
    frames_root = resolve_under_repo(repo_root, args.frames_root)
    export_root = resolve_under_repo(repo_root, args.export_root)
    images_out_root = resolve_under_repo(repo_root, args.images_out)

    if not frames_root.is_dir():
        print(f"Error: frames_root does not exist: {frames_root}", file=sys.stderr)
        print("Did you run download_3RScan.py first?", file=sys.stderr)
        sys.exit(1)

    print(f"Repo root:       {repo_root}")
    print(f"Annotation dir:  {annotation_dir}")
    print(f"Frames root:     {frames_root}")
    print(f"Export (JSON):   {export_root}")
    print(f"Images out:      {images_out_root}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split_name in splits:
        if split_name not in {"train", "test"}:
            print(f"[warn] skipping unknown split '{split_name}'", file=sys.stderr)
            continue
        annotation_path = annotation_dir / f"{split_name}_annotations.json"
        process_split(
            split_name=split_name,
            annotation_path=annotation_path,
            frames_root=frames_root,
            images_out_root=images_out_root,
            export_root=export_root,
            repo_root=repo_root,
            limit=args.limit,
            num_workers=args.num_workers,
        )

    print("All done.")


if __name__ == "__main__":
    main()
