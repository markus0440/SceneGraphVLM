#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build AG SFT intermediates: TOON labels in JSON + 640x480 PNGs for training.

Loads ``annotations/*.pkl`` and reads source frames from ``Charades/frames``. Writes:

- ``datasets/annotations/AG_annot/data_sft_original/{train,test}_annotations_toon_sft.json``
- Resized images under ``datasets/frames/AG_frames/train_images/`` and
  ``datasets/frames/AG_frames/test_images/`` (``<VIDEO>.mp4/<frame>.png``).

Each sample's ``image_path`` is a path **relative to the SceneGraphVLM repo root** (POSIX ``/``).
Run ``sft_to_jsonl_ag.py`` after this to add chat prompts and emit ``*.jsonl``.

**Commands** (``cd`` to the SceneGraphVLM repository root; path flags are repo-relative unless absolute):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/AG_annot/tools/prepare_original_ag_sft.py
  python datasets/annotations/AG_annot/tools/sft_to_jsonl_ag.py

Optional: ``--limit N``, ``--num_workers K``, ``--keep_all``; ``--repo_root <path>`` if auto-detect fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

# This file: <repo>/datasets/annotations/AG_annot/tools/<script>.py
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]

# Default CLI values: relative to repo root
AG_ROOT_REL = "datasets/annotations/AG_annot"
FRAMES_ROOT_REL = "datasets/annotations/AG_annot/Charades/frames"
EXPORT_ROOT_REL = "datasets/annotations/AG_annot/data_sft_original"
IMAGES_OUT_REL = "datasets/frames/AG_frames"


def resolve_under_repo(repo_root: Path, path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / path_arg).resolve()


def path_for_json(repo_root: Path, file_path: Path) -> str:
    """Path relative to repo root, forward slashes (for JSON)."""
    try:
        return str(file_path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(file_path.resolve()).replace("\\", "/")

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# Worker globals for multiprocessing (fork on Linux; read-only pickles)
_G_OBJ: Dict[str, Any] = {}
_G_PER: Dict[str, Any] = {}


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        try:
            return __import__("pickle").load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return __import__("pickle").load(f, encoding="latin1")


def strip_mp4(video_id: str) -> str:
    vid = str(video_id)
    return vid[:-4] if vid.endswith(".mp4") else vid


def ensure_frame_filename(frame: str) -> str:
    frame = str(frame)
    base, ext = os.path.splitext(frame)
    if ext.lower() not in {".png", ".jpg", ".jpeg"}:
        ext = ".png"
    if re.fullmatch(r"\d+", base):
        base = base.zfill(6)
    return base + ext


def normalized_frame_stem(frame: str) -> str:
    base = os.path.splitext(str(frame))[0]
    if re.fullmatch(r"\d+", base):
        base = base.zfill(6)
    return base


def rel_ag_frame_path(split: str, video_id: str, frame_file: str) -> str:
    fname = ensure_frame_filename(frame_file)
    vdir = video_id if str(video_id).endswith(".mp4") else f"{video_id}.mp4"
    split_dir = "train_images" if split == "train" else "test_images"
    return f"{split_dir}/{vdir}/{fname}"


def find_image_path(frames_root: Path, video_id: str, frame_file: str) -> Optional[Path]:
    fname = ensure_frame_filename(frame_file)
    vdir = video_id if str(video_id).endswith(".mp4") else f"{video_id}.mp4"
    p = frames_root / vdir / fname
    return p if p.is_file() else None


def xywh_to_xyxy(bbox: Tuple[float, ...]) -> List[int]:
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    return [x1, y1, x2, y2]


def person_to_xyxy(person_entry: Any) -> Optional[List[int]]:
    if person_entry is None:
        return None
    if isinstance(person_entry, dict) and "bbox" in person_entry:
        b = person_entry["bbox"]
        mode = (person_entry.get("bbox_mode") or "xyxy").lower()
        if hasattr(b, "tolist"):
            b = b.tolist()
        if not b:
            return None
        row = b[0] if isinstance(b[0], (list, tuple)) else b
        if len(row) != 4:
            return None
        x1, y1, x2, y2 = (float(v) for v in row)
        if mode == "xywh":
            x2, y2 = x1 + x2, y1 + y2
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    return None


def resolve_person_entry(persons: Dict[str, Any], video_id: str, frame_file: str) -> Any:
    fname = ensure_frame_filename(frame_file)
    stem = os.path.splitext(fname)[0]
    cands = [
        f"{video_id}/{fname}",
        f"{strip_mp4(video_id)}/{fname}",
        f"{video_id}/{stem}",
        f"{strip_mp4(video_id)}/{stem}",
    ]
    for ck in cands:
        if ck in persons:
            return persons[ck]
    return None


def _bracket_list(labels: List[str]) -> str:
    inner = ",".join(labels)
    return f"[{inner}]"


def generate_toon_format_annotation(
    objects: List[Dict[str, Any]], rel_pairs: List[Dict[str, Any]]
) -> str:
    lines: List[str] = []
    lines.append(f"obj[{len(objects)}]{{id,name,x1,y1,x2,y2}}:")
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        name = str(obj["name"]).replace(",", "-")
        lines.append(f"  {obj['obj_id']},{name},{x1},{y1},{x2},{y2}")
    lines.append(
        f"rel_pairs[{len(rel_pairs)}]{{subj,attention,spatial,contacting,obj}}:"
    )
    for row in rel_pairs:
        a = _bracket_list(row["attention"])
        s = _bracket_list(row["spatial"])
        c = _bracket_list(row["contacting"])
        lines.append(f"  {row['sub_id']},{a},{s},{c},{row['obj_id']}")
    return "\n".join(lines)


def scale_bounding_boxes_in_toon(toon: str, scale_x: float, scale_y: float) -> str:
    out: List[str] = []
    for line in toon.split("\n"):
        s = line.strip()
        if (
            s
            and "," in s
            and "[" not in s
            and not s.startswith("obj[")
            and not s.startswith("rel_pairs[")
        ):
            parts = s.split(",")
            if len(parts) == 6:
                try:
                    oid, name, x1, y1, x2, y2 = parts
                    fx1, fy1, fx2, fy2 = map(float, (x1, y1, x2, y2))
                    nx1 = int(round(fx1 * scale_x))
                    ny1 = int(round(fy1 * scale_y))
                    nx2 = int(round(fx2 * scale_x))
                    ny2 = int(round(fy2 * scale_y))
                    ind = line[: len(line) - len(line.lstrip())]
                    out.append(f"{ind}{oid},{name},{nx1},{ny1},{nx2},{ny2}")
                    continue
                except (ValueError, IndexError):
                    pass
        out.append(line)
    return "\n".join(out)


def split_from_items(obj_items: List[Any]) -> str:
    sp = "train"
    if obj_items:
        md = obj_items[0].get("metadata") or {}
        sp = str(md.get("set") or "train").lower()
    if sp == "val":
        sp = "train"
    if sp not in ("train", "test"):
        sp = "train"
    return sp


def _norm_rel_list(raw: Any) -> List[str]:
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        return [str(raw).strip()] if str(raw).strip() else []
    out = [str(x).strip() for x in raw if x is not None and str(x).strip()]
    seen: set = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def build_objects_and_rel_pairs(
    obj_items: List[Any],
    person_entry: Any,
    only_visible: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    objects: List[Dict[str, Any]] = []
    rel_pairs: List[Dict[str, Any]] = []

    pbox = person_to_xyxy(person_entry)
    person_id: Optional[int] = None
    if pbox:
        person_id = 1
        objects.append({"obj_id": person_id, "name": "person", "bbox": pbox})

    obj_indices: List[Optional[int]] = []
    for it in obj_items or []:
        if only_visible and not it.get("visible", True):
            obj_indices.append(None)
            continue
        bb = it.get("bbox")
        if not bb:
            obj_indices.append(None)
            continue
        oid = len(objects) + 1
        cls = it.get("class") or "object"
        objects.append({"obj_id": oid, "name": str(cls), "bbox": xywh_to_xyxy(tuple(bb))})
        obj_indices.append(oid)

    if person_id is None:
        return objects, rel_pairs

    for src, oid in zip(obj_items or [], obj_indices):
        if oid is None:
            continue
        attention = _norm_rel_list(src.get("attention_relationship"))
        spatial = _norm_rel_list(src.get("spatial_relationship"))
        contacting = _norm_rel_list(src.get("contacting_relationship"))
        if not attention and not spatial and not contacting:
            continue
        rel_pairs.append(
            {
                "sub_id": person_id,
                "obj_id": oid,
                "attention": attention,
                "spatial": spatial,
                "contacting": contacting,
            }
        )

    return objects, rel_pairs


def process_key(
    key: str,
    frames_root: Path,
    images_out_root: Path,
    repo_root: Path,
    only_visible: bool,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if "/" not in key:
        return None
    video_id, frame_file = key.split("/", 1)
    obj_items = _G_OBJ.get(key)
    if obj_items is None:
        return None

    src_img = find_image_path(frames_root, video_id, frame_file)
    if src_img is None:
        return None

    person_entry = resolve_person_entry(_G_PER, video_id, frame_file)
    split = split_from_items(obj_items)
    objects, rel_pairs = build_objects_and_rel_pairs(
        obj_items, person_entry, only_visible
    )
    if not objects:
        return None

    try:
        im = Image.open(src_img).convert("RGB")
    except Exception:
        return None

    ow, oh = im.size
    if ow <= 0 or oh <= 0:
        return None

    sx = TARGET_WIDTH / ow
    sy = TARGET_HEIGHT / oh
    toon = generate_toon_format_annotation(objects, rel_pairs)
    if sx != 1.0 or sy != 1.0:
        toon = scale_bounding_boxes_in_toon(toon, sx, sy)

    rel_sub = rel_ag_frame_path(split, video_id, frame_file)
    out_path = images_out_root / rel_sub
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if ow == TARGET_WIDTH and oh == TARGET_HEIGHT:
        shutil.copy2(src_img, out_path)
    else:
        im.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR).save(out_path)

    stem = normalized_frame_stem(frame_file)
    sample = {
        "image_id": f"{strip_mp4(video_id)}_{stem}",
        "image_path": path_for_json(repo_root, out_path),
        "answer_toon": toon,
    }
    return split, sample


def _worker_key(key: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    return process_key(key, _FRAMES, _IMAGES_OUT, _REPO, _ONLY_VISIBLE)


_FRAMES = Path()
_IMAGES_OUT = Path()
_REPO = Path()
_ONLY_VISIBLE = True


def _init_worker_globals(
    obj: Dict[str, Any],
    per: Dict[str, Any],
    frames: Path,
    images_out: Path,
    repo_root: Path,
    only_visible: bool,
) -> None:
    global _G_OBJ, _G_PER, _FRAMES, _IMAGES_OUT, _REPO, _ONLY_VISIBLE
    _G_OBJ = obj
    _G_PER = per
    _FRAMES = frames
    _IMAGES_OUT = images_out
    _REPO = repo_root
    _ONLY_VISIBLE = only_visible


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Action Genome ==> train/test JSON with TOON answers; frames on disk"
    )
    ap.add_argument(
        "--repo_root",
        default="",
        help="SceneGraphVLM repo root (empty ==> infer from this script location).",
    )
    ap.add_argument(
        "--ag_root",
        default=AG_ROOT_REL,
        help="AG bundle root, relative to repo unless absolute (default: %(default)s).",
    )
    ap.add_argument(
        "--frames_root",
        default=FRAMES_ROOT_REL,
        help="Source Charades frames root (default: %(default)s).",
    )
    ap.add_argument(
        "--export_root",
        default=EXPORT_ROOT_REL,
        help="Output directory for JSON only (default: %(default)s).",
    )
    ap.add_argument(
        "--images_out",
        default=IMAGES_OUT_REL,
        help="Output root for PNGs; uses train_images/ and test_images/ (default: %(default)s).",
    )
    ap.add_argument(
        "--keep_all",
        action="store_true",
        help="Do not filter objects by visible (default: only visible=True)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max frames (0 = all)")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="1 = sequential; >1 = multiprocessing pool",
    )
    args = ap.parse_args()

    only_visible = not args.keep_all

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _REPO_ROOT_DEFAULT
    ag_root = resolve_under_repo(repo_root, args.ag_root)
    ann = ag_root / "annotations"
    pkl_obj = ann / "object_bbox_and_relationship.pkl"
    pkl_per = ann / "person_bbox.pkl"
    frames_root = resolve_under_repo(repo_root, args.frames_root)
    export_root = resolve_under_repo(repo_root, args.export_root)
    images_out_root = resolve_under_repo(repo_root, args.images_out)

    for p, label in ((pkl_obj, "object_bbox_and_relationship.pkl"), (pkl_per, "person_bbox.pkl")):
        if not p.is_file():
            print(f"Error: missing file {p}", file=sys.stderr)
            sys.exit(1)
    if not frames_root.is_dir():
        print(f"Error: frames directory not found: {frames_root}", file=sys.stderr)
        sys.exit(1)

    export_root.mkdir(parents=True, exist_ok=True)
    images_out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pkl_obj} ...")
    obj_rel = load_pickle(pkl_obj)
    print(f"Loading {pkl_per} ...")
    persons = load_pickle(pkl_per)

    keys = list(obj_rel.keys())
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    train_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []
    skipped = 0

    if args.num_workers <= 1:
        global _G_OBJ, _G_PER
        _G_OBJ = obj_rel
        _G_PER = persons
        for key in tqdm(keys, desc="AG frames", unit="fr"):
            r = process_key(key, frames_root, images_out_root, repo_root, only_visible)
            if r is None:
                skipped += 1
                continue
            sp, sample = r
            if sp == "train":
                train_samples.append(sample)
            else:
                test_samples.append(sample)
    else:
        nw = min(args.num_workers, len(keys), cpu_count())
        with Pool(
            processes=nw,
            initializer=_init_worker_globals,
            initargs=(obj_rel, persons, frames_root, images_out_root, repo_root, only_visible),
        ) as pool:
            for r in tqdm(
                pool.imap_unordered(_worker_key, keys, chunksize=32),
                total=len(keys),
                desc="AG frames",
                unit="fr",
            ):
                if r is None:
                    skipped += 1
                    continue
                sp, sample = r
                if sp == "train":
                    train_samples.append(sample)
                else:
                    test_samples.append(sample)

    train_path = export_root / "train_annotations_toon_sft.json"
    test_path = export_root / "test_annotations_toon_sft.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    print(f"Train samples: {len(train_samples):,} ==> {path_for_json(repo_root, train_path)}")
    print(f"Test samples:  {len(test_samples):,} ==> {path_for_json(repo_root, test_path)}")
    print(f"Skipped:       {skipped:,}")
    print(f"JSON (rel repo):   {path_for_json(repo_root, export_root)}/")
    print(f"Images (rel repo): {path_for_json(repo_root, images_out_root)}/")


if __name__ == "__main__":
    main()
