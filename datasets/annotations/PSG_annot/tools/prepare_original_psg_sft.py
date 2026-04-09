#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare PSG for SFT: annotations on disk, pixels from HuggingFace (no raw image folders).

**Annotations** live under ``datasets/annotations/PSG_annot/annotations/``:

  - ``train_annotations.json``, ``test_annotations.json`` (from ``--from-hf`` or your own files)

HF exports omit ``image_path`` and set ``"image_source": "huggingface"``. Resized 640x480
images are written only to ``datasets/frames/PSG_frames/{train_images,test_images}/``.

**SFT JSON only** in ``datasets/annotations/PSG_annot/data_sft_original/``:

  - ``train_annotations_toon_sft.json``, ``test_annotations_toon_sft.json``

Each ``image_path`` in those JSON files is repo-relative. While building SFT, images are
read from HuggingFace (default) unless a local ``image_path`` file exists.

**Typical first-time setup** (downloads annotations JSON into ``annotations/``, then SFT + frames):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/PSG_annot/tools/prepare_original_psg_sft.py --from-hf
  python datasets/annotations/PSG_annot/tools/sft_to_jsonl_psg.py

**If JSON already exists** (rebuild resized frames + SFT; pulls pixels from HF again):

  python datasets/annotations/PSG_annot/tools/prepare_original_psg_sft.py
  # Default: load pixels from HuggingFace. Use --no-hf-images only if every row has a valid local image_path.

Optional: ``--skip-sft``, ``--repo_root``, ``--no-hf-images`` (offline / local images only).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, desc=None, total=None, **kwargs):
        return x

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]

ANNOTATIONS_DIR_REL = "datasets/annotations/PSG_annot/annotations"
EXPORT_ROOT_REL = "datasets/annotations/PSG_annot/data_sft_original"
IMAGES_OUT_REL = "datasets/frames/PSG_frames"

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
SPLITS = ("train", "test")

HF_DATASET_NAMES = {
    "train": "JosephZ/psg_train_sg",
    "test": "JosephZ/psg_test_sg",
}


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


def _get_hf_split_dataset(split: str):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install datasets: pip install datasets") from e

    name = HF_DATASET_NAMES.get(split)
    if not name:
        raise ValueError(f"Unknown split: {split}")

    print(f"[HF] Opening {name} (Hub resolve + mmap; first run can be slow)...")
    dataset_result = load_dataset(name)
    if hasattr(dataset_result, "keys") and hasattr(dataset_result, "__getitem__"):
        keys = list(dataset_result.keys())
        if not keys:
            raise ValueError(f"No splits in {name}")
        split_use = keys[0]
        if len(keys) > 1:
            print(f"[HF] DatasetDict splits {keys}, using '{split_use}'")
        return dataset_result[split_use]
    return dataset_result


def load_from_huggingface(split: str) -> List[Dict[str, Any]]:
    dataset = _get_hf_split_dataset(split)
    annotations: List[Dict[str, Any]] = []

    for item in tqdm(dataset, desc=f"HF export {split}"):
        ann: Dict[str, Any] = {
            "image_id": str(item["image_id"]),
            "image_source": "huggingface",
            "objects": json.loads(item["objects"])
            if isinstance(item["objects"], str)
            else item["objects"],
            "relationships": json.loads(item["relationships"])
            if isinstance(item["relationships"], str)
            else item["relationships"],
        }
        annotations.append(ann)

    print(f"[HF] Prepared {len(annotations)} annotation rows for {split} (no raw images on disk)")
    return annotations


def build_hf_image_loader(split: str) -> Callable[[str], Image.Image]:
    dataset = _get_hf_split_dataset(split)
    id_to_idx: Dict[str, int] = {}
    try:
        index_total = len(dataset)
    except Exception:
        index_total = None
    print(
        "[HF] Indexing image_id -> row (reads row metadata only; "
        "then each image is decoded during resize)..."
    )
    for i, row in tqdm(
        enumerate(dataset),
        total=index_total,
        desc=f"HF index {split}",
        unit="row",
    ):
        id_to_idx[str(row["image_id"])] = i

    def load_rgb(image_id: str) -> Image.Image:
        idx = id_to_idx[str(image_id)]
        row = dataset[idx]
        im = row["image"]
        if hasattr(im, "convert"):
            return im.convert("RGB")
        if isinstance(im, bytes):
            return Image.open(io.BytesIO(im)).convert("RGB")
        try:
            import numpy as np

            if isinstance(im, np.ndarray):
                return Image.fromarray(im).convert("RGB")
        except ImportError:
            pass
        try:
            return Image.open(io.BytesIO(bytes(im))).convert("RGB")
        except Exception as e:
            raise TypeError(f"Unsupported image field type {type(im)}") from e

    return load_rgb


def load_or_create_json(
    split: str, annotations_dir: Path, from_hf: bool, repo_root: Path
) -> List[Dict[str, Any]]:
    json_path = annotations_dir / f"{split}_annotations.json"
    if json_path.is_file():
        print(f"[1] Using existing {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[1] Loaded {len(data)} rows")
        return data
    if from_hf:
        annotations = load_from_huggingface(split)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        print(f"[1] Wrote {path_for_json(repo_root, json_path)}")
        return annotations
    raise FileNotFoundError(
        f"Missing {json_path}. Run with --from-hf once, or place JSON files under annotations/."
    )


def slugify_predicate(pred: str) -> str:
    return " ".join((pred or "").strip().split()).replace(" ", "-")


def build_obj_rel_rows(
    ann: Dict[str, Any],
) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
    id_map: Dict[str, int] = {}
    obj_rows: List[Tuple[Any, ...]] = []
    for idx, o in enumerate(ann["objects"], start=1):
        raw_id = o["id"]
        name = raw_id.rsplit(".", 1)[0] if "." in raw_id else raw_id
        id_map[raw_id] = idx
        obj_rows.append((idx, name, *o["bbox"]))
    rel_rows: List[Tuple[Any, ...]] = []
    for r in ann["relationships"]:
        sj = id_map.get(r["subject"])
        oj = id_map.get(r["object"])
        if sj is not None and oj is not None:
            rel_rows.append((sj, slugify_predicate(r["predicate"]), oj))
    return obj_rows, rel_rows


def annotation_lines_from_rows(
    obj_rows: List[Tuple[Any, ...]], rel_rows: List[Tuple[Any, ...]]
) -> List[str]:
    lines: List[str] = []
    lines.append(f"      obj[{len(obj_rows)}]{{id,name,x1,y1,x2,y2}}:")
    for row in obj_rows:
        lines.append(f"        {','.join(map(str, row))}")
    lines.append(f"      rel[{len(rel_rows)}]{{subj,pred,obj}}:")
    for row in rel_rows:
        lines.append(f"        {','.join(map(str, row))}")
    return lines


def scale_annotation_lines(
    annotation_lines: List[str], scale_x: float, scale_y: float
) -> List[str]:
    new_lines: List[str] = []
    in_obj_block = False
    for line in annotation_lines:
        stripped = line.lstrip()
        if not stripped:
            new_lines.append(line)
            continue
        if stripped.startswith("obj[") and "{id,name,x1,y1,x2,y2}" in stripped:
            in_obj_block = True
            new_lines.append(line)
            continue
        if stripped.startswith("rel[") and "{subj,pred,obj}" in stripped:
            in_obj_block = False
            new_lines.append(line)
            continue
        if in_obj_block:
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) == 6:
                try:
                    idx_s, name, x1s, y1s, x2s, y2s = parts
                    x1, y1, x2, y2 = map(int, (x1s, y1s, x2s, y2s))
                    x1n, y1n, x2n, y2n = (
                        int(round(x1 * scale_x)),
                        int(round(y1 * scale_y)),
                        int(round(x2 * scale_x)),
                        int(round(y2 * scale_y)),
                    )
                    prefix = line[: len(line) - len(stripped)]
                    new_lines.append(f"{prefix}{idx_s},{name},{x1n},{y1n},{x2n},{y2n}")
                    continue
                except (ValueError, IndexError):
                    pass
        new_lines.append(line)
    return new_lines


def open_source_image(
    ann: Dict[str, Any],
    annotations_dir: Path,
    repo_root: Path,
    hf_loader: Callable[[str], Image.Image] | None,
) -> Image.Image:
    ip = ann.get("image_path")
    if ip:
        for base in (annotations_dir, annotations_dir.parent, repo_root):
            cand = (base / ip).resolve()
            if cand.is_file():
                return Image.open(cand).convert("RGB")
    if hf_loader is not None:
        return hf_loader(str(ann["image_id"]))
    raise FileNotFoundError(
        f"No image for image_id={ann.get('image_id')!r}: "
        "add image_path + files on disk, or run without --no-hf-images."
    )


def process_one_sample(
    ann: Dict[str, Any],
    out_frame_dir: Path,
    repo_root: Path,
    annotations_dir: Path,
    hf_loader: Callable[[str], Image.Image] | None,
) -> Dict[str, Any] | None:
    filename = f"{ann['image_id']}.jpg"
    scaled_img_path = out_frame_dir / filename
    obj_rows, rel_rows = build_obj_rel_rows(ann)
    ann_lines = annotation_lines_from_rows(obj_rows, rel_rows)

    try:
        img = open_source_image(ann, annotations_dir, repo_root, hf_loader)
        orig_w, orig_h = img.size
        if orig_w > 0 and orig_h > 0:
            sx = TARGET_WIDTH / orig_w
            sy = TARGET_HEIGHT / orig_h
            resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)
            resized.save(scaled_img_path, quality=95)
            scaled_lines = scale_annotation_lines(ann_lines, sx, sy)
            answer_toon = "\n".join(scaled_lines)
        else:
            answer_toon = "\n".join(ann_lines)
    except Exception as e:
        print(f"[WARN] skip image_id={ann.get('image_id')}: {e}", file=sys.stderr)
        return None

    return {
        "image_id": ann["image_id"],
        "image_path": path_for_json(repo_root, scaled_img_path),
        "answer_toon": answer_toon,
    }


def build_split_sft_json(
    split: str,
    annotations: List[Dict[str, Any]],
    annotations_dir: Path,
    export_root: Path,
    images_out_root: Path,
    repo_root: Path,
    use_hf_images: bool,
) -> Path:
    frames_sub = "train_images" if split == "train" else "test_images"
    out_frame_dir = images_out_root / frames_sub
    out_frame_dir.mkdir(parents=True, exist_ok=True)
    sft_json_path = export_root / f"{split}_annotations_toon_sft.json"

    hf_loader: Callable[[str], Image.Image] | None = None
    if use_hf_images:
        hf_loader = build_hf_image_loader(split)

    samples: List[Dict[str, Any]] = []
    for ann in tqdm(annotations, desc=f"SFT {split}"):
        row = process_one_sample(ann, out_frame_dir, repo_root, annotations_dir, hf_loader)
        if row is not None:
            samples.append(row)

    export_root.mkdir(parents=True, exist_ok=True)
    with open(sft_json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(
        f"[2] Wrote {len(samples)} samples ==> {path_for_json(repo_root, sft_json_path)}"
    )
    print(f"[2] Frames (rel repo): {path_for_json(repo_root, out_frame_dir)}/")
    return sft_json_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PSG: annotations in annotations/, SFT JSON + resized frames (pixels from HF by default)"
    )
    ap.add_argument(
        "--repo_root",
        default="",
        help="SceneGraphVLM repo root (empty ==> infer from this file).",
    )
    ap.add_argument(
        "--annotations_dir",
        default=ANNOTATIONS_DIR_REL,
        help="Directory for train/test *_annotations.json (repo-relative unless absolute).",
    )
    ap.add_argument(
        "--export_root",
        default=EXPORT_ROOT_REL,
        help="Output dir for *_annotations_toon_sft.json.",
    )
    ap.add_argument(
        "--images_out",
        default=IMAGES_OUT_REL,
        help="Output root; train_images/ and test_images/ under it.",
    )
    ap.add_argument(
        "--from-hf",
        action="store_true",
        help="If *_annotations.json is missing, download annotation JSON from HuggingFace (no raw images saved).",
    )
    ap.add_argument(
        "--no-hf-images",
        action="store_true",
        help="Do not load pixels from HuggingFace; every row must have a resolvable local image_path.",
    )
    ap.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip split if *_annotations_toon_sft.json already exists.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _REPO_ROOT_DEFAULT
    annotations_dir = resolve_under_repo(repo_root, args.annotations_dir)
    export_root = resolve_under_repo(repo_root, args.export_root)
    images_out_root = resolve_under_repo(repo_root, args.images_out)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    images_out_root.mkdir(parents=True, exist_ok=True)

    use_hf_images = not args.no_hf_images

    print("=" * 60)
    print("PSG ==> SFT (annotations only on disk; HF pixels unless --no-hf-images)")
    print("=" * 60)

    for split in SPLITS:
        print(f"\n--- split: {split} ---\n")
        annotations = load_or_create_json(split, annotations_dir, args.from_hf, repo_root)

        sft_json_path = export_root / f"{split}_annotations_toon_sft.json"
        if args.skip_sft and sft_json_path.is_file():
            print(f"[2] Skip SFT (exists {sft_json_path.name})")
            continue

        build_split_sft_json(
            split,
            annotations,
            annotations_dir,
            export_root,
            images_out_root,
            repo_root,
            use_hf_images,
        )

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
