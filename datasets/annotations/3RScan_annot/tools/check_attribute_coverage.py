"""Quick sanity-check: how many objects in 3RScan annotations actually have
``color`` / ``material`` attributes filled in.

Run:

    /data/homes/makarov_vd/.conda/envs/swift_qwen_3_5_sft/bin/python \
        datasets/annotations/3RScan_annot/tools/check_attribute_coverage.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ANNOT_DIR = Path(
    "/data/homes/makarov_vd/workspace/SceneGraphVLM/datasets/annotations/"
    "3RScan_annot/annotations"
)


def _pct(part: int, total: int) -> str:
    if total == 0:
        return f"{part} (-)"
    return f"{part} ({100 * part / total:.1f}%)"


def analyze_per_frame(split: str) -> None:
    path = ANNOT_DIR / f"{split}_annotations.json"
    with path.open() as f:
        data = json.load(f)

    n_frames = 0
    n_objs = 0
    n_no_color = 0
    n_no_material = 0
    n_neither = 0
    n_both = 0
    frames_no_color = 0
    frames_no_material = 0
    cls_total: collections.Counter[str] = collections.Counter()
    cls_no_mat: collections.Counter[str] = collections.Counter()
    cls_no_col: collections.Counter[str] = collections.Counter()

    for image in data:
        n_frames += 1
        nodes = image.get("nodes", []) or []
        any_color = False
        any_material = False
        for node in nodes:
            attrs = node.get("data", {}).get("attributes", {}) or {}
            cls = node.get("data", {}).get("class_name", "?")
            color = attrs.get("color")
            material = attrs.get("material")
            n_objs += 1
            cls_total[cls] += 1
            if not color:
                n_no_color += 1
                cls_no_col[cls] += 1
            else:
                any_color = True
            if not material:
                n_no_material += 1
                cls_no_mat[cls] += 1
            else:
                any_material = True
            if not color and not material:
                n_neither += 1
            if color and material:
                n_both += 1
        if not any_color:
            frames_no_color += 1
        if not any_material:
            frames_no_material += 1

    print(f"=== {split} (per-frame annotations) ===")
    print(f"frames:             {n_frames}")
    print(f"objects:            {n_objs}")
    print(f"  no color:         {_pct(n_no_color, n_objs)}")
    print(f"  no material:      {_pct(n_no_material, n_objs)}")
    print(f"  neither set:      {_pct(n_neither, n_objs)}")
    print(f"  both set:         {_pct(n_both, n_objs)}")
    print(f"frames w/o ANY color:    {_pct(frames_no_color, n_frames)}")
    print(f"frames w/o ANY material: {_pct(frames_no_material, n_frames)}")
    print("top-10 classes most often missing material:")
    for c, n in sorted(cls_no_mat.items(), key=lambda kv: -kv[1])[:10]:
        total = cls_total[c]
        print(f"    {c:24s} no_material={n}/{total}  ({100 * n / total:.0f}%)")
    print("top-10 classes most often missing color:")
    for c, n in sorted(cls_no_col.items(), key=lambda kv: -kv[1])[:10]:
        total = cls_total[c]
        print(f"    {c:24s} no_color={n}/{total}  ({100 * n / total:.0f}%)")
    print()


def analyze_objects_json() -> None:
    path = ANNOT_DIR / "objects.json"
    if not path.exists():
        return
    with path.open() as f:
        data = json.load(f)

    scans = data.get("scans", [])
    n_scans = len(scans)
    n_objs = 0
    n_no_color = 0
    n_no_material = 0
    for scan in scans:
        for obj in scan.get("objects", []) or []:
            attrs = obj.get("attributes", {}) or {}
            n_objs += 1
            if not attrs.get("color"):
                n_no_color += 1
            if not attrs.get("material"):
                n_no_material += 1

    print("=== objects.json (scene-level annotations) ===")
    print(f"scans:   {n_scans}")
    print(f"objects: {n_objs}")
    print(f"  no color:    {_pct(n_no_color, n_objs)}")
    print(f"  no material: {_pct(n_no_material, n_objs)}")
    print()


if __name__ == "__main__":
    analyze_per_frame("train")
    analyze_per_frame("test")
    analyze_objects_json()
