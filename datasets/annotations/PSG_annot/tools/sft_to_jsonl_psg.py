#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSG intermediate JSON ==> Swift-style chat jsonl (user prompt: template + example in ``<answer>``).

Reads ``train_annotations_toon_sft.json`` / ``test_annotations_toon_sft.json`` from
``prepare_original_psg_sft.py`` (under ``data_sft_original``). ``image_path`` is
repo-relative; joined with ``--repo_root`` for ``images``.

**Commands** (repository root = SceneGraphVLM):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/PSG_annot/tools/sft_to_jsonl_psg.py

Default output: ``datasets/data_playground/PSG_json/train.jsonl`` and ``test.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]
EXPORT_ROOT_REL = "datasets/annotations/PSG_annot/data_sft_original"
OUT_DIR_REL = "datasets/data_playground/PSG_json"

PSG_CATEGORIES: Dict[str, List[str]] = {
    "thing_classes": [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    ],
    "stuff_classes": [
        "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff",
        "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff",
        "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof",
        "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick",
        "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind",
        "window-other", "tree-merged", "fence-merged", "ceiling-merged",
        "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged",
        "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged",
        "paper-merged", "food-other-merged", "building-other-merged", "rock-merged",
        "wall-other-merged", "rug-merged",
    ],
    "predicate_classes": [
        "over", "in front of", "beside", "on", "in", "attached to", "hanging from",
        "on back of", "falling off", "going down", "painted on", "walking on",
        "running on", "crossing", "standing on", "lying on", "sitting on",
        "flying over", "jumping over", "jumping from", "wearing", "holding",
        "carrying", "looking at", "guiding", "kissing", "eating", "drinking",
        "feeding", "biting", "catching", "picking", "playing with", "chasing",
        "climbing", "cleaning", "playing", "touching", "pushing", "pulling",
        "opening", "cooking", "talking to", "throwing", "slicing", "driving",
        "riding", "parked on", "driving on", "about to hit", "kicking", "swinging",
        "entering", "exiting", "enclosing", "leaning on",
    ],
}

PSG_OBJ_CATEGORIES: List[str] = (
    PSG_CATEGORIES["thing_classes"] + PSG_CATEGORIES["stuff_classes"]
)
PSG_REL_CATEGORIES: List[str] = PSG_CATEGORIES["predicate_classes"]

IMG_PREFIX = "<image>\n"


def resolve_under_repo(repo_root: Path, path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / path_arg).resolve()


EXAMPLE_PSG_IN_ANSWER = (
    "<answer>\n"
    "obj[7]{id,name,x1,y1,x2,y2}:\n"
    "  1,person,281,272,524,438\n"
    "  2,umbrella,273,123,640,434\n"
    "  3,house,0,88,262,426\n"
    "  4,window-other,163,262,195,294\n"
    "  5,tree-merged,0,0,640,440\n"
    "  6,sky-other-merged,0,0,459,123\n"
    "  7,building-other-merged,537,164,640,291\n"
    "rel[5]{subj,pred,obj}:\n"
    "  1,in-front-of,5\n"
    "  3,attached-to,4\n"
    "  4,hanging-from,3\n"
    "  5,beside,3\n"
    "  6,over,5\n"
    "</answer>"
)


def build_user_prompt() -> str:
    """
    Output Format inside <answer>, then Guidelines with full category JSON, 
    then Example inside <answer>, then instruction to generate the complete scene graph for the provided image.
    """
    obj_cls_str = json.dumps(PSG_OBJ_CATEGORIES, ensure_ascii=False)
    rel_cls_str = json.dumps(PSG_REL_CATEGORIES, ensure_ascii=False)
    return (
        "Generate a structured scene graph for an image of size (640 x 480) using the specified "
        "object and relationship categories.\n\n"
        "Output Format:\n\n"
        "<answer>\n"
        "obj[N]{id,name,x1,y1,x2,y2}:\n"
        "  id,name,x1,y1,x2,y2\n"
        "  ...\n"
        "rel[M]{subj,pred,obj}:\n"
        "  subj,pred,obj\n"
        "  ...\n\n"
        "</answer>\n"
        "Guidelines:\n"
        "- Objects:\n"
        "  - Use integer IDs starting from 1 in the id field (e.g., 1, 2, 3).\n"
        f"  - The object name must belong to the predefined object set: {obj_cls_str}.\n"
        "  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.\n"
        "  - Include all visible objects, even if they have no relationships.\n\n"
        "- Relationships:\n"
        "  - Represent interactions using integer object IDs in subj and obj.\n"
        f"  - The pred (predicate) must belong to the predefined relationship set: {rel_cls_str}\n"
        "  - Omit relationships for orphan objects.\n\n"
        "Example output:\n"
        + EXAMPLE_PSG_IN_ANSWER
        + "\n"
        "Now, generate the complete scene graph for the provided image. "
        "Write your response only between <answer> and </answer> tags.\n"
    )


USER_PROMPT = build_user_prompt()


def format_assistant_body(answer_toon: str) -> str:
    out: List[str] = []
    for line in answer_toon.split("\n"):
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("obj[") or (
            stripped.startswith("rel[") and "{subj,pred,obj}" in stripped
        ):
            out.append("      " + stripped)
        else:
            out.append("        " + stripped)
    return "\n".join(out)


def wrap_assistant(answer_toon: str) -> str:
    body = format_assistant_body(answer_toon)
    return "<answer>\n" + body + "\n</answer>\n"


def make_jsonl(in_path: str, out_path: str, repo_root: str) -> int:
    repo_root = os.path.abspath(repo_root)
    with open(in_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(data, desc=os.path.basename(out_path)):
            rel_img = sample["image_path"]
            if os.path.isabs(rel_img):
                abs_img = os.path.normpath(rel_img)
            else:
                abs_img = os.path.normpath(os.path.join(repo_root, rel_img))

            assistant_wrapped = wrap_assistant(sample["answer_toon"])
            item = {
                "messages": [
                    {"role": "user", "content": IMG_PREFIX + USER_PROMPT},
                    {"role": "assistant", "content": assistant_wrapped},
                ],
                "images": [abs_img],
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PSG TOON JSON ==> Swift-style jsonl with <answer> tags"
    )
    ap.add_argument(
        "--repo_root",
        default="",
        help="SceneGraphVLM repo root (empty ==> infer from this file).",
    )
    ap.add_argument(
        "--export_root",
        default=EXPORT_ROOT_REL,
        help="Directory with train/test *_annotations_toon_sft.json.",
    )
    ap.add_argument(
        "--out_dir",
        default=OUT_DIR_REL,
        help="Output directory for train.jsonl / test.jsonl.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _REPO_ROOT_DEFAULT
    export_root = resolve_under_repo(repo_root, args.export_root)
    out_dir = resolve_under_repo(repo_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_in = str(export_root / "train_annotations_toon_sft.json")
    test_in = str(export_root / "test_annotations_toon_sft.json")
    train_out = str(out_dir / "train.jsonl")
    test_out = str(out_dir / "test.jsonl")

    if not os.path.isfile(train_in):
        print(f"Error: missing {train_in}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(test_in):
        print(f"Error: missing {test_in}", file=sys.stderr)
        sys.exit(1)

    n_train = make_jsonl(train_in, train_out, str(repo_root))
    n_test = make_jsonl(test_in, test_out, str(repo_root))
    print("Wrote:", train_out, f"({n_train} lines)")
    print("Wrote:", test_out, f"({n_test} lines)")
    if n_train < 1000:
        print(
            "\n[warn] Few train lines - run prepare_original_psg_sft.py on full data.\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
