#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3RScan: TOON SFT JSON -> Swift-style chat JSONL (``messages`` + ``images``).

Reads ``train_annotations_toon_sft.json`` / ``test_annotations_toon_sft.json``
produced by ``prepare_original_3rscan_sft.py``. Each ``image_path`` is repo-relative
and resolved against ``--repo_root`` to fill the ``images`` array (single absolute
path per sample, same convention as ``sft_to_jsonl_pvsg.py``).

For each frame within a single 3RScan scan, samples are written in scan order:

- ``no_prev_gt`` mode (default): every frame uses the first-frame prompt.
- ``with_prev_gt`` mode: frames after the first one in a scan get the previous
  frame's GT scene graph as temporal context (PVSG-style ``<<PREV_TOON>>``).

The **user** prompt matches the **PVSG** style (``Output Format`` + short
``Guidelines`` + example in ``<answer>``), with the only extension that each
object row includes ``color`` and ``material`` before the box — same as the GT
in ``prepare_original_3rscan_sft.py``. No long class/relation list is embedded
in the prompt (unlike an earlier version of this script).

**Run** (from the SceneGraphVLM repository root):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/3RScan_annot/tools/sft_to_jsonl_3rscan.py

Outputs ``train.jsonl`` / ``test.jsonl`` under
``datasets/data_playground/3RScan_json_<temporal_mode>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]

DEFAULT_INPUT_DIR = "datasets/annotations/3RScan_annot/data_sft_original"
DEFAULT_OUTPUT_DIR = "datasets/data_playground/3RScan_json"

TRAIN_JSON_NAME = "train_annotations_toon_sft.json"
TEST_JSON_NAME = "test_annotations_toon_sft.json"
OUT_TRAIN_JSONL = "train.jsonl"
OUT_TEST_JSONL = "test.jsonl"

IMG_PREFIX = "<image>\n"

TEMPORAL_MODE_NO_PREV_GT = "no_prev_gt"
TEMPORAL_MODE_WITH_PREV_GT = "with_prev_gt"
TEMPORAL_MODES = (TEMPORAL_MODE_NO_PREV_GT, TEMPORAL_MODE_WITH_PREV_GT)


# ===================== PROMPTS (same structure as sft_to_jsonl_pvsg.py) =====================

# Curly braces in TOON examples: do not use str.format(); use .replace("<<PREV_TOON>>", ...).

R3SCAN_OUTPUT_FORMAT_IN_ANSWER = (
    "<answer>\n"
    "obj[N]{id,name,color,material,x1,y1,x2,y2}:\n"
    "  id,name,color,material,x1,y1,x2,y2\n"
    "  ...\n"
    "rel[M]{subj,pred,obj}:\n"
    "  subj,pred,obj\n"
    "  ...\n\n"
    "</answer>\n"
)

R3SCAN_GUIDELINES = (
    "Guidelines:\n"
    "- Objects:\n"
    "  - Use integer IDs starting from 1 in the id field (e.g., 1, 2, 3).\n"
    "  - The name must be the object category name (e.g. chair, wall).\n"
    "  - color and material are short strings; use 'none' if not applicable or unknown.\n"
    "  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.\n"
    "  - Include all visible objects, even if they have no relationships.\n\n"
    "- Relationships:\n"
    "  - Represent interactions using integer object IDs in subj and obj.\n"
    "  - pred is the relationship type (string), such as in-front-of, attached-to, standing-on. "
    "Use hyphens for multi-word predicates.\n"
    "  - Omit relationships for objects that do not participate in any interaction.\n\n"
)

EXAMPLE_R3SCAN_IN_ANSWER = (
    "<answer>\n"
    "obj[5]{id,name,color,material,x1,y1,x2,y2}:\n"
    "  1,floor,beige,none,0,0,640,479\n"
    "  2,wall,white,ceramic,0,0,400,479\n"
    "  3,table,brown,wooden,120,210,520,420\n"
    "  4,chair,black,plastic,180,250,310,470\n"
    "  5,lamp,white,plastic,420,40,500,180\n"
    "rel[4]{subj,pred,obj}:\n"
    "  3,standing-on,1\n"
    "  4,standing-on,1\n"
    "  4,close-by,3\n"
    "  5,attached-to,2\n"
    "</answer>"
)

# Same intro/closing as PVSG (wording), plus 3RScan-specific Output Format + example above.
R3SCAN_PROMPT_INTRO = (
    "Generate a structured scene graph for an image of size (640 x 480) using the following "
    "text format.\n\n"
    "Output Format:\n\n"
)

R3SCAN_CLOSING = (
    "Now, generate the complete scene graph for the provided image. "
    "Write your response only between <answer> and </answer> tags.\n"
)

R3SCAN_TEMPORAL_AFTER_EXAMPLE = (
    "You are also given the previous frame's ground-truth scene graph in TOON format.\n"
    "Use it as temporal context, but rely primarily on the current image.\n"
    "Important:\n"
    "- Include all objects visible in the CURRENT image, even if they did not exist in the "
    "previous graph.\n"
    "- Do NOT include objects that are NOT visible in the current image, even if they exist in "
    "the previous graph.\n"
    "- Output ONLY the complete scene graph for the CURRENT image, using the TOON structure "
    "from the Output Format above, inside one <answer>...</answer> block.\n\n"
    "Previous frame scene graph (TOON):\n"
    "<<PREV_TOON>>\n\n"
)

_PROMPT_UP_TO_EXAMPLE = (
    R3SCAN_PROMPT_INTRO
    + R3SCAN_OUTPUT_FORMAT_IN_ANSWER
    + R3SCAN_GUIDELINES
    + "Example output:\n"
    + EXAMPLE_R3SCAN_IN_ANSWER
    + "\n"
)

PROMPT_NO_PREV = _PROMPT_UP_TO_EXAMPLE + R3SCAN_CLOSING
PROMPT_WITH_PREV = _PROMPT_UP_TO_EXAMPLE + R3SCAN_TEMPORAL_AFTER_EXAMPLE + R3SCAN_CLOSING


def build_prompts() -> Tuple[str, str]:
    return PROMPT_NO_PREV, PROMPT_WITH_PREV


# ===================== JSONL HELPERS =====================

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


def to_rel_image_path(img_path: str, repo_root: Path) -> str:
    p = Path(img_path)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p.resolve()).replace("\\", "/")
    return str(Path(img_path).as_posix())


def abs_image_path(rel_or_abs: str, repo_root: Path) -> str:
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    return os.path.normpath(os.path.join(str(repo_root), rel_or_abs))


def format_assistant_body(answer_toon: str) -> str:
    """Section headers 6 spaces, data lines 8 spaces (PSG/PVSG style)."""
    out: List[str] = []
    for line in answer_toon.split("\n"):
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("obj[") or stripped.startswith("rel["):
            out.append("      " + stripped)
        else:
            out.append("        " + stripped)
    return "\n".join(out)


def wrap_assistant(answer_toon: str) -> str:
    body = format_assistant_body(answer_toon)
    return "<answer>\n" + body + "\n</answer>\n"


_FRAME_RE = re.compile(r"frame-(\d+)")


def extract_scan_and_order(sample: Dict[str, Any], rel_img: str) -> Tuple[str, int]:
    image_id = sample.get("image_id") or ""
    if "_frame-" in image_id:
        scan_id, _, frame_part = image_id.rpartition("_frame-")
        m = _FRAME_RE.search("frame-" + frame_part)
        if m:
            return scan_id, int(m.group(1))

    parts = rel_img.replace("\\", "/").split("/")
    if len(parts) >= 2:
        scan_id = parts[-2]
        m = _FRAME_RE.search(parts[-1])
        if m:
            return scan_id, int(m.group(1))
        return scan_id, 0
    return "__single__", 0


def collect_samples(input_dir: Path, json_basename: str) -> List[Dict[str, Any]]:
    p = input_dir / json_basename
    if not p.is_file():
        return []
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {p}")
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "image_path" not in item or "answer_toon" not in item:
            continue
        out.append(item)
    return out


def write_jsonl(
    samples: List[Dict[str, Any]],
    out_path: Path,
    repo_root: Path,
    split_name: str,
    prompt_no_prev: str,
    prompt_with_prev: str,
    temporal_mode: str,
) -> int:
    groups: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for sample in samples:
        rel_img = to_rel_image_path(sample["image_path"], repo_root)
        scan_id, order = extract_scan_and_order(sample, rel_img)
        groups[scan_id].append((order, rel_img, sample["answer_toon"]))

    for scan_id in groups:
        groups[scan_id].sort(key=lambda x: x[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for scan_id in tqdm(
            sorted(groups.keys()),
            desc=f"Building {split_name} JSONL ({temporal_mode})",
            ncols=80,
        ):
            frames = groups[scan_id]
            if not frames:
                continue

            for idx, (_, rel_img, toon_cur) in enumerate(frames):
                if temporal_mode == TEMPORAL_MODE_NO_PREV_GT or idx == 0:
                    user_body = prompt_no_prev
                else:
                    _, _, toon_prev = frames[idx - 1]
                    user_body = prompt_with_prev.replace("<<PREV_TOON>>", toon_prev)

                item = {
                    "messages": [
                        {"role": "user", "content": IMG_PREFIX + user_body},
                        {"role": "assistant", "content": wrap_assistant(toon_cur)},
                    ],
                    "images": [abs_image_path(rel_img, repo_root)],
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    print(f"[done] {split_name} ({temporal_mode}): wrote {written} samples -> {path_for_json(repo_root, out_path)}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="3RScan TOON JSON -> Swift chat JSONL (PVSG-style user prompt, 8-col TOON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--repo_root", default=str(_REPO_ROOT_DEFAULT))
    ap.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. Final dir is {output_dir}_{temporal_mode}/."
        ),
    )
    ap.add_argument(
        "--temporal_mode",
        default=TEMPORAL_MODE_NO_PREV_GT,
        choices=list(TEMPORAL_MODES),
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_dir = resolve_under_repo(repo_root, args.input_dir)
    output_root = resolve_under_repo(repo_root, args.output_dir)
    output_dir = output_root.with_name(output_root.name + f"_{args.temporal_mode}")

    if not input_dir.is_dir():
        print(f"Error: input_dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Repo root:   {repo_root}")
    print(f"Input dir:   {path_for_json(repo_root, input_dir)}")
    print(f"Output dir:  {path_for_json(repo_root, output_dir)}")
    print(f"Temporal:    {args.temporal_mode}")

    prompt_no_prev, prompt_with_prev = build_prompts()

    train_samples = collect_samples(input_dir, TRAIN_JSON_NAME)
    test_samples = collect_samples(input_dir, TEST_JSON_NAME)
    print(f"Train samples: {len(train_samples)} (from {TRAIN_JSON_NAME})")
    print(f"Test  samples: {len(test_samples)} (from {TEST_JSON_NAME})")

    if not train_samples and not test_samples:
        print("Error: no samples found. Did you run prepare_original_3rscan_sft.py first?", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if train_samples:
        write_jsonl(
            train_samples,
            output_dir / OUT_TRAIN_JSONL,
            repo_root,
            "train",
            prompt_no_prev,
            prompt_with_prev,
            args.temporal_mode,
        )
    if test_samples:
        write_jsonl(
            test_samples,
            output_dir / OUT_TEST_JSONL,
            repo_root,
            "test",
            prompt_no_prev,
            prompt_with_prev,
            args.temporal_mode,
        )

    print("Done")


if __name__ == "__main__":
    main()
