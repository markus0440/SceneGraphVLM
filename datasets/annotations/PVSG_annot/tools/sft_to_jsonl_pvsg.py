#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Swift-style chat JSONL (``messages`` + ``images``) with optional previous-frame GT in the user prompt.

Same layout as ``sft_to_jsonl_psg.py`` / ``sft_to_jsonl_ag.py``:

- ``messages``: ``[{"role":"user","content": "<image>\\n" + ...}, {"role":"assistant","content": "<answer>..."}]``
- ``images``: one **absolute** filesystem path per line (repo root + ``image_path`` from JSON)

**Default inputs** are folders from ``prepare_original_pvsg_sft.py`` (``train_annotations_toon_sft.json``,
``test_annotations_toon_sft.json``). Multiple ``--input_dir`` values merge train with train and test with test.

Outputs exactly ``train.jsonl`` and ``test.jsonl`` under ``--output_dir``.

**Run** (from SceneGraphVLM repo root):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/PVSG_annot/tools/sft_to_jsonl_pvsg.py \\
    --input_dir datasets/annotations/PVSG_annot/data_sft_original

  # Multiple source folders (e.g. different filtered splits):
  python datasets/annotations/PVSG_annot/tools/sft_to_jsonl_pvsg.py \\
    --input_dir path/to/set_a --input_dir path/to/set_b \\
    --output_dir datasets/data_playground/PVSG_json/pvsg_all_data_gt_prompt

**Replace-prompt-only** mode (existing JSONL, same repo-relative paths for I/O):

  python datasets/annotations/PVSG_annot/tools/sft_to_jsonl_pvsg.py \\
    --replace-prompt-only --input path/to/in.jsonl --output path/to/out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# <repo>/datasets/annotations/PVSG_annot/tools/<this>.py
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]

DEFAULT_INPUT_DIR = "datasets/annotations/PVSG_annot/data_sft_original"
DEFAULT_OUTPUT_DIR = "datasets/data_playground/PVSG_json/pvsg_all_data_gt_prompt"
TRAIN_JSON_NAME = "train_annotations_toon_sft.json"
TEST_JSON_NAME = "test_annotations_toon_sft.json"
OUT_TRAIN_JSONL = "train.jsonl"
OUT_TEST_JSONL = "test.jsonl"

IMG_PREFIX = "<image>\n"

# ===================== PROMPTS =====================

# Curly braces in TOON examples: do not use str.format(); use .replace("<<PREV_TOON>>", ...).
# Structure matches sft_to_jsonl_psg.py: Output Format + Example inside <answer>...</answer>.

PVSG_OUTPUT_FORMAT_IN_ANSWER = (
    "<answer>\n"
    "obj[N]{id,name,x1,y1,x2,y2}:\n"
    "  id,name,x1,y1,x2,y2\n"
    "  ...\n"
    "rel[M]{subj,pred,obj}:\n"
    "  subj,pred,obj\n"
    "  ...\n\n"
    "</answer>\n"
)

PVSG_GUIDELINES = (
    "Guidelines:\n"
    "- Objects:\n"
    "  - Use integer IDs starting from 1 in the id field (e.g., 1, 2, 3).\n"
    "  - The name must be the object category name (e.g., person, umbrella).\n"
    "  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.\n"
    "  - Include all visible objects, even if they have no relationships.\n\n"
    "- Relationships:\n"
    "  - Represent interactions using integer object IDs in subj and obj.\n"
    "  - pred is the relationship type (string), such as in-front-of, attached-to, beside.\n"
    "  - Omit relationships for objects that do not participate in any interaction.\n\n"
)

EXAMPLE_PVSG_IN_ANSWER = (
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

PVSG_PROMPT_INTRO = (
    "Generate a structured scene graph for an image of size (640 x 480) using the following "
    "text format.\n\n"
    "Output Format:\n\n"
)

PVSG_CLOSING = (
    "Now, generate the complete scene graph for the provided image. "
    "Write your response only between <answer> and </answer> tags.\n"
)

_PROMPT_UP_TO_EXAMPLE = (
    PVSG_PROMPT_INTRO
    + PVSG_OUTPUT_FORMAT_IN_ANSWER
    + PVSG_GUIDELINES
    + "Example output:\n"
    + EXAMPLE_PVSG_IN_ANSWER
    + "\n"
)

PVSG_TEMPORAL_AFTER_EXAMPLE = (
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

PROMPT_NO_PREV = _PROMPT_UP_TO_EXAMPLE + PVSG_CLOSING

PROMPT_WITH_PREV = _PROMPT_UP_TO_EXAMPLE + PVSG_TEMPORAL_AFTER_EXAMPLE + PVSG_CLOSING


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


def load_prompt_from_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def to_rel_image_path(img_path: str, repo_root: Path) -> str:
    p = Path(img_path)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p.resolve()).replace("\\", "/")
    return str(Path(img_path).as_posix())


def abs_image_path(rel_or_abs: str, repo_root: Path) -> str:
    """Swift ``images`` list: single absolute path (same convention as sft_to_jsonl_psg)."""
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    return os.path.normpath(os.path.join(str(repo_root), rel_or_abs))


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


def _try_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def extract_video_and_order(sample: Dict[str, Any], rel_img: str) -> tuple:
    vid = sample.get("video_id") or sample.get("vid") or sample.get("video")
    if vid is not None:
        for k in ("frame_idx", "frame_id", "frame_num", "timestamp", "time", "frame"):
            if k in sample:
                ok = sample[k]
                ok_int = _try_int(ok)
                return str(vid), (ok_int if ok_int is not None else str(ok))

    parts = rel_img.replace("\\", "/").split("/")
    if len(parts) >= 2:
        cand_vid = parts[-2]
        if re.search(r"\d", cand_vid):
            filename = parts[-1]
            stem = Path(filename).stem
            m = re.search(r"(\d+)", stem)
            if m:
                return cand_vid, int(m.group(1))
            return cand_vid, stem

    return "__single__", rel_img


def normalize_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "image_path" in sample and "answer_toon" in sample:
        return sample
    if "conversations" in sample and "image" in sample:
        conv = sample["conversations"]
        toon = conv[1]["value"] if len(conv) > 1 else ""
        return {
            "image_path": sample["image"],
            "answer_toon": toon,
            "video_id": sample.get("video_id"),
            "frame_idx": sample.get("t"),
        }
    if "messages" in sample and sample.get("images"):
        msgs = sample["messages"]
        if len(msgs) < 2:
            return None
        img0 = sample["images"][0]
        raw = msgs[1].get("content") or ""
        toon = raw
        if "<answer>" in raw:
            inner = raw
            if inner.startswith("<answer>\n"):
                inner = inner[len("<answer>\n") :]
            elif inner.startswith("<answer>"):
                inner = inner[len("<answer>") :]
            inner = inner.rstrip()
            if inner.endswith("</answer>\n"):
                inner = inner[: -len("</answer>\n")]
            elif inner.endswith("</answer>"):
                inner = inner[: -len("</answer>")]
            toon = inner.strip()
        return {
            "image_path": img0,
            "answer_toon": toon,
            "video_id": sample.get("video_id"),
            "frame_idx": sample.get("t"),
        }
    return None


def collect_samples_from_dirs(
    input_dirs: List[Path],
    json_basename: str,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for d in input_dirs:
        p = d / json_basename
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {p}")
        for item in data:
            norm = normalize_sample(item)
            if norm is not None:
                merged.append(norm)
    return merged


def make_jsonl_prev(
    samples: List[Dict[str, Any]],
    out_path: Path,
    repo_root: Path,
    split_name: str,
    prompt_no_prev: str,
    prompt_with_prev: str,
    include_first_frame_no_prev: bool = True,
    first_frame_ratio: float = 1.0,
) -> int:
    groups: Dict[Any, List] = defaultdict(list)

    for sample in samples:
        img_path = sample["image_path"]
        rel_img = to_rel_image_path(img_path, repo_root)
        vid, order_key = extract_video_and_order(sample, rel_img)
        groups[vid].append((order_key, rel_img, sample["answer_toon"]))

    for vid in groups:
        groups[vid].sort(key=lambda x: x[0])

    out_count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for vid in tqdm(list(groups.keys()), desc=f"Building {split_name} JSONL (prev-frame)"):
            frames = groups[vid]
            if not frames:
                continue

            if include_first_frame_no_prev:
                if first_frame_ratio >= 1.0 or (abs(hash(vid)) % 100000) / 100000.0 < first_frame_ratio:
                    _, rel_img0, toon0 = frames[0]
                    item0 = {
                        "messages": [
                            {"role": "user", "content": IMG_PREFIX + prompt_no_prev},
                            {"role": "assistant", "content": wrap_assistant(toon0)},
                        ],
                        "images": [abs_image_path(rel_img0, repo_root)],
                    }
                    fout.write(json.dumps(item0, ensure_ascii=False) + "\n")
                    out_count += 1

            for t in range(1, len(frames)):
                _, rel_prev, toon_prev = frames[t - 1]
                _, rel_cur, toon_cur = frames[t]

                prompt = prompt_with_prev.replace("<<PREV_TOON>>", toon_prev)

                item = {
                    "messages": [
                        {"role": "user", "content": IMG_PREFIX + prompt},
                        {"role": "assistant", "content": wrap_assistant(toon_cur)},
                    ],
                    "images": [abs_image_path(rel_cur, repo_root)],
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_count += 1

    rel_out = path_for_json(repo_root, out_path)
    print(f"[done] {split_name}: wrote {out_count} samples -> {rel_out}")
    return out_count


PREV_TOON_MARKER = "Previous frame scene graph (TOON):\n"
PREV_TOON_END = "\n\nNow, generate"


def replace_prompt_in_jsonl(
    in_path: Path,
    out_path: Path,
    prompt_no_prev: str,
    prompt_with_prev: str,
) -> None:
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Replace prompt"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "messages" in item and len(item["messages"]) >= 2:
                msg0 = item["messages"][0]
                if msg0.get("role") != "user":
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                    continue
                old_human = msg0.get("content", "")
                if PREV_TOON_MARKER in old_human and PREV_TOON_END in old_human:
                    start = old_human.index(PREV_TOON_MARKER) + len(PREV_TOON_MARKER)
                    end = old_human.index(PREV_TOON_END, start)
                    prev_toon = old_human[start:end].strip()
                    msg0["content"] = IMG_PREFIX + prompt_with_prev.replace("<<PREV_TOON>>", prev_toon)
                else:
                    msg0["content"] = IMG_PREFIX + prompt_no_prev
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                continue
            conv = item.get("conversations", [])
            if len(conv) < 2:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                continue
            old_human = conv[0].get("value", "")
            if PREV_TOON_MARKER in old_human and PREV_TOON_END in old_human:
                start = old_human.index(PREV_TOON_MARKER) + len(PREV_TOON_MARKER)
                end = old_human.index(PREV_TOON_END, start)
                prev_toon = old_human[start:end].strip()
                new_human = IMG_PREFIX + prompt_with_prev.replace("<<PREV_TOON>>", prev_toon)
            else:
                new_human = IMG_PREFIX + prompt_no_prev
            conv[0]["value"] = new_human
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"[done] replace-prompt: {count} samples -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PVSG: TOON JSON -> Swift chat jsonl (messages + absolute images), prev-frame prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo_root",
        default=str(_REPO_ROOT_DEFAULT),
        help="SceneGraphVLM repo root (resolves relative --input_dir / --output_dir / files)",
    )
    p.add_argument(
        "--input_dir",
        action="append",
        default=[],
        help=(
            "Folder with annotation JSON from prepare_original_pvsg_sft.py "
            "(repeat for multiple dirs). Default if none given: "
            + DEFAULT_INPUT_DIR
        ),
    )
    p.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output JSONL files (created if missing)",
    )
    p.add_argument(
        "--train_json",
        default=TRAIN_JSON_NAME,
        help="Train split filename inside each --input_dir",
    )
    p.add_argument(
        "--test_json",
        default=TEST_JSON_NAME,
        help="Test split filename inside each --input_dir (from PVSG val split in prepare)",
    )
    p.add_argument(
        "--replace-prompt-only",
        action="store_true",
        help="Only rewrite human prompts in an existing JSONL (--input, --output).",
    )
    p.add_argument(
        "--input",
        type=str,
        default="",
        help="Input JSONL for --replace-prompt-only (repo-relative or absolute)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL for --replace-prompt-only",
    )
    p.add_argument(
        "--no-prev-prompt",
        type=str,
        default="",
        help="File with first-frame prompt; default built-in",
    )
    p.add_argument(
        "--with-prev-prompt",
        type=str,
        default="",
        help="File with <<PREV_TOON>> prompt; default built-in",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    prompt_no_prev = PROMPT_NO_PREV
    prompt_with_prev = PROMPT_WITH_PREV
    if args.no_prev_prompt:
        p = resolve_under_repo(repo_root, args.no_prev_prompt)
        prompt_no_prev = load_prompt_from_file(p)
        print(f"[prompt] no-prev loaded from {path_for_json(repo_root, p)}")
    if args.with_prev_prompt:
        p = resolve_under_repo(repo_root, args.with_prev_prompt)
        prompt_with_prev = load_prompt_from_file(p)
        print(f"[prompt] with-prev loaded from {path_for_json(repo_root, p)}")

    if args.replace_prompt_only:
        if not args.input or not args.output:
            print("Error: --replace-prompt-only requires --input and --output.", file=sys.stderr)
            sys.exit(1)
        in_p = resolve_under_repo(repo_root, args.input)
        out_p = resolve_under_repo(repo_root, args.output)
        if not in_p.is_file():
            print(f"Error: input not found: {in_p}", file=sys.stderr)
            sys.exit(1)
        replace_prompt_in_jsonl(in_p, out_p, prompt_no_prev, prompt_with_prev)
        print("Done")
        return

    input_dirs_arg: List[str] = args.input_dir if args.input_dir else [DEFAULT_INPUT_DIR]
    input_dirs = [resolve_under_repo(repo_root, d) for d in input_dirs_arg]

    for d in input_dirs:
        if not d.is_dir():
            print(f"Error: input_dir is not a directory: {d}", file=sys.stderr)
            sys.exit(1)

    output_dir = resolve_under_repo(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = collect_samples_from_dirs(input_dirs, args.train_json)
    test_samples = collect_samples_from_dirs(input_dirs, args.test_json)

    print(f"Repository root: {repo_root}")
    print(f"Input dirs ({len(input_dirs)}): " + ", ".join(path_for_json(repo_root, d) for d in input_dirs))
    print(f"Output dir: {path_for_json(repo_root, output_dir)}")
    print(f"Train samples (merged): {len(train_samples)} from {args.train_json}")
    print(f"Test samples (merged): {len(test_samples)} from {args.test_json}")

    if not train_samples and not test_samples:
        print(
            "Error: no samples found. Check --input_dir and that JSON files exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    if train_samples:
        make_jsonl_prev(
            train_samples,
            output_dir / OUT_TRAIN_JSONL,
            repo_root,
            "train",
            prompt_no_prev=prompt_no_prev,
            prompt_with_prev=prompt_with_prev,
        )
    else:
        print("[skip] No train samples; missing", args.train_json, "in all input dirs?")

    if test_samples:
        make_jsonl_prev(
            test_samples,
            output_dir / OUT_TEST_JSONL,
            repo_root,
            "test",
            prompt_no_prev=prompt_no_prev,
            prompt_with_prev=prompt_with_prev,
        )
    else:
        print("[skip] No test samples; missing", args.test_json, "in all input dirs?")

    print("Done")


if __name__ == "__main__":
    main()
