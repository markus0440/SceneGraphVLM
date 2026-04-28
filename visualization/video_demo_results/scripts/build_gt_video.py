#!/usr/bin/env python
"""
Build an MP4 that overlays the ground-truth scene graph on each frame of one video.

Expects JSONL in Swift format (messages + images) or sharegpt (conversations + image).

This file also holds shared helpers used by sibling scripts (import build_gt_video):
TOON parse/draw, JSONL filtering, metrics strip, PNG→MP4 (ffmpeg).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMO_ROOT = Path(_SCRIPT_DIR).resolve().parent
_DEFAULT_VIDEOS_OUT = _DEMO_ROOT / "videos_output"


# --- ffmpeg -------------------------------------------------------------------


def check_ffmpeg_or_raise() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found in PATH; install ffmpeg to write MP4.")
    return ff


def encode_png_pattern_to_mp4(ffmpeg_bin: str, pattern: str, fps: float, output_path: str) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {err[-4000:]}")


# --- paths / filtering --------------------------------------------------------


def sample_image_ref(sample: Dict[str, Any]) -> str:
    v = sample.get("image")
    if isinstance(v, str) and v.strip():
        return v
    imgs = sample.get("images") or []
    if imgs and isinstance(imgs[0], str):
        return imgs[0]
    return ""


def sample_frame_sort_key(sample: Dict[str, Any]):
    ref = sample_image_ref(sample)
    stem = Path(ref).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, ref)


def filter_video_frames_swift(samples: List[Dict[str, Any]], video_name: str) -> List[Dict[str, Any]]:
    video_samples = [s for s in samples if video_name in sample_image_ref(s)]
    video_samples.sort(key=sample_frame_sort_key)
    return video_samples


def filter_jsonl_by_video(input_jsonl: Path, output_jsonl: Path, video_name: str) -> tuple[int, int]:
    """Keep lines whose frame path contains video_name; sort by frame index in the filename stem."""
    n_in = n_out = 0
    kept: List[Dict[str, Any]] = []
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with input_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)
            if video_name in sample_image_ref(obj):
                kept.append(obj)
                n_out += 1
    kept.sort(key=sample_frame_sort_key)
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for obj in kept:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return n_in, n_out


def resolve_image_path(images_root: str, image_rel_path: str) -> str:
    if os.path.isabs(image_rel_path):
        return image_rel_path
    images_root_basename = os.path.basename(os.path.normpath(images_root))
    p = image_rel_path
    if p.startswith(images_root_basename + "/") or p.startswith(images_root_basename + "\\"):
        p = p[len(images_root_basename) + 1 :]
    return os.path.join(images_root, p)


# --- scene graph text ---------------------------------------------------------

# Action Genome rel_pairs: subj,[att],[spat],[cont],obj (bracket contents have no nested ])
_REL_PAIR_LINE_RE = re.compile(
    r"^\s*(\d+)\s*,\s*(\[[^\]]*\])\s*,\s*(\[[^\]]*\])\s*,\s*(\[[^\]]*\])\s*,\s*(\d+)\s*$"
)


def _bracket_inner(br: str) -> str:
    br = br.strip()
    if len(br) >= 2 and br[0] == "[" and br[-1] == "]":
        return br[1:-1].strip()
    return br


def _rel_pair_pred(att_b: str, spat_b: str, cont_b: str) -> str:
    chunks = []
    for inner in (_bracket_inner(att_b), _bracket_inner(spat_b), _bracket_inner(cont_b)):
        if not inner:
            continue
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if parts:
            chunks.append("/".join(parts))
    return "|".join(chunks) if chunks else "rel"


_OBJ_HEADER_RE = re.compile(r"^(?:obj)?\[(\d+)\]\{([^}]*)\}", re.IGNORECASE)
_REL_TRIPLE_HEADER_RE = re.compile(
    r"^(?:rel)?\[(\d+)\]\{subj,pred,obj\}",
    re.IGNORECASE,
)


def _extract_obj_schema(line: str) -> Optional[List[str]]:
    m = _OBJ_HEADER_RE.match(line.strip())
    if not m:
        return None
    return [x.strip().lower() for x in m.group(2).split(",") if x.strip()]


def _scan_obj_rel_headers(lines: List[str]) -> tuple[Optional[int], Optional[int], Optional[str], Optional[List[str]]]:
    obj_header_idx = None
    rel_header_idx = None
    rel_mode: Optional[str] = None
    obj_schema: Optional[List[str]] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        schema = _extract_obj_schema(stripped)
        if schema:
            obj_header_idx = i
            obj_schema = schema
        if _REL_TRIPLE_HEADER_RE.match(stripped):
            rel_header_idx = i
            rel_mode = "triple"
        if stripped.startswith("rel_pairs[") and "{subj,attention,spatial,contacting,obj}" in stripped:
            rel_header_idx = i
            rel_mode = "pairs"
    return obj_header_idx, rel_header_idx, rel_mode, obj_schema


def _try_repair_missing_obj_header_block(answer_text: str) -> Optional[str]:
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    obj_i, rel_i, rel_mode, _ = _scan_obj_rel_headers(lines)
    if rel_i is None or rel_mode is None or obj_i is not None:
        return None
    raw_obj_lines = lines[:rel_i]
    rel_and_after = lines[rel_i:]
    fixed_rows: List[str] = []
    next_id = 1
    for raw in raw_obj_lines:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 5:
            continue
        if parts[0].isdigit() and len(parts) >= 6:
            try:
                oid = int(parts[0])
                x1, y1, x2, y2 = [int(v) for v in parts[-4:]]
            except ValueError:
                continue
            head = parts[:-4]
            fixed_rows.append(",".join([*head, str(x1), str(y1), str(x2), str(y2)]))
            next_id = max(next_id, oid + 1)
        elif not parts[0].isdigit():
            try:
                x1, y1, x2, y2 = [int(v) for v in parts[-4:]]
            except ValueError:
                continue
            head = [str(next_id), *parts[:-4]]
            fixed_rows.append(",".join([*head, str(x1), str(y1), str(x2), str(y2)]))
            next_id += 1
    if not fixed_rows:
        return None
    k = len(fixed_rows)
    header = f"obj[{k}]{{id,name,x1,y1,x2,y2}}:"
    return "\n".join([header, *fixed_rows, *rel_and_after])


def scene_graph_output_needs_obj_header_retry(answer_text: str) -> bool:
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    obj_i, rel_i, rel_mode, _ = _scan_obj_rel_headers(lines)
    return rel_i is not None and rel_mode is not None and obj_i is None


def repair_scene_graph_text_if_needed(answer_text: str) -> str:
    fixed = _try_repair_missing_obj_header_block(answer_text)
    return fixed if fixed is not None else answer_text


def parse_scene_graph_output(answer_text: str) -> dict:
    lines = [line.rstrip("\n") for line in answer_text.splitlines()]
    lines = [line for line in lines if line.strip()]
    obj_header_idx, rel_header_idx, rel_mode, obj_schema = _scan_obj_rel_headers(lines)
    if obj_header_idx is None and rel_header_idx is not None and rel_mode is not None:
        repaired = _try_repair_missing_obj_header_block(answer_text)
        if repaired:
            lines = [line.strip() for line in repaired.splitlines() if line.strip()]
            obj_header_idx, rel_header_idx, rel_mode, obj_schema = _scan_obj_rel_headers(lines)
    if obj_header_idx is None or rel_header_idx is None or rel_mode is None:
        raise ValueError("Could not find obj[...] or rel / rel_pairs headers in model response")
    obj_lines = lines[obj_header_idx + 1 : rel_header_idx]
    objects = []
    for raw_line in obj_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) < 6:
            continue
        try:
            obj_id = int(parts[0])
            x1, y1, x2, y2 = [int(v) for v in parts[-4:]]
        except ValueError:
            continue
        name_idx = 1
        if obj_schema and "name" in obj_schema:
            schema_idx = obj_schema.index("name")
            if schema_idx < len(parts):
                name_idx = schema_idx
        name = parts[name_idx] if name_idx < len(parts) else f"obj_{obj_id}"
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        objects.append({"id": obj_id, "label": name, "bbox": [x1, y1, x2, y2]})
    rel_lines = lines[rel_header_idx + 1 :]
    relations = []
    if rel_mode == "triple":
        for raw_line in rel_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = [part.strip() for part in stripped.split(",")]
            if len(parts) != 3:
                continue
            subj_str, pred, obj_str = parts
            try:
                subj_id = int(subj_str)
                obj_id = int(obj_str)
            except ValueError:
                continue
            relations.append({"sub": subj_id, "obj": obj_id, "pred": pred})
    else:
        for raw_line in rel_lines:
            m = _REL_PAIR_LINE_RE.match(raw_line.strip())
            if not m:
                continue
            subj_id, att_b, spat_b, cont_b, obj_id = (
                int(m.group(1)),
                m.group(2),
                m.group(3),
                m.group(4),
                int(m.group(5)),
            )
            pred = _rel_pair_pred(att_b, spat_b, cont_b)
            relations.append({"sub": subj_id, "obj": obj_id, "pred": pred})
    return {"objects": objects, "relations": relations}


def scene_graph_output_is_parseable(answer_text: str) -> bool:
    try:
        parse_scene_graph_output(answer_text)
        return True
    except ValueError:
        return False


def draw_scene_graph(
    image: str,
    answer_text: str,
    max_width: Optional[int] = None,
    title: Optional[str] = None,
    title_wrap_long: bool = False,
) -> Image.Image:
    if isinstance(image, (str, bytes, bytearray)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    original_width, original_height = img.size
    scale = 1.0
    if max_width is not None and original_width > max_width:
        scale = max_width / float(original_width)
        img = img.resize((int(original_width * scale), int(original_height * scale)), Image.LANCZOS)
    display_width, display_height = img.size
    try:
        scene_graph = parse_scene_graph_output(answer_text)
    except ValueError:
        scene_graph = {"objects": [], "relations": []}
    dpi = 100
    fig_width = display_width / dpi
    fig_height = display_height / dpi
    if title:
        if title_wrap_long:
            max_chars_per_line = max(20, int(display_width / 8))
            num_lines = max(1, (len(title) + max_chars_per_line - 1) // max_chars_per_line)
            base_title_height = 0.4
            line_height = 0.15
            title_height_inches = base_title_height + (num_lines - 1) * line_height
            total_height = fig_height + title_height_inches
            fig = plt.figure(figsize=(fig_width, total_height), dpi=dpi, facecolor="white")
            title_ax = fig.add_axes(
                [0, (fig_height / total_height), 1, title_height_inches / total_height],
                facecolor="white",
            )
            base_fontsize = 12
            if len(title) > 100:
                base_fontsize = 10
            if len(title) > 150:
                base_fontsize = 9
            if len(title) > 200:
                base_fontsize = 8
            wrapped_title = "\n".join(textwrap.wrap(title, width=max_chars_per_line))
            title_ax.text(
                0.5,
                0.5,
                wrapped_title,
                fontsize=base_fontsize,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
            )
            title_ax.axis("off")
            ax = fig.add_axes([0, 0, 1, fig_height / total_height], facecolor="white")
        else:
            title_height_inches = 0.5
            total_height = fig_height + title_height_inches
            fig = plt.figure(figsize=(fig_width, total_height), dpi=dpi, facecolor="white")
            title_ax = fig.add_axes(
                [0, (fig_height / total_height), 1, title_height_inches / total_height],
                facecolor="white",
            )
            title_ax.text(
                0.5, 0.5, title, fontsize=14, fontweight="bold", ha="center", va="center", color="black"
            )
            title_ax.axis("off")
            ax = fig.add_axes([0, 0, 1, fig_height / total_height], facecolor="white")
    else:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], facecolor="white")
    ax.imshow(img)
    ax.axis("off")
    num_objects = len(scene_graph["objects"])
    if num_objects > 0:
        palette = plt.cm.tab20(np.linspace(0, 1, max(20, num_objects)))
        centers = {}
        for i, obj in enumerate(scene_graph["objects"]):
            x1, y1, x2, y2 = obj["bbox"]
            x1s, y1s = int(round(x1 * scale)), int(round(y1 * scale))
            x2s, y2s = int(round(x2 * scale)), int(round(y2 * scale))
            box_width = max(1, x2s - x1s)
            box_height = max(1, y2s - y1s)
            color = palette[i % len(palette)]
            ax.add_patch(
                patches.Rectangle(
                    (x1s, y1s), box_width, box_height, linewidth=2, edgecolor=color, facecolor="none"
                )
            )
            fontsize = 7 if box_height >= 14 else (6 if box_height >= 9 else 5)
            ax.text(
                x1s + 2,
                y1s + 2,
                f"{obj['id']}: {obj['label']}",
                fontsize=fontsize,
                color="white",
                va="top",
                bbox=dict(facecolor=color, alpha=0.75, pad=0.6),
            )
            centers[obj["id"]] = ((x1s + x2s) / 2, (y1s + y2s) / 2)
        for relation in scene_graph["relations"]:
            subj_id, obj_id, predicate = relation["sub"], relation["obj"], relation["pred"]
            if subj_id in centers and obj_id in centers:
                subj_x, subj_y = centers[subj_id]
                obj_x, obj_y = centers[obj_id]
                ax.add_patch(
                    FancyArrowPatch(
                        (subj_x, subj_y),
                        (obj_x, obj_y),
                        arrowstyle="->",
                        mutation_scale=14,
                        color="yellow",
                        linewidth=2,
                        alpha=0.9,
                    )
                )
                mid_x, mid_y = (subj_x + obj_x) / 2, (subj_y + obj_y) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    predicate,
                    fontsize=6,
                    color="yellow",
                    bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.2"),
                )
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buffer).convert("RGB")


# --- conversations ------------------------------------------------------------


def get_prediction_text(sample: dict) -> str:
    v = sample.get("predict")
    if isinstance(v, str) and v.strip():
        return v
    for m in sample.get("messages") or []:
        if m.get("role") == "assistant":
            c = (m.get("content") or "").strip()
            if c:
                return c
    conv = sample.get("conversations", [])
    pred_msgs = [c["value"] for c in conv if c.get("from") == "predict"]
    if pred_msgs:
        return pred_msgs[0]
    return ""


def get_gt_annotation(sample: dict) -> str:
    msgs = sample.get("messages") or []
    asst = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
    if asst:
        return "\n".join(asst)
    conv = sample.get("conversations", [])
    gpt_msgs = [c["value"] for c in conv if c.get("from") == "gpt"]
    if gpt_msgs:
        return "\n".join(gpt_msgs)
    if conv:
        return conv[-1]["value"]
    return ""


# --- metrics table ------------------------------------------------------------


def create_metrics_table(metrics: Dict, width: int, gen_time: Optional[float] = None) -> Image.Image:
    headers = [
        ("LLM time", (0.8, 0.95, 0.8)),
        ("IoU", (0.9, 0.85, 0.95)),
        ("Obj P", (1.0, 0.9, 0.8)),
        ("Obj R", (1.0, 0.9, 0.8)),
        ("Obj F1", (1.0, 0.9, 0.8)),
        ("Rel P", (0.9, 0.85, 0.95)),
        ("Rel R", (0.9, 0.85, 0.95)),
        ("Rel F1", (0.9, 0.85, 0.95)),
        ("Qwen score", (1.0, 0.85, 0.9)),
    ]
    num_cols = len(headers)
    col_width = width // num_cols
    font_size = 10
    row_height = int(font_size * 2.2)
    table_height = row_height * 2
    table_img = Image.new("RGB", (width, table_height), color="white")
    draw = ImageDraw.Draw(table_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_data = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            font_data = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
            font_data = ImageFont.load_default()
    x = 0
    for header_name, bg_color in headers:
        bg_rgb = tuple(int(c * 255) for c in bg_color)
        draw.rectangle([x, 0, x + col_width, row_height], fill=bg_rgb, outline="gray", width=1)
        bbox = draw.textbbox((0, 0), header_name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (col_width - tw) // 2, (row_height - th) // 2), header_name, fill="black", font=font)
        x += col_width

    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    if metrics is None:
        metrics = {}
    vllm_time = safe_float(gen_time, 0.0)
    iou = safe_float(metrics.get("bbox_mean_iou_matched"), 0.0)
    obj_p = safe_float(metrics.get("qwen_precision_objects"), 0.0)
    obj_r = safe_float(metrics.get("qwen_recall_objects"), 0.0)
    obj_f1 = safe_float(metrics.get("qwen_f1_objects"), 0.0)
    rel_p = safe_float(metrics.get("qwen_precision_relations"), 0.0)
    rel_r = safe_float(metrics.get("qwen_recall_relations"), 0.0)
    rel_f1 = safe_float(metrics.get("qwen_f1_relations"), 0.0)
    qwen_score = safe_float(metrics.get("qwen_sgg_score", metrics.get("qwen_overall_score")), 0.0)
    values = [
        f"{vllm_time:.3f}",
        f"{iou:.3f}",
        f"{obj_p:.3f}",
        f"{obj_r:.3f}",
        f"{obj_f1:.3f}",
        f"{rel_p:.3f}",
        f"{rel_r:.3f}",
        f"{rel_f1:.3f}",
        f"{qwen_score:.3f}",
    ]
    x = 0
    for value in values:
        draw.rectangle([x, row_height, x + col_width, table_height], fill="white", outline="gray", width=1)
        bbox = draw.textbbox((0, 0), value, font=font_data)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x + (col_width - tw) // 2, row_height + (row_height - th) // 2),
            value,
            fill="black",
            font=font_data,
        )
        x += col_width
    return table_img


def create_qwen_eval_metrics_table(metrics: Dict, width: int) -> Image.Image:
    """
    Per-frame strip: seven metrics from eval per-sample jsonl (object AP@50, R, F1; rel P, R, F1; SGG).
    Values use qwen_* keys from eval; headers are neutral (no judge name).
    """
    headers = [
        ("AP@50", (1.0, 0.9, 0.8)),
        ("Obj R", (1.0, 0.9, 0.8)),
        ("Obj F1", (1.0, 0.9, 0.8)),
        ("Rel P", (0.9, 0.85, 0.95)),
        ("Rel R", (0.9, 0.85, 0.95)),
        ("Rel F1", (0.9, 0.85, 0.95)),
        ("SGG", (1.0, 0.85, 0.9)),
    ]
    num_cols = len(headers)
    col_width = max(1, width // num_cols)
    font_size = 9
    row_height = int(font_size * 2.3)
    table_height = row_height * 2
    table_img = Image.new("RGB", (width, table_height), color="white")
    draw = ImageDraw.Draw(table_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_data = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            font_data = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
            font_data = ImageFont.load_default()

    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    if metrics is None:
        metrics = {}
    obj_p = safe_float(metrics.get("qwen_precision_objects"), 0.0)
    obj_r = safe_float(metrics.get("qwen_recall_objects"), 0.0)
    obj_f1 = safe_float(metrics.get("qwen_f1_objects"), 0.0)
    rel_p = safe_float(metrics.get("qwen_precision_relations"), 0.0)
    rel_r = safe_float(metrics.get("qwen_recall_relations"), 0.0)
    rel_f1 = safe_float(metrics.get("qwen_f1_relations"), 0.0)
    sgg = safe_float(metrics.get("qwen_sgg_score", metrics.get("qwen_overall_score")), 0.0)
    values = [
        f"{obj_p:.3f}",
        f"{obj_r:.3f}",
        f"{obj_f1:.3f}",
        f"{rel_p:.3f}",
        f"{rel_r:.3f}",
        f"{rel_f1:.3f}",
        f"{sgg:.3f}",
    ]

    x = 0
    for header_name, bg_color in headers:
        bg_rgb = tuple(int(c * 255) for c in bg_color)
        draw.rectangle([x, 0, x + col_width, row_height], fill=bg_rgb, outline="gray", width=1)
        bbox = draw.textbbox((0, 0), header_name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + max(0, (col_width - tw) // 2), (row_height - th) // 2), header_name, fill="black", font=font)
        x += col_width

    x = 0
    for value in values:
        draw.rectangle([x, row_height, x + col_width, table_height], fill="white", outline="gray", width=1)
        bbox = draw.textbbox((0, 0), value, font=font_data)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x + max(0, (col_width - tw) // 2), row_height + (row_height - th) // 2),
            value,
            fill="black",
            font=font_data,
        )
        x += col_width
    return table_img


def create_gen_time_only_table(width: int, gen_time: Optional[float]) -> Image.Image:
    headers = [("LLM time (sec)", (0.8, 0.95, 0.8))]
    font_size = 10
    row_height = int(font_size * 2.2)
    table_height = row_height * 2
    table_img = Image.new("RGB", (width, table_height), color="white")
    draw = ImageDraw.Draw(table_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_data = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
        font_data = font
    draw.rectangle([0, 0, width, row_height], fill=tuple(int(c * 255) for c in headers[0][1]), outline="gray", width=1)
    h = headers[0][0]
    bbox = draw.textbbox((0, 0), h, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (row_height - th) // 2), h, fill="black", font=font)
    val = f"{float(gen_time):.3f}" if gen_time is not None else "—"
    draw.rectangle([0, row_height, width, table_height], fill="white", outline="gray", width=1)
    bbox = draw.textbbox((0, 0), val, font=font_data)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, row_height + (row_height - th) // 2), val, fill="black", font=font_data)
    return table_img


def combine_image_with_metrics_table(scene_graph_img: Image.Image, metrics_table: Image.Image) -> Image.Image:
    sg_width, sg_height = scene_graph_img.size
    table_width, table_height = metrics_table.size
    if table_width != sg_width:
        metrics_table = metrics_table.resize((sg_width, table_height), Image.LANCZOS)
        table_width, table_height = sg_width, metrics_table.size[1]
    total_height = sg_height + table_height
    combined = Image.new("RGB", (sg_width, total_height), color="white")
    combined.paste(scene_graph_img, (0, 0))
    combined.paste(metrics_table, (0, sg_height))
    return combined


# --- jsonl / video ------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: float = 10.0) -> None:
    if not frames:
        raise ValueError("No frames provided")
    height, width = frames[0].shape[:2]
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (height, width):
            frame_pil = Image.fromarray(frame).resize((width, height), Image.LANCZOS)
            frames[i] = np.array(frame_pil)
    ff = check_ffmpeg_or_raise()
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            Image.fromarray(frame).save(frame_path, "PNG")
        pattern = os.path.join(temp_dir, "frame_%06d.png")
        encode_png_pattern_to_mp4(ff, pattern, fps, output_path)


# --- CLI: GT video ------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a GT scene-graph video for one video id from a JSONL annotation file."
    )
    p.add_argument(
        "--annotation-file",
        "--input-file",
        dest="annotation_file",
        type=str,
        required=True,
        help="Path to JSONL annotations (e.g. train/test split).",
    )
    p.add_argument(
        "--video-name",
        type=str,
        required=True,
        help="Video id: substring of the frame path (e.g. 0004_11566980553).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_VIDEOS_OUT),
        help=f"Directory for output (default: {_DEFAULT_VIDEOS_OUT}).",
    )
    p.add_argument(
        "--output-filename",
        type=str,
        default="",
        help="Output MP4 basename (default: {video_name}_GT.mp4, e.g. 0004_11566980553_GT.mp4).",
    )
    p.add_argument(
        "--output-parent",
        type=str,
        default=None,
        help="If set: write output-filename directly here (no video-name subfolder).",
    )
    p.add_argument(
        "--images-root",
        type=str,
        default="",
        help="Root for relative paths in images/image; omit if paths in JSONL are absolute.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output MP4 frame rate (passed to ffmpeg -framerate).",
    )
    p.add_argument("--max-width", type=int, default=1920, help="0 = do not resize by width.")
    args = p.parse_args()
    max_width = args.max_width if args.max_width > 0 else None
    fps = float(args.fps)
    if fps <= 0:
        print("ERROR: fps must be positive", file=sys.stderr)
        sys.exit(1)

    output_filename = args.output_filename.strip() or f"{args.video_name}_GT.mp4"
    if not output_filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        output_filename += ".mp4"

    all_samples = load_jsonl(args.annotation_file)
    video_samples = filter_video_frames_swift(all_samples, args.video_name)
    if not video_samples:
        print(
            f"ERROR: no samples for video {args.video_name!r} in {args.annotation_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.output_parent:
        out_dir = Path(args.output_parent)
        out_path = out_dir / output_filename
    else:
        out_dir = Path(args.output_dir) / args.video_name
        out_path = out_dir / output_filename
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list = []
    skipped_missing_image = 0
    skipped_no_gt = 0
    for i, sample in enumerate(video_samples):
        print(f"Frame {i + 1}/{len(video_samples)}...", end="\r", flush=True)
        ref = sample_image_ref(sample)
        if not ref:
            skipped_missing_image += 1
            continue
        ip = resolve_image_path(args.images_root, ref)
        if not os.path.isfile(ip):
            print(f"\nWARNING: image file not found: {ip}")
            skipped_missing_image += 1
            continue
        gt = get_gt_annotation(sample)
        if not gt or not gt.strip():
            skipped_no_gt += 1
            continue
        try:
            vis = draw_scene_graph(ip, gt, max_width=max_width, title="GT", title_wrap_long=False)
        except Exception as e:
            print(f"\nERROR frame {i}: {e}")
            continue
        frames.append(np.array(vis))

    summary = f"\nWrote {len(frames)} frames -> {out_path}"
    if skipped_missing_image or skipped_no_gt:
        summary += (
            f" | skipped: {skipped_missing_image} (no image path or missing file), "
            f"{skipped_no_gt} (no GT text in sample)"
        )
    else:
        summary += " | skipped: none"
    print(summary)
    if not frames:
        sys.exit(1)
    create_video_from_frames(frames, str(out_path), fps=fps)
    print("Done.")


if __name__ == "__main__":
    main()
