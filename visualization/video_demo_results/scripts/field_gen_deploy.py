#!/usr/bin/env python3
r"""
Field deployment: GEN-style temporal inference for PVSG (obj + rel triples).

Embedded prompts match `datasets/.../PVSG_json/pvsg_all_data_gt_prompt` GEN-style rows:
first frame + chain with header ``Previous frame scene graph (TOON):``. The previous
frame slot is filled with the model's own last prediction (like GEN_prompt
``--prev-source model``); TOON is pasted raw after the header, as in PVSG training.

Does not import infer_swift_gen_prompt.py.

Writes:
  <results-root>/predicts_and_metrics/field_deploy/<stem>/<run_name>.jsonl
  <results-root>/videos_output/field_deploy/<stem>/<run_name>.mp4

Dependencies: ms-swift (+ vllm optional), torch, tqdm, ffmpeg for real video files, and
build_gt_video for MP4 overlay unless --no-video.
"""
from __future__ import annotations

PVSG_HEAD = "<image>\nGenerate a structured scene graph for an image of size (640 x 480) using the following text format.\n\nOutput Format:\n\n<answer>\nobj[N]{id,name,x1,y1,x2,y2}:\n  id,name,x1,y1,x2,y2\n  ...\nrel[M]{subj,pred,obj}:\n  subj,pred,obj\n  ...\n\n</answer>\nGuidelines:\n- Objects:\n  - Use integer IDs starting from 1 in the id field (e.g., 1, 2, 3).\n  - The name must be the object category name (e.g., person, umbrella).\n  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.\n  - Include all visible objects, even if they have no relationships.\n\n- Relationships:\n  - Represent interactions using integer object IDs in subj and obj.\n  - pred is the relationship type (string), such as in-front-of, attached-to, beside.\n  - Omit relationships for objects that do not participate in any interaction.\n\nExample output:\n<answer>\nobj[7]{id,name,x1,y1,x2,y2}:\n  1,person,281,272,524,438\n  2,umbrella,273,123,640,434\n  3,house,0,88,262,426\n  4,window-other,163,262,195,294\n  5,tree-merged,0,0,640,440\n  6,sky-other-merged,0,0,459,123\n  7,building-other-merged,537,164,640,291\nrel[5]{subj,pred,obj}:\n  1,in-front-of,5\n  3,attached-to,4\n  4,hanging-from,3\n  5,beside,3\n  6,over,5\n</answer>\n"
PVSG_FIRST_TAIL = "Now, generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags.\n"
PVSG_CHAIN_PREFIX = "<image>\nGenerate a structured scene graph for an image of size (640 x 480) using the following text format.\n\nOutput Format:\n\n<answer>\nobj[N]{id,name,x1,y1,x2,y2}:\n  id,name,x1,y1,x2,y2\n  ...\nrel[M]{subj,pred,obj}:\n  subj,pred,obj\n  ...\n\n</answer>\nGuidelines:\n- Objects:\n  - Use integer IDs starting from 1 in the id field (e.g., 1, 2, 3).\n  - The name must be the object category name (e.g., person, umbrella).\n  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.\n  - Include all visible objects, even if they have no relationships.\n\n- Relationships:\n  - Represent interactions using integer object IDs in subj and obj.\n  - pred is the relationship type (string), such as in-front-of, attached-to, beside.\n  - Omit relationships for objects that do not participate in any interaction.\n\nExample output:\n<answer>\nobj[7]{id,name,x1,y1,x2,y2}:\n  1,person,281,272,524,438\n  2,umbrella,273,123,640,434\n  3,house,0,88,262,426\n  4,window-other,163,262,195,294\n  5,tree-merged,0,0,640,440\n  6,sky-other-merged,0,0,459,123\n  7,building-other-merged,537,164,640,291\nrel[5]{subj,pred,obj}:\n  1,in-front-of,5\n  3,attached-to,4\n  4,hanging-from,3\n  5,beside,3\n  6,over,5\n</answer>\nYou are also given the previous frame's ground-truth scene graph in TOON format.\nUse it as temporal context, but rely primarily on the current image.\nImportant:\n- Include all objects visible in the CURRENT image, even if they did not exist in the previous graph.\n- Do NOT include objects that are NOT visible in the current image, even if they exist in the previous graph.\n- Output ONLY the complete scene graph for the CURRENT image, using the TOON structure from the Output Format above, inside one <answer>...</answer> block.\n\nPrevious frame scene graph (TOON):\n"
PVSG_CHAIN_SUFFIX = "\n\nNow, generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags.\n"


import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

try:
    from swift.model import get_model_processor, get_processor
    from swift.template import get_template
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine, VllmEngine
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "ms-swift is required (same as GEN_prompt infer). Install e.g.:\n"
        "  pip install 'ms-swift[llm]' -U\n"
        "  pip install vllm   # for --infer-backend vllm\n"
        f"Import error: {e}"
    ) from e

_THINK_END = "</redacted_thinking>"
_SG_OBJ_HDR = re.compile(r"^\s*obj\[\d+\]\{id,name", re.MULTILINE)
_SG_REL_HDR = re.compile(r"^\s*rel\[\d+\]\{subj,pred,obj\}", re.MULTILINE)
_SG_REL_PAIRS_HDR = re.compile(r"^\s*rel_pairs\[\d+\]\{subj,attention", re.MULTILINE)


def strip_thinking_suffix(text: str) -> str:
    s = text.strip()
    while _THINK_END in s:
        idx = s.find(_THINK_END)
        prefix = s[:idx]
        suffix = s[idx + len(_THINK_END) :].lstrip()
        if _SG_OBJ_HDR.search(prefix) or _SG_REL_HDR.search(prefix) or _SG_REL_PAIRS_HDR.search(prefix):
            s = (prefix + "\n" + suffix).strip() if suffix else prefix.strip()
            break
        if not suffix:
            return prefix.strip() if prefix.strip() else ""
        s = suffix
    return s


def truncate_first_answer_block(text: str) -> str:
    s = text.strip()
    k = s.find("</answer>")
    if k != -1:
        return s[: k + len("</answer>")].strip()
    if "<answer>" in s:
        return s.rstrip() + "\n</answer>\n"
    return s


def finalize_vl_prediction(raw: str) -> str:
    return truncate_first_answer_block(strip_thinking_suffix(raw))


def has_obj_header(text: str) -> bool:
    return bool(_SG_OBJ_HDR.search(text or ""))


def parse_torch_dtype(name: str) -> torch.dtype:
    d = getattr(torch, name, None)
    if d is None or not isinstance(d, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name}")
    return d


def get_user_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    raise ValueError("No user message in sample")


def set_user_text(sample: Dict[str, Any], new_text: str) -> None:
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            msg["content"] = new_text
            return
    raise ValueError("No user message in sample")


def get_assistant_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    return ""


def apply_wh_size(prompt: str, w: int, h: int) -> str:
    return prompt.replace("(640 x 480)", f"({w} x {h})")


def build_prompt_first(head: str, first_tail: str, w: int, h: int) -> str:
    return apply_wh_size(head, w, h) + apply_wh_size(first_tail, w, h)


def build_prompt_chain(
    chain_prefix: str,
    chain_suffix: str,
    prev_prediction: str,
    w: int,
    h: int,
    *,
    wrap_prev_in_answer: bool,
) -> str:
    """PVSG legacy header uses raw TOON after ``Previous frame scene graph (TOON):`` (no extra <answer> wrap)."""
    prefix = apply_wh_size(chain_prefix, w, h)
    suffix = apply_wh_size(chain_suffix, w, h)
    prev = (prev_prediction or "").strip()
    if wrap_prev_in_answer and prev and not prev.lstrip().lower().startswith("<answer"):
        prev = f"<answer>\n{prev}\n</answer>"
    return prefix + prev + suffix


def build_swift_engine(
    *,
    model_id_or_path: str,
    infer_backend: str,
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    template_type: Optional[str],
    enable_thinking: bool,
    response_prefix: Optional[str],
    torch_dtype: torch.dtype,
) -> Union[VllmEngine, TransformersEngine]:
    if infer_backend == "vllm":
        processor = get_processor(
            model_id_or_path=model_id_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        tmpl = get_template(
            processor,
            template_type=template_type,
            enable_thinking=enable_thinking,
            response_prefix=response_prefix,
        )
        return VllmEngine(
            model_id_or_path,
            template=tmpl,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 1},
        )
    if infer_backend == "transformers":
        model, processor = get_model_processor(
            model_id_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        tmpl = get_template(
            processor,
            template_type=template_type,
            enable_thinking=enable_thinking,
            response_prefix=response_prefix,
        )
        return TransformersEngine(model, template=tmpl, max_batch_size=max(1, batch_size))
    raise ValueError(f"Unknown infer_backend: {infer_backend}")


def check_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found in PATH.")
    return ff


def extract_frames_video(video: Path, out_dir: Path, fps: float) -> List[Path]:
    check_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        check_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted to {out_dir}")
    return frames


def list_frames_directory(d: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    if not files:
        raise RuntimeError(f"No images in {d}")
    return files


def image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def synthesize_samples(frame_paths: List[Path], w: int, h: int) -> List[Dict[str, Any]]:
    first_u = build_prompt_first(PVSG_HEAD, PVSG_FIRST_TAIL, w, h)
    out: List[Dict[str, Any]] = []
    for p in frame_paths:
        out.append(
            {
                "messages": [
                    {"role": "user", "content": first_u},
                    {"role": "assistant", "content": ""},
                ],
                "images": [str(p.resolve())],
            }
        )
    return out


def run_realtime_infer(
    *,
    samples: List[Dict[str, Any]],
    engine: Union[VllmEngine, TransformersEngine],
    fallback_engine: Optional[Union[VllmEngine, TransformersEngine]],
    out_path: Path,
    run_name: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    no_stop_on_answer_close: bool,
    stop_sequence: Optional[List[str]],
    w: int,
    h: int,
) -> None:
    stop_list: Optional[List[str]] = None
    if no_stop_on_answer_close:
        stop_list = list(stop_sequence) if stop_sequence else None
    else:
        stop_list = ["</answer>"]
        if stop_sequence:
            stop_list = stop_list + list(stop_sequence)
    req_cfg = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, stop=stop_list or [])

    max_wave = len(samples)
    predictions: List[Optional[str]] = [None] * len(samples)
    gen_times: List[Optional[float]] = [None] * len(samples)
    messages_used: List[Optional[List[Any]]] = [None] * len(samples)
    user_text_used: List[Optional[str]] = [None] * len(samples)
    pred_errors: List[Optional[str]] = [None] * len(samples)
    prev_pred: Optional[str] = None

    pbar = tqdm(total=len(samples), desc="field GEN infer", unit="frame")
    for wave_idx in range(max_wave):
        gidx = wave_idx
        sample = samples[gidx]
        wave_samples: List[Dict[str, Any]] = []
        wave_paths: List[str] = []
        wave_gidx: List[int] = []

        try:
            img_path = sample["images"][0]
            if not os.path.isfile(img_path):
                raise FileNotFoundError(img_path)

            if wave_idx == 0:
                run_sample = sample
            else:
                run_sample = copy.deepcopy(sample)
                new_user = build_prompt_chain(
                    PVSG_CHAIN_PREFIX,
                    PVSG_CHAIN_SUFFIX,
                    prev_pred or "",
                    w,
                    h,
                    wrap_prev_in_answer=False,
                )
                set_user_text(run_sample, new_user)

            messages_used[gidx] = copy.deepcopy(run_sample.get("messages"))
            user_text_used[gidx] = get_user_text(run_sample)
            wave_samples.append(run_sample)
            wave_paths.append(img_path)
            wave_gidx.append(gidx)
        except Exception as e:
            messages_used[gidx] = copy.deepcopy(sample.get("messages"))
            user_text_used[gidx] = get_user_text(sample) if sample.get("messages") else ""
            pred_errors[gidx] = str(e)
            predictions[gidx] = ""
            gen_times[gidx] = None
            prev_pred = ""
            pbar.update(1)
            continue

        bs = max(1, batch_size)
        for b0 in range(0, len(wave_samples), bs):
            b_samples = wave_samples[b0 : b0 + bs]
            b_paths = wave_paths[b0 : b0 + bs]
            b_gidx = wave_gidx[b0 : b0 + bs]

            requests: List[InferRequest] = []
            meta: List[Tuple[int, Dict[str, Any]]] = []
            for run_sample, path, gi in zip(b_samples, b_paths, b_gidx):
                try:
                    user = get_user_text(run_sample)
                    requests.append(InferRequest(messages=[{"role": "user", "content": user}], images=[path]))
                    meta.append((gi, samples[gi]))
                except Exception as e:
                    pred_errors[gi] = str(e)
                    predictions[gi] = ""
                    gen_times[gi] = None
                    prev_pred = ""
                    pbar.update(1)

            if not requests:
                continue

            t0 = time.perf_counter()
            try:
                resp_list = engine.infer(requests, req_cfg)
            except Exception as e:
                for gi, samp in meta:
                    pred_errors[gi] = f"engine.infer failed: {e}"
                    predictions[gi] = ""
                    gen_times[gi] = None
                    prev_pred = ""
                    pbar.update(1)
                continue

            elapsed = time.perf_counter() - t0
            per = elapsed / max(1, len(resp_list))

            for (gi, samp), resp in zip(meta, resp_list):
                raw = resp.choices[0].message.content or ""
                pred_clean = finalize_vl_prediction(raw)

                if fallback_engine is not None and not has_obj_header(pred_clean):
                    try:
                        fb_req = InferRequest(
                            messages=[
                                {
                                    "role": "user",
                                    "content": user_text_used[gi] or get_user_text(samp),
                                }
                            ],
                            images=[samp["images"][0]],
                        )
                        fb_resp = fallback_engine.infer([fb_req], req_cfg)[0]
                        fb_pred = finalize_vl_prediction(fb_resp.choices[0].message.content or "")
                        if has_obj_header(fb_pred):
                            pred_clean = fb_pred
                    except Exception:
                        pass

                predictions[gi] = pred_clean
                gen_times[gi] = per
                prev_pred = pred_clean
                pbar.update(1)

    pbar.close()

    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            pred = predictions[idx] or ""
            user_prompt = user_text_used[idx] or get_user_text(sample)
            record: Dict[str, Any] = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": pred},
                ],
                "images": sample.get("images"),
                "content": "",
                "predict": pred,
                "model_name": run_name,
                "gen_time_sec": gen_times[idx],
            }
            if pred_errors[idx]:
                record["predict_error"] = pred_errors[idx]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp.replace(out_path)


def render_pred_video(
    jsonl_path: Path,
    out_mp4: Path,
    fps: float,
    max_width: int,
    title: str,
) -> None:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    import numpy as np
    from build_gt_video import create_video_from_frames, draw_scene_graph, get_prediction_text

    rows: List[Dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    frames: List[Any] = []
    for i, sample in enumerate(rows):
        print(f"Video frame {i + 1}/{len(rows)}...", end="\r", flush=True)
        ip = (sample.get("images") or [None])[0]
        if not ip or not os.path.isfile(ip):
            continue
        pred = get_prediction_text(sample)
        if not pred:
            continue
        try:
            img = draw_scene_graph(ip, pred, max_width=max_width if max_width > 0 else None, title=title, title_wrap_long=True)
        except Exception as e:
            print(f"\nWARNING frame {i}: {e}")
            continue
        frames.append(np.array(img))
    print(f"\nEncoding {len(frames)} frames -> {out_mp4}", flush=True)
    if not frames:
        raise RuntimeError("No frames to encode for video")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    create_video_from_frames(frames, str(out_mp4), fps=fps)


def main() -> int:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _DEMO_ROOT = _SCRIPT_DIR.parent
    default_predicts = _DEMO_ROOT / "predicts_and_metrics"
    default_videos = _DEMO_ROOT / "videos_output"
    subdir = "field_deploy"

    p = argparse.ArgumentParser(
        description="Standalone field GEN (PVSG obj/rel): video or frames dir → temporal chaining → jsonl + MP4.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--video",
        type=str,
        default="",
        help="Video file (MP4/MOV, ffmpeg at --fps) OR a directory of frames (e.g. …/0A8CF.mp4/).",
    )
    src.add_argument("--frames-dir", type=str, default="", help="Directory of ordered frame images.")
    p.add_argument("--model", type=str, required=True, help="HF id or local checkpoint (ms-swift).")
    p.add_argument("--fps", type=float, default=10.0, help="Extraction rate for --video and output MP4 fps.")
    p.add_argument(
        "--results-root",
        type=str,
        default="",
        help=f"Parent folder (default: {_DEMO_ROOT}). Writes predicts_and_metrics/{subdir}/ and videos_output/{subdir}/.",
    )
    p.add_argument("--run-name", type=str, default="", help="Basename for .jsonl / .mp4 (default: <stem>_field_gen).")
    p.add_argument("--stem", type=str, default="", help="Override folder name under field_deploy/ (default: from video or frames dir).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--infer-backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--torch-dtype", type=str, default="bfloat16")
    p.add_argument("--template-type", type=str, default="")
    p.add_argument("--response-prefix", type=str, default="")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--auto-obj-prefix-fallback", action="store_true")
    p.add_argument("--no-stop-on-answer-close", action="store_true")
    p.add_argument("--stop-sequence", action="append", default=None)
    p.add_argument("--max-width", type=int, default=1920, help="Visualization width (0 = native).")
    p.add_argument("--no-video", action="store_true", help="Only write jsonl.")
    p.add_argument("--force", action="store_true", help="Overwrite outputs.")
    p.add_argument("--keep-extracted-frames", action="store_true", help="Keep ffmpeg frames under artifact dir.")
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="",
        metavar="IDS",
        help="Set CUDA_VISIBLE_DEVICES before loading the model (e.g. 0).",
    )
    args = p.parse_args()

    if args.cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()

    demo_root = Path(args.results_root).expanduser().resolve() if args.results_root.strip() else _DEMO_ROOT
    predicts_root = demo_root / "predicts_and_metrics" / subdir
    videos_root = demo_root / "videos_output" / subdir

    video_file: Optional[Path] = None
    frames_dir_direct: Optional[Path] = None

    if args.video.strip():
        vp = Path(args.video.strip()).expanduser().resolve()
        if vp.is_file():
            video_file = vp
            stem = args.stem.strip() or vp.stem
        elif vp.is_dir():
            frames_dir_direct = vp
            stem = args.stem.strip() or vp.stem
        else:
            print(f"ERROR: path not found (expected file or directory): {vp}", file=sys.stderr)
            return 1
    else:
        fd = Path(args.frames_dir.strip()).expanduser().resolve()
        if not fd.is_dir():
            print(f"ERROR: not a directory: {fd}", file=sys.stderr)
            return 1
        frames_dir_direct = fd
        stem = args.stem.strip() or fd.name

    run_name = args.run_name.strip() or f"{stem}_field_gen"
    artifact_dir = predicts_root / stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = artifact_dir / f"{run_name}.jsonl"
    frames_cache = artifact_dir / "_extracted_frames"
    out_mp4 = (videos_root / stem / f"{run_name}.mp4").resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    if out_jsonl.exists() and not args.force:
        print(f"[skip] exists: {out_jsonl} (use --force)", file=sys.stderr)
        return 0

    if args.fps <= 0:
        print("ERROR: --fps must be positive", file=sys.stderr)
        return 1

    if video_file is not None:
        print(f"[frames] extracting from {video_file} at fps={args.fps} -> {frames_cache}")
        if frames_cache.exists() and args.force:
            shutil.rmtree(frames_cache)
        frame_paths = extract_frames_video(video_file, frames_cache, args.fps)
    else:
        assert frames_dir_direct is not None
        print(f"[frames] using directory {frames_dir_direct}")
        frame_paths = list_frames_directory(frames_dir_direct)

    w, h = image_size(frame_paths[0])
    samples = synthesize_samples(frame_paths, w, h)

    template_type = args.template_type.strip() or None
    response_prefix = args.response_prefix.strip() or None
    torch_dtype = parse_torch_dtype(args.torch_dtype.strip())

    print(f"[init] model={args.model!r} backend={args.infer_backend} frames={len(samples)} size={w}x{h}")
    engine = build_swift_engine(
        model_id_or_path=args.model,
        infer_backend=args.infer_backend,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        template_type=template_type,
        enable_thinking=bool(args.enable_thinking),
        response_prefix=response_prefix,
        torch_dtype=torch_dtype,
    )

    fallback_engine: Optional[Union[VllmEngine, TransformersEngine]] = None
    if args.auto_obj_prefix_fallback and (response_prefix or "") != "obj[":
        fallback_engine = build_swift_engine(
            model_id_or_path=args.model,
            infer_backend=args.infer_backend,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            template_type=template_type,
            enable_thinking=bool(args.enable_thinking),
            response_prefix="obj[",
            torch_dtype=torch_dtype,
        )

    run_realtime_infer(
        samples=samples,
        engine=engine,
        fallback_engine=fallback_engine,
        out_path=out_jsonl,
        run_name=run_name,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        no_stop_on_answer_close=args.no_stop_on_answer_close,
        stop_sequence=args.stop_sequence,
        w=w,
        h=h,
    )
    print(f"[done] jsonl -> {out_jsonl}")

    if not args.no_video:
        try:
            render_pred_video(
                out_jsonl,
                out_mp4,
                fps=float(args.fps),
                max_width=args.max_width,
                title="Prediction (PVSG GEN field)",
            )
            print(f"[done] video -> {out_mp4}")
        except Exception as e:
            print(f"WARNING: video step failed: {e}", file=sys.stderr)

    if video_file is not None and not args.keep_extracted_frames:
        try:
            shutil.rmtree(frames_cache)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
