#!/usr/bin/env python3
"""
OpenRouter vision inference for scene-graph JSONL in ms-swift format (messages + images).

  cd metrics/open-router
  export OPEN_ROUTER_KEY=...   # or use --env path/to/.env
  python run_open_router_gt_prompt.py openai/gpt-5.4-mini \\
    --data-test ../../datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \\
    --output-dir ../results/checkpoints-inference/open-router-models/PVSG-GT-prompt \\
    --output-prefix psfr

  Output: {output_dir}/{model_short}-{output_prefix}.jsonl

  PSG example:
  python run_open_router_gt_prompt.py openai/gpt-5.4-mini \\
    --data-test ../../datasets/data_playground/PSG_json/test.jsonl \\
    --output-dir ../results/checkpoints-inference/open-router-models/PSG \\
    --output-prefix psg
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

SCRIPT_DIR = Path(__file__).resolve().parent


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def mime_for_image_path(filepath: str) -> str:
    low = filepath.lower()
    if low.endswith(".png"):
        return "image/png"
    if low.endswith(".webp"):
        return "image/webp"
    if low.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def response_img(query: str, filepath: str, model: str, api_key: str) -> dict:
    base64_image = encode_image_to_base64(filepath)
    mime = mime_for_image_path(filepath)
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "scene-graph-inference",
            "X-Title": "Scene graph inference",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def response_img_with_retries(
    query: str,
    filepath: str,
    model: str,
    api_key: str,
    max_retries: int = 4,
    base_delay: float = 2.0,
) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            return response_img(query, filepath, model, api_key)
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
        ) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
                continue
            raise last_error
        except Exception as e:
            err_msg = str(e).lower()
            if "prematurely" in err_msg or "connection" in err_msg or "timeout" in err_msg:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))
                    continue
            raise
    raise last_error


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def swift_user_text(sample: dict) -> str:
    for m in sample.get("messages") or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def swift_images(sample: dict) -> list[str]:
    return list(sample.get("images") or [])


def swift_assistant_text(sample: dict) -> str:
    for m in sample.get("messages") or []:
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def wrap_answer_block(text: str) -> str:
    t = text.strip()
    if not t:
        return "<answer>\n</answer>"
    low = t.lower()
    if low.startswith("<answer>"):
        return t
    return f"<answer>\n{t}\n</answer>"


def parse_scene_graph_output(answer_text: str) -> dict:
    lines = [line.rstrip("\n") for line in answer_text.splitlines()]
    lines = [line for line in lines if line.strip()]

    obj_header_idx = None
    rel_header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("obj[") and "{id,name,x1,y1,x2,y2}" in stripped:
            obj_header_idx = i
        if stripped.startswith("rel[") and "{subj,pred,obj}" in stripped:
            rel_header_idx = i

    if obj_header_idx is None or rel_header_idx is None:
        raise ValueError("Missing obj[...] or rel[...] headers in model output")

    obj_lines = lines[obj_header_idx + 1 : rel_header_idx]
    objects = []
    for raw_line in obj_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 6:
            continue
        idx_str, name, x1_str, y1_str, x2_str, y2_str = parts
        try:
            obj_id = int(idx_str)
            x1, y1 = int(x1_str), int(y1_str)
            x2, y2 = int(x2_str), int(y2_str)
        except ValueError:
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        objects.append({"id": obj_id, "label": name, "bbox": [x1, y1, x2, y2]})

    rel_lines = lines[rel_header_idx + 1 :]
    relations = []
    for raw_line in rel_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 3:
            continue
        subj_str, pred, obj_str = parts
        try:
            subj_id, obj_id = int(subj_str), int(obj_str)
        except ValueError:
            continue
        relations.append({"sub": subj_id, "obj": obj_id, "pred": pred})

    return {"objects": objects, "relations": relations}


GPT5_MINI_GRID = (448, 336)
GPT5_NANO_GRID = (640, 480)


def bbox_model_to_pixels(bbox, model_id: str, img_width: int, img_height: int) -> list:
    mid = model_id.lower() if model_id else ""
    if "gpt-5-mini" in mid or "gpt-5.4-mini" in mid:
        mx, my = max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
        if mx <= 460 and my <= 350:
            mw, mh = GPT5_MINI_GRID
            x1 = bbox[0] * img_width / mw
            y1 = bbox[1] * img_height / mh
            x2 = bbox[2] * img_width / mw
            y2 = bbox[3] * img_height / mh
        else:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        return [
            int(round(max(0, min(x1, x2)))),
            int(round(max(0, min(y1, y2)))),
            int(round(min(img_width, max(x1, x2)))),
            int(round(min(img_height, max(y1, y2)))),
        ]
    if "gpt-5-nano" in mid or "gpt-5.4-nano" in mid:
        mx, my = max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
        if mx <= 460 and my <= 350:
            mw, mh = GPT5_MINI_GRID
            x1 = bbox[0] * img_width / mw
            y1 = bbox[1] * img_height / mh
            x2 = bbox[2] * img_width / mw
            y2 = bbox[3] * img_height / mh
        else:
            mw, mh = GPT5_NANO_GRID
            x1 = bbox[0] * img_width / mw
            y1 = bbox[1] * img_height / mh
            x2 = bbox[2] * img_width / mw
            y2 = bbox[3] * img_height / mh
        return [
            int(round(max(0, min(x1, x2)))),
            int(round(max(0, min(y1, y2)))),
            int(round(min(img_width, max(x1, x2)))),
            int(round(min(img_height, max(y1, y2)))),
        ]
    if "gemini" in mid:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        m = max(x1, y1, x2, y2)
        norm = 1000.0 if (m > 1.0 and m <= 1000) else 1.0
        x1 = (x1 / norm) * img_width
        y1 = (y1 / norm) * img_height
        x2 = (x2 / norm) * img_width
        y2 = (y2 / norm) * img_height
        x1, x2 = max(0, min(x1, x2)), min(img_width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(img_height, max(y1, y2))
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    a, b, c, d = bbox[0], bbox[1], bbox[2], bbox[3]
    m = max(a, b, c, d)
    if m <= 1.0:
        x1 = max(0, min(a, c)) * img_width
        x2 = min(1.0, max(a, c)) * img_width
        y1 = max(0, min(b, d)) * img_height
        y2 = min(1.0, max(b, d)) * img_height
    else:
        x1 = max(0, min(a, c))
        x2 = min(img_width, max(a, c))
        y1 = max(0, min(b, d))
        y2 = min(img_height, max(b, d))
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def get_corrected_prediction_text(
    pred_text: str, model_id: str, img_width: int, img_height: int
) -> str:
    inner = pred_text.strip()
    if inner.lower().startswith("<answer>"):
        m = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            pred_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        inner = m.group(1).strip() if m else inner
    scene = parse_scene_graph_output(inner)
    if model_id:
        for obj in scene["objects"]:
            obj["bbox"] = bbox_model_to_pixels(obj["bbox"], model_id, img_width, img_height)
    lines = [f"obj[{len(scene['objects'])}]{{id,name,x1,y1,x2,y2}}:"]
    for o in scene["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        lines.append(f"  {o['id']},{o['label']},{x1},{y1},{x2},{y2}")
    lines.append(f"rel[{len(scene['relations'])}]{{subj,pred,obj}}:")
    for r in scene["relations"]:
        lines.append(f"  {r['sub']},{r['pred']},{r['obj']}")
    body = "\n".join(lines)
    return wrap_answer_block(body)


def model_id_to_short_name(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[1]
    return model_id


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def run_infer(args: argparse.Namespace) -> None:
    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = SCRIPT_DIR / env_path
    load_dotenv(env_path)
    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if not api_key:
        sys.exit("OPEN_ROUTER_KEY missing (env or dotenv file).")

    model_id = args.model.strip()
    short_name = safe_filename(model_id_to_short_name(model_id))
    jsonl_path = Path(args.data_test).resolve()
    file_suffix = args.output_prefix.strip()
    if file_suffix.lower().endswith(".jsonl"):
        file_suffix = file_suffix[: -len(".jsonl")]
    file_suffix = safe_filename(file_suffix).lstrip("-_")
    if not file_suffix:
        sys.exit("--output-prefix must be non-empty (e.g. psfr, psg)")
    meta_suffix = (args.dataset_suffix or file_suffix).strip()
    out_dir = (Path.cwd() / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{short_name}-{file_suffix}.jsonl"

    samples = read_jsonl(jsonl_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    n = len(samples)

    with open(out_file, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(tqdm(samples, desc="infer", unit="sample")):
            imgs = swift_images(sample)
            if not imgs:
                tqdm.write(f"skip {i + 1}/{n}: no images[]")
                continue
            image_path = str(Path(imgs[0]).expanduser())
            if not os.path.isfile(image_path):
                tqdm.write(f"skip {i + 1}/{n}: missing file {image_path}")
                continue
            prompt_text = swift_user_text(sample).replace("<image>", "").strip()
            t0 = time.perf_counter()
            try:
                res = response_img_with_retries(
                    prompt_text,
                    image_path,
                    model_id,
                    api_key,
                    max_retries=args.max_retries,
                    base_delay=args.retry_delay,
                )
            except Exception as e:
                tqdm.write(f"API error [{i + 1}/{n}]: {e}")
                continue
            elapsed = time.perf_counter() - t0
            usage = res.get("usage", {}) or {}
            cost = usage.get("cost")
            if cost is None:
                cost = 0.0
            pred_text = res.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

            img_w = sample.get("width")
            img_h = sample.get("height")
            if img_w is None or img_h is None:
                try:
                    with Image.open(image_path) as im:
                        img_w, img_h = im.size
                except Exception:
                    img_w, img_h = 640, 480

            try:
                pred_assistant = get_corrected_prediction_text(
                    pred_text, model_id, int(img_w), int(img_h)
                )
            except Exception:
                pred_assistant = wrap_answer_block(pred_text)

            out_record = {
                "messages": [
                    {"role": "user", "content": swift_user_text(sample)},
                    {"role": "assistant", "content": pred_assistant},
                ],
                "images": swift_images(sample),
                "content": swift_assistant_text(sample),
                "predict": pred_assistant,
                "model_name": short_name,
                "dataset_suffix": meta_suffix,
                "gen_time_sec": round(elapsed, 6),
                "cost": cost,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"done: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenRouter scene-graph inference (swift JSONL).",
    )
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="OpenRouter model id, e.g. openai/gpt-5.4-mini",
    )
    parser.add_argument(
        "--data-test",
        type=str,
        default=None,
        help="Swift-format test JSONL (messages + images). Required for inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (inference only).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Filename suffix after model name: {model_short}-{output_prefix}.jsonl (e.g. psfr).",
    )
    parser.add_argument(
        "--dataset-suffix",
        type=str,
        default="",
        help="Stored in each row as dataset_suffix (default: same as --output-prefix).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="env",
        help="Dotenv path with OPEN_ROUTER_KEY (default: ./env next to script).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max samples (inference).")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=2.0)

    args = parser.parse_args()

    if not args.model or not args.data_test:
        sys.exit(
            "usage: python run_open_router_gt_prompt.py MODEL --data-test JSONL "
            "--output-dir DIR --output-prefix SUFFIX [...]"
        )
    if not args.output_dir:
        sys.exit("inference requires --output-dir")
    if not args.output_prefix:
        sys.exit("inference requires --output-prefix (e.g. psfr)")

    run_infer(args)


if __name__ == "__main__":
    main()
