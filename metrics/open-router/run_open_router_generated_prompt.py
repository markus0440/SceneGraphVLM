#!/usr/bin/env python3
"""
OpenRouter PVSG inference with *generated* temporal context: the block after
\"Previous frame scene graph (TOON):\" in the dataset user message is replaced
by the model's own TOON from the previous frame (plain text, like GT test.jsonl).

User prompt text is taken verbatim from --data-test (same layout as GT-prompt:
<answer> in the example and \"Write your response only between <answer> and </answer> tags.\").

  cd metrics/open-router
  export OPEN_ROUTER_KEY=...
  python run_open_router_generated_prompt.py openai/gpt-5.4-mini \\
    --data-test ../../datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \\
    --output-dir ../results/checkpoints-inference/open-router-models/PVSG-GEN-prompt \\
    --output-prefix psfr

  Output: {output_dir}/{model_short}-{output_prefix}.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
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

PREV_HEADER = "Previous frame scene graph (TOON):\n"
NOW_GENERATE = "Now, generate the complete scene graph"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def swift_user(sample: dict) -> str:
    for m in sample.get("messages") or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def swift_images(sample: dict) -> list[str]:
    return list(sample.get("images") or [])


def swift_assistant(sample: dict) -> str:
    for m in sample.get("messages") or []:
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def extract_wh_from_user(user_text: str) -> tuple[int, int]:
    m = re.search(r"image of size \((\d+)\s*x\s*(\d+)\)", user_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 640, 480


def inject_prev_toon(ref_user: str, prev_toon: str) -> str:
    """Replace GT TOON between PREV_HEADER and NOW_GENERATE with model chain TOON."""
    if not prev_toon.strip() or PREV_HEADER not in ref_user:
        return ref_user
    before, rest = ref_user.split(PREV_HEADER, 1)
    idx = rest.find(NOW_GENERATE)
    if idx == -1:
        return ref_user
    suffix = rest[idx:]
    return before + PREV_HEADER + prev_toon.strip() + "\n\n" + suffix


def parse_vid_t(image_abs: str) -> tuple[str, int]:
    p = Path(image_abs)
    try:
        t = int(p.stem)
    except ValueError:
        t = 0
    return p.parent.name, t


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


def wrap_answer_block(text: str) -> str:
    t = text.strip()
    if not t:
        return "<answer>\n</answer>"
    if t.lower().startswith("<answer>"):
        return t
    return f"<answer>\n{t}\n</answer>"


def parse_scene_graph_output(answer_text: str) -> dict:
    inner = answer_text.strip()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", inner, flags=re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1).strip()

    lines = [line.rstrip("\n") for line in inner.splitlines()]
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


def _bbox_gpt5_to_pixels(bbox, mid: str, img_width: int, img_height: int) -> list:
    mx, my = max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
    if "gpt-5-mini" in mid or "gpt-5.4-mini" in mid:
        if mx <= 460 and my <= 350:
            mw, mh = GPT5_MINI_GRID
            x1 = bbox[0] * img_width / mw
            y1 = bbox[1] * img_height / mh
            x2 = bbox[2] * img_width / mw
            y2 = bbox[3] * img_height / mh
        else:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    else:
        if mx <= 460 and my <= 350:
            mw, mh = GPT5_MINI_GRID
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


def get_corrected_prediction_text(
    pred_text: str, model_id: str, img_width: int, img_height: int
) -> str:
    scene = parse_scene_graph_output(pred_text)
    mid = (model_id or "").lower()
    if model_id:
        for obj in scene["objects"]:
            if "gpt-5-mini" in mid or "gpt-5.4-mini" in mid or "gpt-5-nano" in mid or "gpt-5.4-nano" in mid:
                obj["bbox"] = _bbox_gpt5_to_pixels(obj["bbox"], mid, img_width, img_height)
            elif "gemini" in mid:
                bbox = obj["bbox"]
                x1, y1, x2, y2 = bbox
                m = max(x1, y1, x2, y2)
                norm = 1000.0 if (m > 1.0 and m <= 1000) else 1.0
                x1 = (x1 / norm) * img_width
                y1 = (y1 / norm) * img_height
                x2 = (x2 / norm) * img_width
                y2 = (y2 / norm) * img_height
                x1, x2 = max(0, min(x1, x2)), min(img_width, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(img_height, max(y1, y2))
                obj["bbox"] = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    lines_out = [f"obj[{len(scene['objects'])}]{{id,name,x1,y1,x2,y2}}:"]
    for o in scene["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        lines_out.append(f"  {o['id']},{o['label']},{x1},{y1},{x2},{y2}")
    lines_out.append(f"rel[{len(scene['relations'])}]{{subj,pred,obj}}:")
    for r in scene["relations"]:
        lines_out.append(f"  {r['sub']},{r['pred']},{r['obj']}")
    return "\n".join(lines_out)


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
    file_suffix = args.output_prefix.strip()
    if file_suffix.lower().endswith(".jsonl"):
        file_suffix = file_suffix[: -len(".jsonl")]
    file_suffix = safe_filename(file_suffix).lstrip("-_")
    if not file_suffix:
        sys.exit("--output-prefix must be non-empty (e.g. psfr)")
    meta_suffix = (args.dataset_suffix or file_suffix).strip()

    data_path = Path(args.data_test).resolve()
    samples = read_jsonl(data_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    n = len(samples)

    vids: list[str] = []
    ts: list[int] = []
    ref_users: list[str] = []
    for s in samples:
        ru = swift_user(s)
        ref_users.append(ru)
        imgs = swift_images(s)
        if not imgs:
            vids.append("")
            ts.append(0)
        else:
            vid, t = parse_vid_t(imgs[0])
            vids.append(vid)
            ts.append(t)

    by_video: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i in range(n):
        by_video[vids[i]].append((ts[i], i))
    for vid in by_video:
        by_video[vid].sort(key=lambda x: x[0])

    predictions: list[str | None] = [None] * n
    gen_times: list[float | None] = [None] * n
    costs: list[float] = [0.0] * n
    user_full: list[str] = [""] * n

    out_dir = (Path.cwd() / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{short_name}-{file_suffix}.jsonl"

    total_cost = 0.0
    pbar = tqdm(total=n, desc="gen-prompt infer", unit="frame")

    for vid, frame_list in by_video.items():
        prev_toon = ""
        for t, idx in frame_list:
            sample = samples[idx]
            imgs = swift_images(sample)
            if not imgs:
                tqdm.write(f"skip idx={idx}: no images")
                pbar.update(1)
                continue
            image_path = str(Path(imgs[0]).expanduser())
            if not os.path.isfile(image_path):
                tqdm.write(f"skip idx={idx}: missing {image_path}")
                pbar.update(1)
                continue

            ref_u = ref_users[idx]
            full_user = inject_prev_toon(ref_u, prev_toon) if t >= 1 else ref_u
            user_full[idx] = full_user
            prompt_api = full_user.replace("<image>", "").strip()

            t0 = time.perf_counter()
            try:
                res = response_img_with_retries(
                    prompt_api,
                    image_path,
                    model_id,
                    api_key,
                    max_retries=args.max_retries,
                    base_delay=args.retry_delay,
                )
            except Exception as e:
                tqdm.write(f"API error idx={idx} vid={vid} t={t}: {e}")
                predictions[idx] = ""
                gen_times[idx] = 0.0
                prev_toon = ""
                pbar.update(1)
                continue
            elapsed = time.perf_counter() - t0
            usage = res.get("usage", {}) or {}
            cost = usage.get("cost", 0.0) or 0.0
            pred_raw = res.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

            w, h = extract_wh_from_user(ref_u)
            if sample.get("width") is not None and sample.get("height") is not None:
                w, h = int(sample["width"]), int(sample["height"])
            else:
                try:
                    with Image.open(image_path) as im:
                        w, h = im.size
                except Exception:
                    pass

            try:
                corrected = get_corrected_prediction_text(pred_raw, model_id, w, h)
            except Exception:
                corrected = pred_raw.strip()
                m = re.search(
                    r"<answer>\s*(.*?)\s*</answer>",
                    corrected,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if m:
                    corrected = m.group(1).strip()

            predictions[idx] = corrected
            gen_times[idx] = elapsed
            costs[idx] = cost
            total_cost += float(cost)
            prev_toon = corrected
            pbar.update(1)

    pbar.close()

    with open(out_file, "w", encoding="utf-8") as fout:
        for idx in range(n):
            sample = samples[idx]
            pred = predictions[idx] or ""
            assistant = wrap_answer_block(pred)
            fout.write(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_full[idx] or ref_users[idx],
                            },
                            {"role": "assistant", "content": assistant},
                        ],
                        "images": swift_images(sample),
                        "content": swift_assistant(sample),
                        "predict": assistant,
                        "model_name": short_name,
                        "dataset_suffix": meta_suffix,
                        "gen_time_sec": round(gen_times[idx], 6) if gen_times[idx] is not None else None,
                        "cost": costs[idx],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"done: {out_file}  total_cost={total_cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenRouter PVSG GEN-prompt (model-chained previous frame) inference.",
    )
    parser.add_argument(
        "model",
        help="OpenRouter model id, e.g. openai/gpt-5.4-mini",
    )
    parser.add_argument(
        "--data-test",
        type=str,
        required=True,
        help="Swift JSONL (messages + images); user text must match GT layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Filename suffix: {model_short}-{output_prefix}.jsonl",
    )
    parser.add_argument(
        "--dataset-suffix",
        type=str,
        default="",
        help="Row metadata dataset_suffix (default: same as --output-prefix).",
    )
    parser.add_argument("--env", type=str, default="env")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=2.0)

    args = parser.parse_args()
    run_infer(args)


if __name__ == "__main__":
    main()
