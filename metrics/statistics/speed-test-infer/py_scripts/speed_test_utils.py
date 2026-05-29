"""Helpers for speed-test manifest I/O, environment metadata, and token accounting."""
from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib.metadata as importlib_metadata

_IMAGE_SIZE_RE = re.compile(r"image of size \((\d+)\s*x\s*(\d+)\)", re.IGNORECASE)

# Swift skips expanding vision tokens in encode for these backends (vLLM expands at runtime).
_VLLM_VISION_BACKENDS = frozenset({"vllm", "lmdeploy", "sglang"})


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_user_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    raise ValueError("No user message in sample")


def get_system_text(sample: Dict[str, Any]) -> str:
    parts: List[str] = []
    for msg in sample.get("messages", []):
        if msg.get("role") == "system":
            parts.append((msg.get("content") or "").strip())
    return "\n".join(p for p in parts if p)


def resolve_image_path(sample: Dict[str, Any], images_base: Optional[str]) -> str:
    imgs = sample.get("images") or []
    if not imgs:
        raise ValueError("No images in sample")
    p = imgs[0]
    if os.path.isabs(p):
        return p
    if not images_base:
        return p
    return os.path.join(images_base, p)


def parse_image_size_from_prompt(user_text: str, image_path: str) -> Tuple[int, int]:
    m = _IMAGE_SIZE_RE.search(user_text or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return 0, 0


def backend_display_name(infer_backend: str) -> str:
    return {
        "vllm": "vLLM",
        "transformers": "HF",
        "lmdeploy": "LMDeploy",
        "sglang": "SGLang",
    }.get(infer_backend, infer_backend)


def result_output_dir(base_dir: Path, infer_backend: str) -> Path:
    """Results layout: ``<base>/vLLM/`` or ``<base>/HF/`` (plus legacy flat files)."""
    sub = backend_display_name(infer_backend)
    return base_dir / sub


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "gpu_name": None,
        "gpu_memory_total_gb": None,
        "driver_version": None,
        "cuda_version": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_total_gb"] = round(props.total_memory / (1024**3), 3)
            info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 1 and not info.get("gpu_name"):
                info["gpu_name"] = parts[0]
            if len(parts) >= 2 and info.get("gpu_memory_total_gb") is None:
                mem = parts[1].replace(" MiB", "").strip()
                try:
                    info["gpu_memory_total_gb"] = round(float(mem) / 1024, 3)
                except ValueError:
                    pass
            if len(parts) >= 3:
                info["driver_version"] = parts[2]
    except Exception:
        pass
    return info


def collect_environment() -> Dict[str, Any]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "conda_env": conda_env,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "ms_swift": package_version("ms-swift"),
            "vllm": package_version("vllm"),
            "lmdeploy": package_version("lmdeploy"),
            "sglang": package_version("sglang"),
        },
        "env_vars": {
            "IMAGE_MAX_TOKEN_NUM": os.environ.get("IMAGE_MAX_TOKEN_NUM"),
            "VIDEO_MAX_TOKEN_NUM": os.environ.get("VIDEO_MAX_TOKEN_NUM"),
            "MAX_PIXELS": os.environ.get("MAX_PIXELS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu": collect_gpu_info(),
    }


def _get_tokenizer(tmpl: Any) -> Any:
    proc = getattr(tmpl, "processor", None)
    if proc is not None:
        tok = getattr(proc, "tokenizer", None)
        if tok is not None:
            return tok
    return getattr(tmpl, "tokenizer", None)


def _effective_system_text(tmpl: Any, system_text: str) -> str:
    explicit = (system_text or "").strip()
    if explicit:
        return explicit
    meta = getattr(tmpl, "template_meta", None)
    default = getattr(meta, "default_system", None) if meta is not None else None
    return (default or "").strip()


def _count_visual_tokens_from_images(tmpl: Any, image_paths: List[str]) -> Optional[int]:
    """Qwen-VL style: sum(prod(image_grid_thw) // merge_size**2) per image."""
    if not image_paths:
        return None
    proc = getattr(tmpl, "processor", None)
    processor = getattr(proc, "image_processor", None) if proc is not None else None
    if processor is None:
        return None
    merge_size = getattr(processor, "merge_size", None)
    if merge_size is None:
        return None

    try:
        from qwen_vl_utils import fetch_image
    except ImportError:
        fetch_image = None

    patch_size = getattr(processor, "patch_size", None)
    images: List[Any] = []
    for path in image_paths:
        if fetch_image is not None:
            kwargs = {}
            if patch_size is not None:
                kwargs["image_patch_size"] = patch_size
            images.append(fetch_image({"image": path}, **kwargs))
        else:
            images.append(path)

    try:
        media_inputs = processor(images=images, return_tensors="pt", do_resize=False)
    except Exception:
        return None
    grid = media_inputs.get("image_grid_thw")
    if grid is None:
        return None

    merge_length = int(merge_size) ** 2
    total = 0
    for i in range(len(grid)):
        total += int(grid[i].prod().item() // merge_length)
    cap = resolve_visual_token_budget()
    if cap is not None:
        total = min(total, cap)
    return total


def _count_template_prefix_tokens(tmpl: Any) -> int:
    prefix = getattr(tmpl, "response_prefix", None) or ""
    if not prefix:
        return 0
    tok = _get_tokenizer(tmpl)
    if tok is None:
        return 0
    return count_text_tokens(tok, prefix)


def count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = enc.get("input_ids") if isinstance(enc, dict) else enc
        return len(ids) if ids is not None else 0
    except Exception:
        return len(tokenizer.encode(text, add_special_tokens=False))


def measure_prompt_tokens(
    engine: Any,
    *,
    user_text: str,
    image_path: str,
    system_text: str,
    prompt_tokens_total: Optional[int] = None,
) -> Dict[str, Optional[int]]:
    """
    Token breakdown aligned with Swift inference and engine ``usage.prompt_tokens``.

    For vLLM/LMDeploy/SGLang, ``template.encode`` leaves a single ``<|image_pad|>`` placeholder
    per image; real vision token count comes from the image processor grid (same as Qwen-VL ``_encode``).
    ``input_tokens_total`` prefers ``prompt_tokens_total`` from the backend when available.

    ``system_prompt_tokens``: explicit system in the sample or ``template.default_system`` (often None
    for Qwen3.5). Chat-template overhead (e.g. thinking ``response_prefix``) is in ``template_prefix_tokens``.
    """
    out: Dict[str, Optional[int]] = {
        "system_prompt_tokens": 0,
        "template_prefix_tokens": 0,
        "user_prompt_tokens": None,
        "visual_tokens": None,
        "text_prompt_tokens_total": None,
        "input_tokens_total": None,
    }
    tmpl = getattr(engine, "template", None)
    if tmpl is None:
        return out

    try:
        from swift.infer_engine import InferRequest
    except ImportError:
        return out

    effective_system = _effective_system_text(tmpl, system_text)
    system_tokens = 0
    if effective_system:
        try:
            sys_enc = tmpl.encode(
                InferRequest(messages=[{"role": "system", "content": effective_system}])
            )
            system_tokens = len(sys_enc.get("input_ids") or [])
        except Exception:
            tok = _get_tokenizer(tmpl)
            if tok is not None:
                system_tokens = count_text_tokens(tok, effective_system)

    out["system_prompt_tokens"] = system_tokens
    template_prefix_tokens = _count_template_prefix_tokens(tmpl)
    out["template_prefix_tokens"] = template_prefix_tokens

    try:
        full_enc = tmpl.encode(
            InferRequest(
                messages=[{"role": "user", "content": user_text}],
                images=[image_path],
            )
        )
        input_ids = full_enc.get("input_ids") or []
        image_token_id = getattr(tmpl, "image_token_id", None)
        placeholders = (
            sum(1 for t in input_ids if t == image_token_id) if image_token_id is not None else 0
        )

        visual_from_grid = _count_visual_tokens_from_images(tmpl, [image_path])
        mode = getattr(tmpl, "mode", None)
        text_from_encode = max(len(input_ids) - placeholders, 0)

        if mode in _VLLM_VISION_BACKENDS:
            if visual_from_grid is not None:
                visual_tokens = visual_from_grid
            elif prompt_tokens_total is not None and text_from_encode > 0:
                visual_tokens = max(int(prompt_tokens_total) - text_from_encode, 0)
            else:
                visual_tokens = placeholders
            if prompt_tokens_total is not None:
                input_total = int(prompt_tokens_total)
                text_total = max(input_total - int(visual_tokens), 0)
            else:
                input_total = len(input_ids)
                text_total = text_from_encode
        else:
            if visual_from_grid is not None:
                visual_tokens = visual_from_grid
            elif image_token_id is not None:
                visual_tokens = sum(1 for t in input_ids if t == image_token_id)
            else:
                visual_tokens = resolve_visual_token_budget()
            text_total = max(len(input_ids) - int(visual_tokens or 0), 0)
            input_total = int(prompt_tokens_total) if prompt_tokens_total is not None else len(input_ids)

        out["visual_tokens"] = int(visual_tokens) if visual_tokens is not None else None
        out["text_prompt_tokens_total"] = text_total
        out["input_tokens_total"] = input_total
        out["user_prompt_tokens"] = max(
            text_total - system_tokens - template_prefix_tokens,
            0,
        )
    except Exception:
        if prompt_tokens_total is not None:
            out["input_tokens_total"] = int(prompt_tokens_total)
            vt = _count_visual_tokens_from_images(tmpl, [image_path])
            if vt is None:
                vt = resolve_visual_token_budget()
            out["visual_tokens"] = vt
            if vt is not None:
                out["text_prompt_tokens_total"] = max(int(prompt_tokens_total) - vt, 0)
                out["user_prompt_tokens"] = max(
                    out["text_prompt_tokens_total"] - system_tokens - template_prefix_tokens,
                    0,
                )

    return out


def resolve_visual_token_budget() -> Optional[int]:
    raw = os.environ.get("IMAGE_MAX_TOKEN_NUM", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def usage_fields(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens_total": None,
            "completion_tokens": None,
        }
    return {
        "prompt_tokens_total": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def slugify(text: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w.\-]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def result_filename(gpu_name: Optional[str], model_display: str, infer_backend: str) -> str:
    """e.g. NVIDIA_GeForce_RTX_5060_Ti_Qwen3.5-0.8B_vllm.json (single underscores)."""
    gpu_slug = slugify(gpu_name or "unknown_gpu")
    model_slug = slugify(model_display)
    backend_slug = slugify(infer_backend)
    return f"{gpu_slug}_{model_slug}_{backend_slug}.json"


def aggregate_numeric(values: List[float]) -> Dict[str, Any]:
    """Mean, median, sample std (σ), and ``mean ± σ`` string for timing metrics."""
    from statistics import mean, median, stdev

    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "phys": None,
        }
    n = len(values)
    m = float(mean(values))
    med = float(median(values))
    sd = float(stdev(values)) if n >= 2 else 0.0
    if n >= 2:
        phys = f"({m:.4g} ± {sd:.4g})"
    else:
        phys = f"{m:.4g}"
    return {"n": n, "mean": m, "median": med, "std": sd, "phys": phys}


# metric key → (label, unit) for summary tables
SUMMARY_METRIC_META: Dict[str, tuple[str, str]] = {
    "tokens_per_second": ("Tokens per second", "tok/s"),
    "time_to_first_token_sec": ("Time to first token", "s"),
    "vision_encoding_time_sec": ("Vision encoding time", "s"),
    "decode_time_sec": ("Decode time", "s"),
    "total_time_sec": ("Total time", "s"),
    "output_text_tokens": ("Output tokens", "tokens"),
    "input_tokens_total": ("Input tokens (total)", "tokens"),
    "text_prompt_tokens_total": ("Text prompt tokens (total)", "tokens"),
    "system_prompt_tokens": ("System prompt tokens", "tokens"),
    "template_prefix_tokens": ("Template prefix tokens", "tokens"),
    "user_prompt_tokens": ("User prompt tokens", "tokens"),
    "visual_tokens": ("Visual tokens", "tokens"),
}


def build_summary_rows(aggregate: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, (label, unit) in SUMMARY_METRIC_META.items():
        block = aggregate.get(key)
        if not block or block.get("mean") is None:
            continue
        rows.append(
            {
                "metric": key,
                "label": label,
                "unit": unit,
                "n": block.get("n"),
                "mean": block.get("mean"),
                "std": block.get("std"),
                "median": block.get("median"),
                "phys": block.get("phys"),
            }
        )
    return rows


def _fmt_num(v: Optional[float], *, sci_small: bool = True) -> str:
    if v is None:
        return "—"
    if sci_small and abs(v) < 0.01 and v != 0:
        return f"{v:.4g}"
    if abs(v) >= 1000:
        return f"{v:.4g}"
    if float(int(v)) == float(v):
        return str(int(v))
    return f"{v:.4g}"


def format_summary_table_ascii(
    *,
    gpu: Optional[str],
    model_display: str,
    accelerator: str,
    num_samples: int,
    summary_rows: List[Dict[str, Any]],
) -> str:
    """Plain ASCII table (no Markdown)."""
    header_lines = [
        "Speed test summary",
        "",
        f"GPU: {gpu or '—'}",
        f"Model: {model_display}",
        f"Accelerator: {accelerator}",
        f"Samples: {num_samples}",
        "",
    ]
    columns = [
        ("Metric", "label"),
        ("Mean", "mean"),
        ("± std", "std"),
        ("Median", "median"),
        ("Unit", "unit"),
    ]

    def cell(row: Dict[str, Any], field: str) -> str:
        if field == "label":
            return str(row.get("label", ""))
        if field == "unit":
            return str(row.get("unit", ""))
        if field == "mean":
            return _fmt_num(row.get("mean"))
        if field == "std":
            n = row.get("n") or 0
            return _fmt_num(row.get("std")) if n >= 2 else "—"
        if field == "median":
            return _fmt_num(row.get("median"))
        return ""

    rows_data = summary_rows
    widths = []
    for title, field in columns:
        w = len(title)
        for r in rows_data:
            w = max(w, len(cell(r, field)))
        widths.append(w)

    def sep_line() -> str:
        return "|" + "|".join("-" * (w + 2) for w in widths) + "|"

    def data_line(row: Dict[str, Any]) -> str:
        parts = []
        for i, (title, field) in enumerate(columns):
            parts.append(" " + cell(row, field).ljust(widths[i]) + " ")
        return "|" + "|".join(parts) + "|"

    header_row = "|" + "|".join(
        " " + title.ljust(widths[i]) + " " for i, (title, _) in enumerate(columns)
    ) + "|"

    body = [sep_line(), header_row, sep_line()]
    for r in rows_data:
        body.append(data_line(r))
    body.append(sep_line())

    return "\n".join(header_lines + body) + "\n"
