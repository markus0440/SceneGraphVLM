#!/usr/bin/env python3
"""
Scene-graph generation speed benchmark (batch size 1, PSG GT-prompt).

Writes one JSON per (model, backend) run under metrics/results/speed_test_infer/{vLLM,HF}/.

Example:
  export CUDA_VISIBLE_DEVICES=0
  export IMAGE_MAX_TOKEN_NUM=1024
  python metrics/statistics/speed-test-infer/py_scripts/speed_test_infer.py \\
    --model Qwen/Qwen3.5-0.8B \\
    --model-display-name Qwen3.5-0.8B \\
    --manifest datasets/data_playground/PSG_json/test.jsonl \\
    --infer-backend vllm \\
    --warmup-runs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_STATS_DIR = Path(__file__).resolve().parent
_STI_ROOT = _STATS_DIR.parent
_REPO_ROOT = _STI_ROOT.parents[2]  # py_scripts → speed-test-infer → statistics → metrics → repo root
if str(_STATS_DIR) not in sys.path:
    sys.path.insert(0, str(_STATS_DIR))

from speed_test_utils import (  # noqa: E402
    backend_display_name,
    collect_environment,
    get_system_text,
    get_user_text,
    parse_image_size_from_prompt,
    read_jsonl,
    resolve_image_path,
    aggregate_numeric,
    build_summary_rows,
    format_summary_table_ascii,
    measure_prompt_tokens,
    result_filename,
    result_output_dir,
    usage_fields,
)
from swift_engine import BACKEND_CHOICES, build_swift_engine, parse_torch_dtype  # noqa: E402
from smolvlm_engine import SmolVLMEngine, build_smolvlm_engine, is_smolvlm_model  # noqa: E402
from lmdeploy_native_engine import (  # noqa: E402
    LmdeployNativeEngine,
    build_lmdeploy_native_engine,
    needs_native_lmdeploy,
)
from model_patches import apply_model_patches, patch_remote_engine  # noqa: E402

try:
    from swift.infer_engine import InferRequest, RequestConfig
    from swift.metrics import InferStats
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "ms-swift is required. Install with:\n"
        "  pip install 'ms-swift[llm]' -U\n"
        f"Import error: {e}"
    ) from e

_DEFAULT_MANIFEST = _REPO_ROOT / "datasets/data_playground/PSG_json/test.jsonl"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "metrics/results/speed_test_infer"


def _get_tokenizer(engine: Any) -> Any:
    tmpl = getattr(engine, "template", None)
    if tmpl is not None:
        proc = getattr(tmpl, "processor", None)
        if proc is not None:
            tok = getattr(proc, "tokenizer", None)
            if tok is not None:
                return tok
    proc = getattr(engine, "processor", None)
    if proc is not None:
        return getattr(proc, "tokenizer", proc)
    raise RuntimeError("Could not resolve tokenizer from Swift engine")


def _run_streaming_infer(
    engine: Any,
    request: InferRequest,
    req_cfg: RequestConfig,
) -> Dict[str, Any]:
    """Single-sample streaming infer; returns text, TTFT, and token usage."""
    metric = InferStats()
    t0 = time.perf_counter()
    gen = engine.infer([request], req_cfg, metrics=[metric])[0]
    ttft_sec: Optional[float] = None
    parts: List[str] = []
    last_resp: Any = None
    for chunk in gen:
        if chunk is None:
            continue
        last_resp = chunk
        delta = chunk.choices[0].delta.content or ""
        if delta and ttft_sec is None:
            ttft_sec = time.perf_counter() - t0
        if delta:
            parts.append(delta)
    total_sec = time.perf_counter() - t0
    text = "".join(parts)
    stats = metric.compute()
    usage = usage_fields(last_resp) if last_resp is not None else {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = stats.get("num_generated_tokens")
    prompt_tokens_total = usage.get("prompt_tokens_total")
    if prompt_tokens_total is None:
        prompt_tokens_total = stats.get("num_prompt_tokens")
    return {
        "text": text,
        "total_time_sec": total_sec,
        "time_to_first_token_sec": ttft_sec,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens": completion_tokens,
    }


def _run_batch_infer(
    engine: Any,
    request: InferRequest,
    req_cfg: RequestConfig,
) -> Dict[str, Any]:
    """Non-streaming fallback (warmup and if stream unsupported)."""
    metric = InferStats()
    t0 = time.perf_counter()
    resp_list = engine.infer([request], req_cfg, metrics=[metric])
    total_sec = time.perf_counter() - t0
    resp = resp_list[0]
    text = resp.choices[0].message.content or ""
    stats = metric.compute()
    usage = usage_fields(resp)
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = stats.get("num_generated_tokens")
    prompt_tokens_total = usage.get("prompt_tokens_total")
    if prompt_tokens_total is None:
        prompt_tokens_total = stats.get("num_prompt_tokens")
    return {
        "text": text,
        "total_time_sec": total_sec,
        "time_to_first_token_sec": None,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens": completion_tokens,
    }


def _derive_timings(
    *,
    total_time_sec: float,
    time_to_first_token_sec: Optional[float],
    completion_tokens: Optional[int],
) -> Dict[str, Optional[float]]:
    decode_sec: Optional[float] = None
    if time_to_first_token_sec is not None:
        decode_sec = max(total_time_sec - time_to_first_token_sec, 0.0)
    tokens_per_second: Optional[float] = None
    if completion_tokens and completion_tokens > 0:
        denom = decode_sec if decode_sec and decode_sec > 0 else total_time_sec
        if denom > 0:
            tokens_per_second = completion_tokens / denom
    vision_encoding_time_sec: Optional[float] = None
    if time_to_first_token_sec is not None:
        # VLMs: TTFT is dominated by vision+text prefill; no separate hook in Swift.
        vision_encoding_time_sec = time_to_first_token_sec
    return {
        "decode_time_sec": decode_sec,
        "tokens_per_second": tokens_per_second,
        "vision_encoding_time_sec": vision_encoding_time_sec,
    }


def _measure_sample_smolvlm(
    engine: SmolVLMEngine,
    *,
    user_text: str,
    image_path: str,
    system_text: str,
    max_new_tokens: int,
    temperature: float,
    stop: List[str],
    use_stream: bool,
) -> Dict[str, Any]:
    out = engine.run_infer(
        user_text=user_text,
        image_path=image_path,
        system_text=system_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=stop,
        use_stream=use_stream,
    )
    tok = engine.measure_prompt_tokens(
        user_text=user_text,
        image_path=image_path,
        system_text=system_text,
        prompt_tokens_total=out.get("prompt_tokens_total"),
    )
    timings = _derive_timings(
        total_time_sec=float(out["total_time_sec"]),
        time_to_first_token_sec=out.get("time_to_first_token_sec"),
        completion_tokens=out.get("completion_tokens"),
    )
    w, h = parse_image_size_from_prompt(user_text, image_path)
    return {
        "image_path": image_path,
        "image_width": w,
        "image_height": h,
        "image_size": f"{w}x{h}" if w and h else None,
        "system_prompt_tokens": tok.get("system_prompt_tokens"),
        "template_prefix_tokens": tok.get("template_prefix_tokens"),
        "user_prompt_tokens": tok.get("user_prompt_tokens"),
        "visual_tokens": tok.get("visual_tokens"),
        "text_prompt_tokens_total": tok.get("text_prompt_tokens_total"),
        "input_tokens_total": tok.get("input_tokens_total"),
        "prompt_tokens_total": out.get("prompt_tokens_total"),
        "output_text_tokens": out.get("completion_tokens"),
        "vision_encoding_time_sec": timings["vision_encoding_time_sec"],
        "time_to_first_token_sec": out.get("time_to_first_token_sec"),
        "decode_time_sec": timings["decode_time_sec"],
        "total_time_sec": out["total_time_sec"],
        "tokens_per_second": timings["tokens_per_second"],
        "output_chars": len(out.get("text") or ""),
    }


def _measure_sample_lmdeploy_native(
    engine: LmdeployNativeEngine,
    *,
    user_text: str,
    image_path: str,
    system_text: str,
    max_new_tokens: int,
    temperature: float,
    stop: List[str],
    use_stream: bool,
) -> Dict[str, Any]:
    out = engine.run_infer(
        user_text=user_text,
        image_path=image_path,
        system_text=system_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=stop,
        use_stream=use_stream,
    )
    tok = measure_prompt_tokens(
        engine,
        user_text=user_text,
        image_path=image_path,
        system_text=system_text,
        prompt_tokens_total=out.get("prompt_tokens_total"),
    )
    timings = _derive_timings(
        total_time_sec=float(out["total_time_sec"]),
        time_to_first_token_sec=out.get("time_to_first_token_sec"),
        completion_tokens=out.get("completion_tokens"),
    )
    w, h = parse_image_size_from_prompt(user_text, image_path)
    return {
        "image_path": image_path,
        "image_width": w,
        "image_height": h,
        "image_size": f"{w}x{h}" if w and h else None,
        "system_prompt_tokens": tok.get("system_prompt_tokens"),
        "template_prefix_tokens": tok.get("template_prefix_tokens"),
        "user_prompt_tokens": tok.get("user_prompt_tokens"),
        "visual_tokens": tok.get("visual_tokens"),
        "text_prompt_tokens_total": tok.get("text_prompt_tokens_total"),
        "input_tokens_total": tok.get("input_tokens_total"),
        "prompt_tokens_total": out.get("prompt_tokens_total"),
        "output_text_tokens": out.get("completion_tokens"),
        "vision_encoding_time_sec": timings["vision_encoding_time_sec"],
        "time_to_first_token_sec": out.get("time_to_first_token_sec"),
        "decode_time_sec": timings["decode_time_sec"],
        "total_time_sec": out["total_time_sec"],
        "tokens_per_second": timings["tokens_per_second"],
        "output_chars": len(out.get("text") or ""),
    }


def _measure_sample(
    engine: Any,
    *,
    user_text: str,
    image_path: str,
    req_cfg_stream: RequestConfig,
    req_cfg_batch: RequestConfig,
    system_text: str,
    use_stream: bool,
) -> Dict[str, Any]:
    request = InferRequest(
        messages=[{"role": "user", "content": user_text}],
        images=[image_path],
    )
    if use_stream:
        try:
            out = _run_streaming_infer(engine, request, req_cfg_stream)
        except Exception as e:
            out = _run_batch_infer(engine, request, req_cfg_batch)
            out["stream_error"] = str(e)
    else:
        out = _run_batch_infer(engine, request, req_cfg_batch)

    tok = measure_prompt_tokens(
        engine,
        user_text=user_text,
        image_path=image_path,
        system_text=system_text,
        prompt_tokens_total=out.get("prompt_tokens_total"),
    )

    timings = _derive_timings(
        total_time_sec=float(out["total_time_sec"]),
        time_to_first_token_sec=out.get("time_to_first_token_sec"),
        completion_tokens=out.get("completion_tokens"),
    )
    w, h = parse_image_size_from_prompt(user_text, image_path)
    row: Dict[str, Any] = {
        "image_path": image_path,
        "image_width": w,
        "image_height": h,
        "image_size": f"{w}x{h}" if w and h else None,
        "system_prompt_tokens": tok.get("system_prompt_tokens"),
        "template_prefix_tokens": tok.get("template_prefix_tokens"),
        "user_prompt_tokens": tok.get("user_prompt_tokens"),
        "visual_tokens": tok.get("visual_tokens"),
        "text_prompt_tokens_total": tok.get("text_prompt_tokens_total"),
        "input_tokens_total": tok.get("input_tokens_total"),
        "prompt_tokens_total": out.get("prompt_tokens_total"),
        "output_text_tokens": out.get("completion_tokens"),
        "vision_encoding_time_sec": timings["vision_encoding_time_sec"],
        "time_to_first_token_sec": out.get("time_to_first_token_sec"),
        "decode_time_sec": timings["decode_time_sec"],
        "total_time_sec": out["total_time_sec"],
        "tokens_per_second": timings["tokens_per_second"],
        "output_chars": len(out.get("text") or ""),
    }
    if out.get("stream_error"):
        row["stream_error"] = out["stream_error"]
    return row


def _aggregate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = (
        "vision_encoding_time_sec",
        "time_to_first_token_sec",
        "decode_time_sec",
        "total_time_sec",
        "tokens_per_second",
        "output_text_tokens",
        "input_tokens_total",
        "text_prompt_tokens_total",
        "system_prompt_tokens",
        "template_prefix_tokens",
        "user_prompt_tokens",
        "visual_tokens",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        vals = [float(s[key]) for s in samples if s.get(key) is not None]
        out[key] = aggregate_numeric(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PSG scene-graph speed benchmark → JSON.")
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument(
        "--model-display-name",
        default="",
        help="Human-readable model label for JSON/tables (default: basename of --model).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Swift JSONL manifest (messages + images).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for result JSON files.",
    )
    parser.add_argument(
        "--infer-backend",
        choices=BACKEND_CHOICES,
        default="vllm",
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="Warmup inferences per image (excluded).")
    parser.add_argument("--limit", type=int, default=0, help="Max manifest rows (0 = all).")
    parser.add_argument("--images-base", default="", help="Prefix for relative image paths.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--template-type", default="")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--vision-batch-size", type=int, default=8, help="LMDeploy only.")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming TTFT measurement (wall-clock only).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing result JSON.")
    parser.add_argument(
        "--no-print-table",
        action="store_true",
        help="Do not print ASCII summary table after the JSON is saved.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    model_display = args.model_display_name.strip() or Path(args.model).name
    images_base = args.images_base.strip() or None
    template_type = args.template_type.strip() or None
    torch_dtype = parse_torch_dtype(args.torch_dtype.strip())
    use_stream = not args.no_stream
    # LMDeploy PyTorch: streaming leaves the session/engine in a bad state on ms-swift (2nd
    # request → CUDA illegal access). Batch infer + async_end in full path is stable.
    use_swift_stream = use_stream and args.infer_backend != "lmdeploy"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = collect_environment()["gpu"].get("gpu_name")
    out_name = result_filename(gpu_name, model_display, args.infer_backend)
    out_dir = result_output_dir(args.output_dir, args.infer_backend)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    if out_path.exists() and not args.force:
        print(f"[skip] exists: {out_path} (use --force)", file=sys.stderr)
        sys.exit(0)

    samples_raw = read_jsonl(manifest_path)
    if args.limit > 0:
        samples_raw = samples_raw[: args.limit]

    stop_list = ["</answer>"]
    req_cfg_stream = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop=stop_list,
        stream=True,
    )
    req_cfg_batch = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop=stop_list,
        stream=False,
    )

    use_smolvlm = is_smolvlm_model(args.model)
    use_native_lmdeploy = args.infer_backend == "lmdeploy" and needs_native_lmdeploy(args.model)
    if use_smolvlm and args.infer_backend not in ("vllm", "transformers", "sglang"):
        raise SystemExit(
            f"SmolVLM uses a native path (ms-swift has no smolvlm model_type). "
            f"Use --infer-backend vllm, transformers, or sglang, not {args.infer_backend!r}."
        )
    if use_native_lmdeploy:
        try:
            import lmdeploy as _lmdeploy

            if tuple(int(x) for x in _lmdeploy.__version__.split(".")[:2]) < (0, 10):
                raise SystemExit(
                    f"Qwen3/Qwen3.5 LMDeploy needs lmdeploy>=0.10 (found {_lmdeploy.__version__}). "
                    "Use: conda activate swift_qwen_lmdeploy"
                )
        except ImportError as e:
            raise SystemExit(
                "Qwen3/Qwen3.5 LMDeploy needs env swift_qwen_lmdeploy. "
                "Run: bash envs/sh_scripts/install_swift_qwen_lmdeploy.sh"
            ) from e

    print(
        f"[init] model={args.model!r} display={model_display!r} "
        f"backend={args.infer_backend} manifest={manifest_path} n={len(samples_raw)}"
    )
    init_note = (
        "[init] loading SmolVLM engine (native vLLM/transformers; ms-swift skipped)..."
        if use_smolvlm
        else "[init] loading native lmdeploy pipeline (Qwen3/Qwen3.5; ms-swift LmdeployEngine skipped)..."
        if use_native_lmdeploy
        else "[init] loading engine (first run may download weights / compile kernels; GPU idle until backend starts)..."
    )
    print(init_note, flush=True)
    apply_model_patches(args.model)
    try:
        if use_smolvlm:
            engine = build_smolvlm_engine(
                model_id_or_path=args.model,
                infer_backend=args.infer_backend,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                torch_dtype=torch_dtype,
            )
        elif use_native_lmdeploy:
            engine = build_lmdeploy_native_engine(
                model_id_or_path=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                template_type=template_type,
                enable_thinking=False,
                response_prefix=None,
                torch_dtype=torch_dtype,
                vision_batch_size=args.vision_batch_size,
            )
        else:
            engine = build_swift_engine(
                model_id_or_path=args.model,
                infer_backend=args.infer_backend,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                template_type=template_type,
                enable_thinking=False,
                response_prefix=None,
                torch_dtype=torch_dtype,
                vision_batch_size=args.vision_batch_size,
            )
            if args.infer_backend == "transformers":
                patch_remote_engine(engine)
    except Exception as e:
        err_payload = {
            "status": "engine_init_failed",
            "error": str(e),
            "gpu": collect_environment()["gpu"].get("gpu_name"),
            "model": args.model,
            "model_display_name": model_display,
            "infer_backend": args.infer_backend,
            "accelerator": backend_display_name(args.infer_backend),
        }
        out_path.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Engine init failed: {e}") from e

    measured: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples_raw):
        user_text = get_user_text(sample)
        system_text = get_system_text(sample)
        image_path = resolve_image_path(sample, images_base)
        if not os.path.isfile(image_path):
            print(f"[warn] skip missing image: {image_path}", file=sys.stderr)
            continue
        sample_id = Path(image_path).stem
        print(f"[sample {idx + 1}/{len(samples_raw)}] {sample_id}", flush=True)

        for w in range(max(0, args.warmup_runs)):
            if isinstance(engine, SmolVLMEngine):
                _measure_sample_smolvlm(
                    engine,
                    user_text=user_text,
                    image_path=image_path,
                    system_text=system_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stop=stop_list,
                    use_stream=False,
                )
            elif isinstance(engine, LmdeployNativeEngine):
                _measure_sample_lmdeploy_native(
                    engine,
                    user_text=user_text,
                    image_path=image_path,
                    system_text=system_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stop=stop_list,
                    use_stream=False,
                )
            else:
                _measure_sample(
                    engine,
                    user_text=user_text,
                    image_path=image_path,
                    req_cfg_stream=req_cfg_stream,
                    req_cfg_batch=req_cfg_batch,
                    system_text=system_text,
                    use_stream=use_swift_stream,
                )

        if isinstance(engine, SmolVLMEngine):
            row = _measure_sample_smolvlm(
                engine,
                user_text=user_text,
                image_path=image_path,
                system_text=system_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                stop=stop_list,
                use_stream=use_stream and engine.backend == "transformers",
            )
        elif isinstance(engine, LmdeployNativeEngine):
            row = _measure_sample_lmdeploy_native(
                engine,
                user_text=user_text,
                image_path=image_path,
                system_text=system_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                stop=stop_list,
                use_stream=False,
            )
        else:
            row = _measure_sample(
                engine,
                user_text=user_text,
                image_path=image_path,
                req_cfg_stream=req_cfg_stream,
                req_cfg_batch=req_cfg_batch,
                system_text=system_text,
                use_stream=use_swift_stream,
            )
        row["sample_id"] = sample_id
        measured.append(row)
        print(
            f"  tok/s={row.get('tokens_per_second')} "
            f"ttft={row.get('time_to_first_token_sec')} "
            f"out_tokens={row.get('output_text_tokens')}",
            flush=True,
        )

    env = collect_environment()
    aggregate = _aggregate(measured)
    summary_rows = build_summary_rows(aggregate)
    accelerator = backend_display_name(args.infer_backend)
    result: Dict[str, Any] = {
        "status": "ok" if measured else "no_samples",
        "gpu": env["gpu"].get("gpu_name"),
        "model": args.model,
        "model_display_name": model_display,
        "infer_backend": args.infer_backend,
        "accelerator": accelerator,
        "manifest": str(manifest_path),
        "num_manifest_rows": len(samples_raw),
        "num_measured_samples": len(measured),
        "protocol": {
            "batch_size": 1,
            "warmup_runs_per_image": args.warmup_runs,
            "stream_for_ttft": (
                use_stream and getattr(engine, "backend", None) == "transformers"
                if use_smolvlm
                else use_swift_stream
            ),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "stop": stop_list,
            "tokens_per_second_formula": "output_tokens / decode_time_sec; if TTFT missing, decode_time=total_time",
            "vision_encoding_time_note": (
                "Equals TTFT when streaming; VL prefill is not split by Swift."
                if not use_smolvlm
                else "HF SmolVLM: equals TTFT when streaming. vLLM SmolVLM: TTFT not measured (batch chat API)."
            ),
            "engine_note": (
                "native SmolVLM (ms-swift has no smolvlm model_type)"
                if use_smolvlm
                else "native lmdeploy pipeline (Qwen3/Qwen3.5; lmdeploy>=0.10)"
                if use_native_lmdeploy
                else "ms-swift"
            ),
            "aggregate_std_note": "std is sample standard deviation (σ) over measured images, n>=2.",
        },
        "environment": env,
        "samples": measured,
        "aggregate": aggregate,
        "summary_table": summary_rows,
    }
    table_txt = ""
    if summary_rows:
        table_txt = format_summary_table_ascii(
            gpu=result.get("gpu"),
            model_display=model_display,
            accelerator=accelerator,
            num_samples=len(measured),
            summary_rows=summary_rows,
        )
        result["summary_table_text"] = table_txt

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    print(f"[done] {out_path}")

    if not args.no_print_table and table_txt:
        print()
        print(table_txt)


if __name__ == "__main__":
    main()
