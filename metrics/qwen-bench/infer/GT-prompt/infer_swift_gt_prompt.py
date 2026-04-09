#!/usr/bin/env python3
"""
Batch inference via ms-swift inference engines (vLLM or Transformers) on Swift JSONL:
  {"messages": [{"role":"user","content":"<image>\\n..."}, {"role":"assistant", ...}], "images": ["..."]}

Uses swift.template.get_template(..., enable_thinking=False) by default (no chain-of-thought).

Example:
  export CUDA_VISIBLE_DEVICES=0
  export IMAGE_MAX_TOKEN_NUM=1024
  python infer_swift_gt_prompt.py \\
    --model sft/Qwen3.5/work_dirs/pvsg_maxinfo_Qwen3.5-0.8B/v5-20260409-135802/checkpoint-8844 \\
    --test-jsonl ../../../../datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test_clean.jsonl \\
    --output-dir ../../../../metrics/results/checkpoints-inference/sft/PVSG \\
    --run-name Qwen3.5-0.8B-SFT-maxinfo-checkpoint-8844-psg \\
    --batch-size 64

Zero-shot HF id:
  --model Qwen/Qwen3.5-VL-7B-Instruct --run-name Qwen3.5-VL-7B-zero-shot-psg ...

Requires: pip install 'ms-swift[llm]' and (for --infer-backend vllm) vllm
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
        "ms-swift is required. Install with e.g.:\n"
        "  pip install 'ms-swift[llm]' -U\n"
        "  pip install vllm   # for --infer-backend vllm\n"
        f"Import error: {e}"
    ) from e

_THINK_END = "</redacted_thinking>"
_SG_OBJ_HDR = re.compile(r"^\s*obj\[\d+\]\{id,name", re.MULTILINE)
_SG_REL_HDR = re.compile(r"^\s*rel\[\d+\]\{subj,pred,obj\}", re.MULTILINE)
_SG_REL_PAIRS_HDR = re.compile(
    r"^\s*rel_pairs\[\d+\]\{subj,attention", re.MULTILINE
)


def strip_thinking_suffix(text: str) -> str:
    s = text.strip()
    while _THINK_END in s:
        idx = s.find(_THINK_END)
        prefix = s[:idx]
        suffix = s[idx + len(_THINK_END) :].lstrip()
        if (
            _SG_OBJ_HDR.search(prefix)
            or _SG_REL_HDR.search(prefix)
            or _SG_REL_PAIRS_HDR.search(prefix)
        ):
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


def get_assistant_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    raise ValueError("No assistant message in sample")


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


def parse_torch_dtype(name: str) -> torch.dtype:
    d = getattr(torch, name, None)
    if d is None or not isinstance(d, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name}")
    return d


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
    """vLLM: processor + template only; Transformers: full model + template."""
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
        return TransformersEngine(
            model,
            template=tmpl,
            max_batch_size=max(1, batch_size),
        )

    raise ValueError(f"Unknown infer_backend: {infer_backend}")


def _error_record(
    sample: Dict[str, Any],
    *,
    model_path: str,
    run_name: str,
    dataset_tag: str,
    checkpoint_step: str,
    test_jsonl: Path,
    err: str,
    img_path: str,
) -> Dict[str, Any]:
    try:
        gt = get_assistant_text(sample)
    except Exception:
        gt = ""
    try:
        user_prompt = get_user_text(sample)
    except Exception:
        user_prompt = ""
    return {
        "messages": sample.get("messages"),
        "images": sample.get("images"),
        "content": gt,
        "predict": "",
        "model_name": run_name,
        "gen_time_sec": None,
        "predict_error": err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swift (vLLM or Transformers) batch inference on Swift-format JSONL → eval-style jsonl.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or local path (SFT checkpoint or base/zero-shot).",
    )
    parser.add_argument(
        "--test-jsonl",
        required=True,
        type=Path,
        help="Swift JSONL (messages + images), e.g. PSG_json/test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the output jsonl (created if missing).",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Logical run label and output basename, e.g. Qwen3.5-0.8B-zero-shot-psg",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument(
        "--infer-backend",
        choices=("vllm", "transformers"),
        default="vllm",
        help="vLLM (default) or HuggingFace Transformers via swift TransformersEngine",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="RequestConfig.max_tokens",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM only")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM only; default: 0.45",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM only")
    parser.add_argument(
        "--images-base",
        default="",
        help="Prefix for relative paths in images[]",
    )
    parser.add_argument(
        "--dataset-tag",
        default="swift_sgg",
        help="dataset_name field in each output row",
    )
    parser.add_argument(
        "--checkpoint-step",
        default="",
        help="Metadata checkpoint_step (for eval tooling); empty → 'inference'",
    )
    parser.add_argument(
        "--template-type",
        default="",
        help="Optional swift template_type; empty = auto from model",
    )
    parser.add_argument(
        "--response-prefix",
        default="",
        help="Optional template response_prefix (e.g. prefill for tagged outputs)",
    )
    parser.add_argument(
        "--auto-obj-prefix-fallback",
        action="store_true",
        help="If output has no obj[...] header, retry sample with response_prefix='obj['.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="torch dtype name, e.g. bfloat16, float16",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to get_template (default: off)",
    )
    parser.add_argument(
        "--no-stop-on-answer-close",
        action="store_true",
        help="Do not add stop string </answer> to RequestConfig",
    )
    parser.add_argument(
        "--stop-sequence",
        action="append",
        default=None,
        metavar="STR",
        help="Extra stop strings (repeatable)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output file")
    args = parser.parse_args()

    images_base = args.images_base or None
    template_type = args.template_type.strip() or None
    response_prefix = args.response_prefix.strip() or None
    ckpt_step = args.checkpoint_step.strip() or "inference"
    torch_dtype = parse_torch_dtype(args.torch_dtype.strip())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.run_name}.jsonl"
    if out_path.exists() and not args.force:
        print(f"[skip] exists: {out_path} (use --force)", file=sys.stderr)
        sys.exit(0)

    gpu_mem = args.gpu_memory_utilization

    enable_thinking = bool(args.enable_thinking)

    print(f"[init] model={args.model!r} backend={args.infer_backend} thinking={enable_thinking}")
    engine = build_swift_engine(
        model_id_or_path=args.model,
        infer_backend=args.infer_backend,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=args.tensor_parallel_size,
        template_type=template_type,
        enable_thinking=enable_thinking,
        response_prefix=response_prefix,
        torch_dtype=torch_dtype,
    )
    fallback_engine: Optional[Union[VllmEngine, TransformersEngine]] = None
    if args.auto_obj_prefix_fallback and (response_prefix or "") != "obj[":
        print("[init] auto obj-prefix fallback is enabled")
        fallback_engine = build_swift_engine(
            model_id_or_path=args.model,
            infer_backend=args.infer_backend,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=gpu_mem,
            tensor_parallel_size=args.tensor_parallel_size,
            template_type=template_type,
            enable_thinking=enable_thinking,
            response_prefix="obj[",
            torch_dtype=torch_dtype,
        )

    samples = read_jsonl(args.test_jsonl)
    print(f"[init] samples={len(samples)} from {args.test_jsonl}")

    stop_list: Optional[List[str]] = None
    if args.no_stop_on_answer_close:
        stop_list = list(args.stop_sequence) if args.stop_sequence else None
    else:
        stop_list = ["</answer>"]
        if args.stop_sequence:
            stop_list = stop_list + list(args.stop_sequence)

    req_cfg = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop=stop_list or [],
    )

    tmp = out_path.with_suffix(".jsonl.tmp")
    n_ok = 0
    bs = max(1, args.batch_size)

    with tmp.open("w", encoding="utf-8") as fout:
        i = 0
        pbar = tqdm(total=len(samples), desc="infer", unit="sample")
        while i < len(samples):
            batch = samples[i : i + bs]
            requests: List[InferRequest] = []
            meta: List[Tuple[int, Dict[str, Any], str, str]] = []

            for j, sample in enumerate(batch):
                idx = i + j
                img_path = ""
                try:
                    gt = get_assistant_text(sample)
                    user_content = get_user_text(sample)
                    img_path = resolve_image_path(sample, images_base)
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError(img_path)
                    requests.append(
                        InferRequest(
                            messages=[{"role": "user", "content": user_content}],
                            images=[img_path],
                        )
                    )
                    meta.append((idx, sample, gt, img_path))
                except Exception as e:
                    try:
                        img_path = resolve_image_path(sample, images_base)
                    except Exception:
                        img_path = ""
                    rec = _error_record(
                        sample,
                        model_path=args.model,
                        run_name=args.run_name,
                        dataset_tag=args.dataset_tag,
                        checkpoint_step=ckpt_step,
                        test_jsonl=args.test_jsonl,
                        err=str(e),
                        img_path=img_path,
                    )
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pbar.update(1)

            i += len(batch)
            if not requests:
                continue

            t0 = time.perf_counter()
            resp_list = engine.infer(requests, req_cfg)
            elapsed = time.perf_counter() - t0
            per = elapsed / len(requests)

            for (_, sample, gt, img_path), resp in zip(meta, resp_list):
                raw = resp.choices[0].message.content or ""
                pred_clean = finalize_vl_prediction(raw)
                if fallback_engine is not None and not has_obj_header(pred_clean):
                    try:
                        fb_req = InferRequest(
                            messages=[{"role": "user", "content": get_user_text(sample)}],
                            images=[img_path],
                        )
                        fb_resp = fallback_engine.infer([fb_req], req_cfg)[0]
                        fb_pred = finalize_vl_prediction(fb_resp.choices[0].message.content or "")
                        if has_obj_header(fb_pred):
                            pred_clean = fb_pred
                    except Exception:
                        pass
                rec = {
                    "messages": sample.get("messages"),
                    "images": sample.get("images"),
                    "content": gt,
                    "predict": pred_clean,
                    "model_name": args.run_name,
                    "gen_time_sec": per,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
                pbar.update(1)

        pbar.close()

    tmp.replace(out_path)
    print(f"[done] {n_ok} rows → {out_path}")


if __name__ == "__main__":
    main()
