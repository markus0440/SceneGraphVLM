"""Shared ms-swift inference engine factory for speed tests and benchmarks."""
from __future__ import annotations

from typing import Any, Optional, Union

import torch

try:
    from swift.model import get_model_processor, get_processor
    from swift.template import get_template
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ms-swift is required. Install with: pip install 'ms-swift' -U"
    ) from e

from model_patches import (  # noqa: E402
    apply_model_patches,
    patch_lmdeploy_api_compat,
    patch_lmdeploy_consumer_gpu,
    patch_remote_model,
    patch_swift_lmdeploy_engine,
    patch_swift_sglang_engine,
    patch_swift_submodel_loader,
)
from lmdeploy_native_engine import needs_native_lmdeploy  # noqa: E402

patch_swift_submodel_loader()

Engine = Any
BACKEND_CHOICES = ("vllm", "transformers", "lmdeploy", "sglang")


def parse_torch_dtype(name: str) -> torch.dtype:
    d = getattr(torch, name, None)
    if d is None or not isinstance(d, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name}")
    return d


def build_swift_engine(
    *,
    model_id_or_path: str,
    infer_backend: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    template_type: Optional[str],
    enable_thinking: bool,
    response_prefix: Optional[str],
    torch_dtype: torch.dtype,
    vision_batch_size: int = 8,
) -> Engine:
    """Build a Swift infer engine (vLLM, HF, LMDeploy, or SGLang)."""
    if infer_backend not in BACKEND_CHOICES:
        raise ValueError(f"Unknown infer_backend: {infer_backend!r}; expected one of {BACKEND_CHOICES}")

    if infer_backend == "vllm":
        try:
            from swift.infer_engine import VllmEngine
        except ImportError as e:
            raise ImportError(
                "VllmEngine requires vllm. Use env swift_qwen_3_5_sft or: pip install vllm"
            ) from e
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
        model_key = model_id_or_path.lower()
        vllm_engine_kwargs: dict = {}
        max_num_seqs = 256
        if "qwen3.5-4b" in model_key:
            # vLLM profiles vision encoder at 16384 tokens by default → OOM on 16GB.
            vllm_engine_kwargs["skip_mm_profiling"] = True
            # batch=1 benchmark; default max_num_seqs=256 over-reserves KV on 16GB.
            max_num_seqs = 1
        if any(k in model_key for k in ("deepseek-vl2", "deepseek_vl2")):
            # Avoid FlashInfer MoE JIT compile OOM on 16GB host RAM; triton is stable here.
            vllm_engine_kwargs["kernel_config"] = {"moe_backend": "triton"}
        enforce_eager = any(k in model_key for k in ("deepseek-vl2", "deepseek_vl2"))
        return VllmEngine(
            model_id_or_path,
            template=tmpl,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=enforce_eager,
            engine_kwargs=vllm_engine_kwargs or None,
        )

    if infer_backend == "transformers":
        try:
            from swift.infer_engine import TransformersEngine
        except ImportError as e:
            raise ImportError(
                "TransformersEngine requires ms-swift. pip install 'ms-swift' -U"
            ) from e
        apply_model_patches(model_id_or_path)
        model, processor = get_model_processor(
            model_id_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        patch_remote_model(model)
        tmpl = get_template(
            processor,
            template_type=template_type,
            enable_thinking=enable_thinking,
            response_prefix=response_prefix,
        )
        return TransformersEngine(
            model,
            template=tmpl,
            max_batch_size=1,
        )

    if infer_backend == "lmdeploy":
        if needs_native_lmdeploy(model_id_or_path):
            raise ImportError(
                "Qwen3/Qwen3.5 on LMDeploy uses native lmdeploy.pipeline (lmdeploy>=0.10), "
                "not ms-swift LmdeployEngine. Use env swift_qwen_lmdeploy."
            )
        patch_lmdeploy_api_compat()
        patch_lmdeploy_consumer_gpu()
        patch_swift_lmdeploy_engine()
        try:
            from swift.infer_engine import LmdeployEngine
        except ImportError as e:
            raise ImportError(
                "LmdeployEngine requires lmdeploy. Use env swift_qwen_lmdeploy or: "
                "bash envs/sh_scripts/install_swift_qwen_lmdeploy.sh"
            ) from e
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
        return LmdeployEngine(
            model_id_or_path,
            template=tmpl,
            torch_dtype=torch_dtype,
            session_len=max_model_len,
            cache_max_entry_count=gpu_memory_utilization,
            vision_batch_size=1,
            engine_kwargs={"tp": tensor_parallel_size},
        )

    if infer_backend == "sglang":
        patch_swift_sglang_engine()
        try:
            from swift.infer_engine import SglangEngine
        except ImportError as e:
            raise ImportError(
                "SglangEngine requires sglang. Use env swift_qwen_sglang or: "
                "bash envs/sh_scripts/install_swift_qwen_sglang.sh"
            ) from e
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
        return SglangEngine(
            model_id_or_path,
            template=tmpl,
            torch_dtype=torch_dtype,
            tp_size=tensor_parallel_size,
            mem_fraction_static=gpu_memory_utilization,
            context_length=max_model_len,
        )

    raise AssertionError(f"Unhandled infer_backend: {infer_backend}")
