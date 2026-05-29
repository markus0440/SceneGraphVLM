"""Native lmdeploy VLM pipeline for Qwen3-VL and Qwen3.5 (lmdeploy>=0.10; ms-swift LmdeployEngine stops at 0.8)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from model_patches import patch_lmdeploy_consumer_gpu, patch_lmdeploy_qwen35_consumer_kernels

_NATIVE_LMDEPLOY_RE = re.compile(
    r"qwen3(?:[._-]5|\.5)|qwen3-vl|qwen3_vl",
    re.I,
)


def needs_native_lmdeploy(model_id_or_path: str) -> bool:
    """Return True when the checkpoint must use native ``lmdeploy.pipeline`` (not ms-swift LmdeployEngine)."""
    return bool(_NATIVE_LMDEPLOY_RE.search(model_id_or_path or ""))


def normalize_user_text(user_text: str) -> str:
    text = (user_text or "").strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip("\n")
    return text


@dataclass
class LmdeployNativeEngine:
    model_id_or_path: str
    pipe: Any
    template: Any

    def build_messages(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str = "",
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if (system_text or "").strip():
            messages.append({"role": "system", "content": system_text.strip()})
        img_url = Path(image_path).resolve().as_uri()
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": normalize_user_text(user_text)},
                ],
            }
        )
        return messages

    def run_infer(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str = "",
        max_new_tokens: int,
        temperature: float,
        stop: Sequence[str],
        use_stream: bool,
    ) -> Dict[str, Any]:
        if use_stream:
            return self._run_stream(
                user_text=user_text,
                image_path=image_path,
                system_text=system_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
            )
        return self._run_batch(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
        )

    def _gen_config(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        stop: Sequence[str],
    ) -> Any:
        from lmdeploy import GenerationConfig

        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0 if temperature <= 0 else 0.8,
            "top_k": 1 if temperature <= 0 else 40,
        }
        stops = [s for s in stop if s]
        if stops:
            kwargs["stop_words"] = stops
        return GenerationConfig(**kwargs)

    def _response_text(self, response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        text = getattr(response, "text", None)
        if text is not None:
            return str(text)
        if isinstance(response, (list, tuple)) and response:
            return self._response_text(response[0])
        return str(response)

    def _response_token_counts(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        if response is None:
            return prompt_tokens, completion_tokens
        if isinstance(response, (list, tuple)) and response:
            return self._response_token_counts(response[0])
        for attr in ("prompt_token_num", "prompt_tokens", "num_prompt_tokens", "input_token_len"):
            val = getattr(response, attr, None)
            if val is not None:
                prompt_tokens = int(val)
                break
        for attr in (
            "generate_token_num",
            "completion_tokens",
            "num_token",
            "num_tokens",
            "generate_token_len",
        ):
            val = getattr(response, attr, None)
            if val is not None:
                completion_tokens = int(val)
                break
        return prompt_tokens, completion_tokens

    def _run_batch(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str,
        max_new_tokens: int,
        temperature: float,
        stop: Sequence[str],
    ) -> Dict[str, Any]:
        messages = self.build_messages(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
        )
        gen_config = self._gen_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
        )
        t0 = time.perf_counter()
        response = self.pipe(messages, gen_config=gen_config)
        total_sec = time.perf_counter() - t0
        text = self._response_text(response)
        prompt_tokens_total, completion_tokens = self._response_token_counts(response)
        if completion_tokens is None and text:
            tok = getattr(getattr(self.template, "processor", None), "tokenizer", None)
            if tok is not None:
                completion_tokens = len(tok.encode(text, add_special_tokens=False))
        return {
            "text": text,
            "total_time_sec": total_sec,
            "time_to_first_token_sec": None,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens": completion_tokens,
        }

    def _run_stream(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str,
        max_new_tokens: int,
        temperature: float,
        stop: Sequence[str],
    ) -> Dict[str, Any]:
        messages = self.build_messages(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
        )
        gen_config = self._gen_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
        )
        t0 = time.perf_counter()
        ttft_sec: Optional[float] = None
        parts: List[str] = []
        prompt_tokens_total: Optional[int] = None
        completion_tokens: Optional[int] = None
        try:
            for chunk in self.pipe.stream_infer(messages, gen_config=gen_config):
                if chunk is None:
                    continue
                delta = self._response_text(chunk)
                if delta and ttft_sec is None:
                    ttft_sec = time.perf_counter() - t0
                if delta:
                    parts.append(delta)
                p, c = self._response_token_counts(chunk)
                if p is not None:
                    prompt_tokens_total = p
                if c is not None:
                    completion_tokens = c
        except AttributeError:
            return self._run_batch(
                user_text=user_text,
                image_path=image_path,
                system_text=system_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
            )
        total_sec = time.perf_counter() - t0
        text = "".join(parts)
        if completion_tokens is None and text:
            tok = getattr(getattr(self.template, "processor", None), "tokenizer", None)
            if tok is not None:
                completion_tokens = len(tok.encode(text, add_special_tokens=False))
        return {
            "text": text,
            "total_time_sec": total_sec,
            "time_to_first_token_sec": ttft_sec,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens": completion_tokens,
        }


def build_lmdeploy_native_engine(
    *,
    model_id_or_path: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    template_type: Optional[str],
    enable_thinking: bool,
    response_prefix: Optional[str],
    torch_dtype: torch.dtype,
    vision_batch_size: int = 1,
) -> LmdeployNativeEngine:
    if not needs_native_lmdeploy(model_id_or_path):
        raise ValueError(
            f"Model {model_id_or_path!r} is not routed to native lmdeploy; use ms-swift LmdeployEngine."
        )

    patch_lmdeploy_consumer_gpu()
    patch_lmdeploy_qwen35_consumer_kernels()

    try:
        from lmdeploy import PytorchEngineConfig, VisionConfig, pipeline
    except ImportError as e:
        raise ImportError(
            "Native lmdeploy path requires lmdeploy>=0.10. "
            "Use env swift_qwen_lmdeploy: bash envs/sh_scripts/install_swift_qwen_lmdeploy.sh"
        ) from e

    try:
        from swift.model import get_processor
        from swift.template import get_template
    except ImportError as e:
        raise ImportError("ms-swift is required for chat template / token breakdown") from e

    processor = get_processor(
        model_id_or_path=model_id_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    template = get_template(
        processor,
        template_type=template_type,
        enable_thinking=enable_thinking,
        response_prefix=response_prefix,
    )

    backend_config = PytorchEngineConfig(
        session_len=max_model_len,
        cache_max_entry_count=gpu_memory_utilization,
        tp=tensor_parallel_size,
        max_prefill_token_num=4096,
        eager_mode=False,
    )
    vision_config = VisionConfig(max_batch_size=max(1, vision_batch_size))

    pipe = pipeline(
        model_id_or_path,
        backend_config=backend_config,
        vision_config=vision_config,
    )
    return LmdeployNativeEngine(
        model_id_or_path=model_id_or_path,
        pipe=pipe,
        template=template,
    )
