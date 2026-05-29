"""Native SmolVLM inference (ms-swift does not register ``smolvlm`` model_type)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

_SMOLVLM_HINT_RE = re.compile(r"smolvlm", re.I)


def is_smolvlm_model(model_id_or_path: str) -> bool:
    """Return True when the checkpoint is SmolVLM / SmolVLM2."""
    if _SMOLVLM_HINT_RE.search(model_id_or_path):
        return True
    path = Path(model_id_or_path)
    cfg_path = path / "config.json" if path.is_dir() else None
    if cfg_path and cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return cfg.get("model_type") == "smolvlm"
        except Exception:
            return False
    return False


def normalize_user_text(user_text: str) -> str:
    """PSG manifest puts ``<image>`` in text; SmolVLM expects image as a separate part."""
    text = (user_text or "").strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip("\n")
    return text


def _make_stop_criteria(tokenizer: Any, stop_strings: Sequence[str], prompt_len: int) -> Any:
    from transformers import StoppingCriteria, StoppingCriteriaList

    stops = [s for s in stop_strings if s]

    class StopOnSubstrings(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
            if not stops:
                return False
            new_ids = input_ids[0, prompt_len:]
            if new_ids.numel() == 0:
                return False
            text = tokenizer.decode(new_ids, skip_special_tokens=False)
            return any(s in text for s in stops)

    return StoppingCriteriaList([StopOnSubstrings()])


@dataclass
class SmolVLMEngine:
    backend: str
    processor: Any
    llm: Any = None
    model: Any = None
    allowed_local_media_path: str = "/"

    def build_hf_messages(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str = "",
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if (system_text or "").strip():
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_text.strip()}],
                }
            )
        img = Image.open(image_path).convert("RGB")
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": normalize_user_text(user_text)},
                ],
            }
        )
        return messages

    def build_vllm_messages(
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

    def encode_prompt(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str = "",
    ) -> Dict[str, Any]:
        messages = self.build_hf_messages(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
        )
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def measure_prompt_tokens(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str = "",
        prompt_tokens_total: Optional[int] = None,
    ) -> Dict[str, Optional[int]]:
        enc = self.encode_prompt(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
        )
        input_ids = enc["input_ids"][0].tolist()
        image_token_id = self.processor.image_token_id
        visual_tokens = sum(1 for t in input_ids if t == image_token_id)
        input_total = int(prompt_tokens_total) if prompt_tokens_total is not None else len(input_ids)
        text_total = max(input_total - visual_tokens, 0)

        system_tokens = 0
        if (system_text or "").strip():
            try:
                sys_enc = self.processor.apply_chat_template(
                    [{"role": "system", "content": [{"type": "text", "text": system_text.strip()}]}],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                system_tokens = int(sys_enc["input_ids"].shape[1])
            except Exception:
                tok = self.processor.tokenizer
                system_tokens = len(tok.encode(system_text.strip(), add_special_tokens=False))

        template_prefix_tokens = 0
        user_prompt_tokens = max(text_total - system_tokens - template_prefix_tokens, 0)
        return {
            "system_prompt_tokens": system_tokens,
            "template_prefix_tokens": template_prefix_tokens,
            "user_prompt_tokens": user_prompt_tokens,
            "visual_tokens": int(visual_tokens),
            "text_prompt_tokens_total": text_total,
            "input_tokens_total": input_total,
        }

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
        if self.backend == "vllm":
            return self._run_vllm(
                user_text=user_text,
                image_path=image_path,
                system_text=system_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
            )
        if self.backend == "sglang":
            return self._run_sglang(
                user_text=user_text,
                image_path=image_path,
                system_text=system_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
            )
        return self._run_transformers(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
            use_stream=use_stream,
        )

    def _run_vllm(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str,
        max_new_tokens: int,
        temperature: float,
        stop: Sequence[str],
    ) -> Dict[str, Any]:
        from vllm import SamplingParams
        import time

        messages = self.build_vllm_messages(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
        )
        sp = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=list(stop),
        )
        t0 = time.perf_counter()
        outputs = self.llm.chat(messages, sampling_params=sp, use_tqdm=False)
        total_sec = time.perf_counter() - t0
        out = outputs[0]
        text = out.outputs[0].text or ""
        prompt_tokens_total = len(out.prompt_token_ids or [])
        completion_tokens = len(out.outputs[0].token_ids or [])
        return {
            "text": text,
            "total_time_sec": total_sec,
            "time_to_first_token_sec": None,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens": completion_tokens,
        }

    def _run_transformers(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str,
        max_new_tokens: int,
        temperature: float,
        stop: Sequence[str],
        use_stream: bool,
    ) -> Dict[str, Any]:
        from transformers import TextIteratorStreamer
        import time

        inputs = self.encode_prompt(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
        )
        device = self.model.device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "stopping_criteria": _make_stop_criteria(self.processor.tokenizer, stop, prompt_len),
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        if use_stream:
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            gen_kwargs["streamer"] = streamer
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            parts: List[str] = []
            ttft_sec: Optional[float] = None
            for piece in streamer:
                if piece and ttft_sec is None:
                    ttft_sec = time.perf_counter() - t0
                if piece:
                    parts.append(piece)
            thread.join()
            total_sec = time.perf_counter() - t0
            text = "".join(parts)
            completion_tokens = len(self.processor.tokenizer.encode(text, add_special_tokens=False))
        else:
            out_ids = self.model.generate(**gen_kwargs)
            total_sec = time.perf_counter() - t0
            new_ids = out_ids[0, prompt_len:]
            text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            completion_tokens = int(new_ids.shape[0])
            ttft_sec = None

        prompt_tokens_total = int(inputs["input_ids"].shape[1])
        return {
            "text": text,
            "total_time_sec": total_sec,
            "time_to_first_token_sec": ttft_sec,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens": completion_tokens,
        }

    def _run_sglang(
        self,
        *,
        user_text: str,
        image_path: str,
        system_text: str,
        max_new_tokens: int,
        temperature: float,
        stop: Sequence[str],
    ) -> Dict[str, Any]:
        import time
        from PIL import Image

        messages = self.build_hf_messages(
            user_text=user_text,
            image_path=image_path,
            system_text=system_text,
        )
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        img = Image.open(image_path).convert("RGB")
        sp: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stop": list(stop),
        }
        t0 = time.perf_counter()
        out = self.llm.generate(prompt=prompt, image_data=[img], sampling_params=sp)
        total_sec = time.perf_counter() - t0
        meta = out.get("meta_info") or {}
        text = out.get("text") or ""
        prompt_tokens_total = int(meta.get("prompt_tokens") or 0)
        completion_tokens = int(meta.get("completion_tokens") or 0)
        return {
            "text": text,
            "total_time_sec": total_sec,
            "time_to_first_token_sec": None,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens": completion_tokens,
        }


def build_smolvlm_engine(
    *,
    model_id_or_path: str,
    infer_backend: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    torch_dtype: torch.dtype,
) -> SmolVLMEngine:
    if infer_backend not in ("vllm", "transformers", "sglang"):
        raise ValueError(
            f"SmolVLM native path supports infer_backend vllm|transformers|sglang, got {infer_backend!r}"
        )

    try:
        from transformers import AutoProcessor
    except ImportError as e:
        raise ImportError("transformers is required for SmolVLM") from e

    processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)

    if infer_backend == "vllm":
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("vllm is required for SmolVLM + vLLM") from e
        llm = LLM(
            model=model_id_or_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 1},
            allowed_local_media_path="/",
            dtype=str(torch_dtype).replace("torch.", ""),
        )
        return SmolVLMEngine(backend="vllm", processor=processor, llm=llm)

    if infer_backend == "sglang":
        from model_patches import patch_sglang_smolvlm_compat

        patch_sglang_smolvlm_compat()
        try:
            import num2words  # noqa: F401
            from sglang import Engine
        except ImportError as e:
            raise ImportError(
                "SmolVLM + SGLang requires sglang and num2words: pip install num2words"
            ) from e
        llm = Engine(
            model_path=model_id_or_path,
            trust_remote_code=True,
            mem_fraction_static=gpu_memory_utilization,
            context_length=max_model_len,
        )
        return SmolVLMEngine(backend="sglang", processor=processor, llm=llm)

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as e:
        raise ImportError("AutoModelForImageTextToText requires recent transformers") from e

    model = AutoModelForImageTextToText.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()
    return SmolVLMEngine(backend="transformers", processor=processor, model=model)
