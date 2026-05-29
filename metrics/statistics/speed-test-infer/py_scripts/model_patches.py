"""Workarounds for remote-code models on transformers 4.5x + ms-swift."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

_REMOTE_CAUSAL_LM_NAMES = (
    "InternLM2ForCausalLM",
    "DeepseekV2ForCausalLM",
)


def _patch_causal_lm_class(cls: Any) -> Any:
    """Remote-code causal LMs often lost ``generate()`` on transformers 4.50+."""
    if cls is None or not isinstance(cls, type):
        return cls
    name = getattr(cls, "__name__", "")
    if name not in _REMOTE_CAUSAL_LM_NAMES:
        return cls
    try:
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        return cls
    if issubclass(cls, GenerationMixin):
        if name == "InternLM2ForCausalLM":
            _patch_internlm2_prepare_inputs(cls)
        elif name == "DeepseekV2ForCausalLM":
            _patch_deepseek_prepare_inputs(cls)
        return cls
    cls.__bases__ = cls.__bases__ + (GenerationMixin,)
    if name == "InternLM2ForCausalLM":
        _patch_internlm2_prepare_inputs(cls)
    elif name == "DeepseekV2ForCausalLM":
        _patch_deepseek_prepare_inputs(cls)
    return cls


def _patch_deepseek_prepare_inputs(cls: Any) -> None:
    """DeepSeek-VL2 ships a pre-4.57 ``prepare_inputs_for_generation`` that yields empty decode steps."""
    if cls is None or getattr(cls, "__name__", "") != "DeepseekV2ForCausalLM":
        return
    if getattr(cls, "_sgvlm_deepseek_prepare_patched", False):
        return
    try:
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        return
    cls.prepare_inputs_for_generation = GenerationMixin.prepare_inputs_for_generation  # type: ignore[method-assign]
    cls._sgvlm_deepseek_prepare_patched = True


def _patch_internlm2_prepare_inputs(cls: Any) -> None:
    if cls is None or getattr(cls, "__name__", "") != "InternLM2ForCausalLM":
        return
    if getattr(cls, "_sgvlm_prepare_inputs_patched", False):
        return
    orig = cls.prepare_inputs_for_generation

    def _prepare_inputs(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            try:
                first = past_key_values[0]
                layer = first[0] if isinstance(first, (tuple, list)) else first
                if layer is None or not hasattr(layer, "shape"):
                    past_key_values = None
            except (IndexError, TypeError, AttributeError):
                past_key_values = None
        return orig(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    cls.prepare_inputs_for_generation = _prepare_inputs  # type: ignore[method-assign]
    cls._sgvlm_prepare_inputs_patched = True


def _patch_loaded_causal_lm_modules() -> None:
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        name = getattr(mod, "__name__", "") or ""
        if "modeling_" not in name:
            continue
        for cls_name in _REMOTE_CAUSAL_LM_NAMES:
            _patch_causal_lm_class(getattr(mod, cls_name, None))


def patch_swift_submodel_loader() -> None:
    """Patch remote causal LM before ms-swift forwards ``generate`` from a submodel."""
    try:
        import swift.model.utils as mu
    except ImportError:
        return
    if getattr(mu, "_sgvlm_causal_lm_loader_patched", False):
        return
    mu._sgvlm_causal_lm_loader_patched = True
    _orig = mu.use_submodel_func

    def _wrapped(model: Any, submodel_name: str, func_list=None) -> None:
        sub = getattr(model, submodel_name, None)
        if sub is not None:
            _patch_causal_lm_class(type(sub))
            _ensure_generation_config(sub)
        return _orig(model, submodel_name, func_list)

    mu.use_submodel_func = _wrapped  # type: ignore[assignment]

    for mod_name in (
        "swift.model.models.internlm",
        "swift.model.models.internvl",
        "swift.model.models.deepseek",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "use_submodel_func"):
            mod.use_submodel_func = _wrapped


def patch_remote_causal_lm() -> None:
    patch_swift_submodel_loader()
    try:
        import transformers.dynamic_module_utils as dmu
    except ImportError:
        _patch_loaded_causal_lm_modules()
        return

    if getattr(dmu, "_sgvlm_causal_lm_patched", False):
        _patch_loaded_causal_lm_modules()
        return
    dmu._sgvlm_causal_lm_patched = True

    _orig = dmu.get_class_from_dynamic_module

    def _wrap(*args, **kwargs):
        return _patch_causal_lm_class(_orig(*args, **kwargs))

    dmu.get_class_from_dynamic_module = _wrap  # type: ignore[method-assign]
    _patch_loaded_causal_lm_modules()


def _ensure_generation_config(module: Any) -> None:
    if module is None:
        return
    try:
        import transformers
        from transformers import GenerationConfig
    except ImportError:
        return

    gc = getattr(module, "generation_config", None)
    if gc is None:
        cfg = getattr(module, "config", None)
        try:
            gc = GenerationConfig.from_model_config(cfg) if cfg is not None else GenerationConfig()
        except Exception:
            gc = GenerationConfig()
        module.generation_config = gc
    if getattr(gc, "transformers_version", None) is None:
        gc.transformers_version = transformers.__version__


def patch_transformers_llama_flash_compat() -> None:
    """DeepSeek-VL2 remote code imports ``LlamaFlashAttention2`` removed in transformers 4.5x."""
    try:
        import transformers.models.llama.modeling_llama as llama_mod
    except ImportError:
        return
    if not hasattr(llama_mod, "LlamaFlashAttention2"):
        llama_mod.LlamaFlashAttention2 = llama_mod.LlamaAttention  # type: ignore[attr-defined]


def patch_remote_model(model: Any) -> None:
    if model is None:
        return
    for attr in ("language_model", "language", "llm", "text_model", "model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            _patch_causal_lm_class(type(sub))
            _ensure_generation_config(sub)


def patch_deepseek_dynamic_cache_compat() -> None:
    """DeepSeek-VL2 remote code expects Cache APIs removed in transformers 4.5x."""
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except ImportError:
        return

    if not hasattr(Cache, "get_usable_length"):
        def _get_usable_length(self, seq_length=None, layer_idx=0):  # noqa: ARG001
            if hasattr(self, "get_seq_length"):
                try:
                    return self.get_seq_length(layer_idx)
                except TypeError:
                    return self.get_seq_length()
            return 0

        Cache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(  # type: ignore[attr-defined]
            lambda self: self.get_seq_length() if hasattr(self, "get_seq_length") else 0
        )


def patch_deepseek_llama_attention_compat() -> None:
    """DeepSeek MHA layers call ``LlamaAttention`` with the pre-4.57 API."""
    try:
        import torch
        import transformers.models.llama.modeling_llama as llama_mod
    except ImportError:
        return
    if getattr(llama_mod, "_sgvlm_deepseek_attn_patched", False):
        return
    llama_mod._sgvlm_deepseek_attn_patched = True
    OrigAttention = llama_mod.LlamaAttention

    class DeepseekMhaLlamaAttention(OrigAttention):
        def __init__(self, config, layer_idx):
            super().__init__(config, layer_idx)
            self.rotary_emb = llama_mod.LlamaRotaryEmbedding(config=config)

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            pk = past_key_values if past_key_values is not None else past_key_value
            if position_embeddings is None:
                if position_ids is None:
                    batch_size, seq_len = hidden_states.shape[:2]
                    device = hidden_states.device
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
            attn_output, attn_weights = super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=pk,
                cache_position=cache_position,
            )
            if use_cache:
                return attn_output, attn_weights, pk
            return attn_output, attn_weights, None

    llama_mod.LlamaAttention = DeepseekMhaLlamaAttention  # type: ignore[misc]
    llama_mod.LlamaFlashAttention2 = DeepseekMhaLlamaAttention  # type: ignore[misc]


def patch_deepseek_vl2_config_for_vllm() -> None:
    """vLLM 0.21 expects ``text_config``; DeepSeek-VL2 remote code uses ``language_config``."""
    try:
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
    except ImportError:
        return
    if hasattr(DeepseekVLV2Config, "text_config"):
        return
    DeepseekVLV2Config.text_config = property(lambda self: self.language_config)  # type: ignore[attr-defined]


def patch_remote_engine(engine: Any) -> None:
    model = getattr(engine, "model", None)
    patch_remote_model(model)
    _ensure_generation_config(model)


def patch_deepseek_modelscope_cache() -> None:
    """Patch DeepSeek-VL2 git checkout for transformers 4.5x + vLLM worker subprocesses."""
    cache_root = Path.home() / ".cache/modelscope/hub/_github/DeepSeek-VL2/deepseek_vl2/models"
    deepseek_py = cache_root / "modeling_deepseek.py"
    vl2_py = cache_root / "modeling_deepseek_vl_v2.py"
    if deepseek_py.is_file():
        text = deepseek_py.read_text(encoding="utf-8")
        replacements = [
            (
                "from transformers.models.llama.modeling_llama import (\n"
                "    LlamaAttention,\n"
                "    LlamaFlashAttention2\n"
                ")",
                "from transformers.models.llama.modeling_llama import LlamaAttention\n"
                "LlamaFlashAttention2 = LlamaAttention",
            ),
            (
                "                past_length = past_key_values.seen_tokens\n"
                "                max_cache_length = past_key_values.get_max_length()",
                "                past_length = past_key_values.get_seq_length()\n"
                "                max_cache_length = None",
            ),
        ]
        updated = text
        for old, new in replacements:
            if old in updated and new not in updated:
                updated = updated.replace(old, new)
        if updated != text:
            deepseek_py.write_text(updated, encoding="utf-8")
    if vl2_py.is_file():
        text = vl2_py.read_text(encoding="utf-8")
        needle = "            self.language_config = DeepseekV2Config(**language_config)\n\n        self.tile_tag = tile_tag"
        insert = (
            "            self.language_config = DeepseekV2Config(**language_config)\n\n"
            "        self.text_config = self.language_config\n\n        self.tile_tag = tile_tag"
        )
        if needle in text and insert not in text:
            vl2_py.write_text(text.replace(needle, insert), encoding="utf-8")


def _lmdeploy_consumer_gpu() -> bool:
    """RTX 50xx (sm_120): limited SMEM; TurboMind unstable — prefer PyTorch engine."""
    import os

    if os.environ.get("LMDEPLOY_FORCE_PYTORCH", "").strip() in ("1", "true", "yes"):
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        smem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
        return cap[0] >= 12 and smem <= 102400
    except Exception:
        return False


def patch_lmdeploy_consumer_gpu() -> None:
    """5060 Ti / sm_120: force PyTorch backend; shrink Triton flash-attn tile sizes."""
    if not _lmdeploy_consumer_gpu():
        return
    try:
        import lmdeploy.archs as archs
    except ImportError:
        return
    if getattr(archs, "_sgvlm_consumer_gpu_patched", False):
        return
    archs._sgvlm_consumer_gpu_patched = True

    _orig_autoget_backend = archs.autoget_backend

    def _autoget_backend(model_path: str):
        backend = _orig_autoget_backend(model_path)
        if backend == "turbomind":
            archs.logger.warning(
                "SceneGraphVLM: forcing PyTorch engine on consumer GPU (sm_120 / limited SMEM)."
            )
            return "pytorch"
        return backend

    archs.autoget_backend = _autoget_backend

    try:
        import torch

        import lmdeploy.pytorch.kernels.cuda.flashattention as fa
    except ImportError:
        return

    smem = None
    if torch.cuda.is_available():
        smem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor

    if hasattr(fa, "_kernel_meta_sm12x") and smem is not None and smem <= 102400:
        if not getattr(fa, "_sgvlm_sm12x_smem_patched", False):
            fa._sgvlm_sm12x_smem_patched = True
            _orig_sm12x = fa._kernel_meta_sm12x

            def _kernel_meta_sm12x(BLOCK_DK: int, shared_kv: bool):
                # 5060 Ti SMEM ~101 KiB; default sm12x tiles need ~168 KiB.
                return 32, 32, 4, 2

            fa._kernel_meta_sm12x = _kernel_meta_sm12x

    if not hasattr(fa, "flash_attention_fwd"):
        return
    if getattr(fa, "_sgvlm_smem_patched", False):
        return
    fa._sgvlm_smem_patched = True
    _orig_fwd = fa.flash_attention_fwd
    _orig_sm9x = fa._kernel_meta_sm9x

    def _kernel_meta_consumer(BLOCK_DK: int, shared_kv: bool):
        # 5060 Ti SMEM ~101 KiB; sm9x defaults need ~124 KiB.
        return 32, 32, 4, 2

    def _flash_attention_fwd(*args, **kwargs):
        import torch

        if fa._nv_cap is None:
            fa._nv_cap = torch.cuda.get_device_capability()
        smem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
        if fa._nv_cap[0] >= 9 and smem <= 102400:
            fa._kernel_meta_sm9x = _kernel_meta_consumer
        try:
            return _orig_fwd(*args, **kwargs)
        finally:
            fa._kernel_meta_sm9x = _orig_sm9x

    fa.flash_attention_fwd = _flash_attention_fwd
    try:
        import lmdeploy.pytorch.kernels.cuda as cuda_kernels

        cuda_kernels.flash_attention_fwd = _flash_attention_fwd
    except ImportError:
        pass


def patch_lmdeploy_qwen35_consumer_kernels() -> None:
    """Qwen3.5 GDN uses TileLang causal_conv1d; default tiles exceed sm_120 SMEM (~101 KiB)."""
    if not _lmdeploy_consumer_gpu():
        return
    try:
        import lmdeploy.pytorch.kernels.cuda.causal_conv1d as cc
    except ImportError:
        return
    if getattr(cc, "_sgvlm_qwen35_smem_patched", False):
        return
    cc._sgvlm_qwen35_smem_patched = True

    _orig_fwd = cc.causal_conv1d_fwd
    _orig_update_fwd = cc.causal_conv1d_update_fwd

    def _causal_conv1d_fwd(*args, **kwargs):
        if len(args) >= 8:
            args = (*args[:7], 2)
        else:
            kwargs.setdefault("num_warps", 2)
        kwargs.setdefault("ChunkSizeL", 16)
        return _orig_fwd(*args, **kwargs)

    def _causal_conv1d_update_fwd(*args, **kwargs):
        if len(args) >= 11:
            args = (*args[:10], 1)
        else:
            kwargs.setdefault("num_warps", 1)
        return _orig_update_fwd(*args, **kwargs)

    cc.causal_conv1d_fwd = _causal_conv1d_fwd
    cc.causal_conv1d_update_fwd = _causal_conv1d_update_fwd

    try:
        import lmdeploy.pytorch.backends.cuda.causal_conv1d as cc_builder
    except ImportError:
        return

    _orig_build = cc_builder.CausalConv1dCudaBuilder.build

    @staticmethod
    def _build() -> cc_builder.CausalConv1dImpl:
        if cc_builder.has_dao():
            return cc_builder.CausalConv1dDaoImpl()
        return _orig_build()

    cc_builder.CausalConv1dCudaBuilder.build = _build  # type: ignore[method-assign]

    try:
        import lmdeploy.pytorch.kernels.cuda.gated_delta_rule as gdr
    except ImportError:
        return

    _orig_gdr = gdr.fused_recurrent_gated_delta_rule
    _orig_gdr_kernel = gdr.fused_recurrent_gated_delta_rule_fwd

    def _gdr_kernel_fwd(*args, **kwargs):
        kwargs["num_warps"] = 1
        return _orig_gdr_kernel(*args, **kwargs)

    def _gdr_fwd(*args, **kwargs):
        gdr.fused_recurrent_gated_delta_rule_fwd = _gdr_kernel_fwd
        try:
            return _orig_gdr(*args, **kwargs)
        finally:
            gdr.fused_recurrent_gated_delta_rule_fwd = _orig_gdr_kernel

    gdr.fused_recurrent_gated_delta_rule = _gdr_fwd


def patch_lmdeploy_api_compat() -> None:
    """Shim lmdeploy API moves so ms-swift 4.2 LmdeployEngine works on lmdeploy 0.10–0.13."""
    import sys

    try:
        import lmdeploy.api as api
        from lmdeploy.archs import autoget_backend_config
    except ImportError:
        return
    if not hasattr(api, "autoget_backend_config"):
        api.autoget_backend_config = autoget_backend_config

    try:
        import lmdeploy.serve.core.async_engine as async_engine_mod
    except ImportError:
        return
    if not hasattr(async_engine_mod, "best_match_model"):
        async_engine_mod.best_match_model = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules.setdefault("lmdeploy.serve.async_engine", async_engine_mod)


def patch_swift_lmdeploy_engine() -> None:
    """Consumer GPUs (5060 Ti): eager PyTorch backend; allow lmdeploy>=0.9 for Qwen3.5."""
    try:
        import swift.infer_engine.lmdeploy_engine as le
        from swift.infer_engine.lmdeploy_engine import LmdeployEngine
        from transformers.utils.versions import require_version as _orig_require_version
    except ImportError:
        return
    if getattr(le, "_sgvlm_lmdeploy_patched", False):
        return
    le._sgvlm_lmdeploy_patched = True

    def _require_version(requirement, hint=None):
        if isinstance(requirement, str) and "lmdeploy<0.9" in requirement.replace(" ", ""):
            return
        return _orig_require_version(requirement, hint)

    le.require_version = _require_version

    _orig_prepare = LmdeployEngine._prepare_engine_kwargs

    def _prepare_engine_kwargs(self, engine_kwargs):
        if engine_kwargs is None:
            engine_kwargs = {}
        else:
            engine_kwargs = dict(engine_kwargs)
        engine_kwargs.setdefault("eager_mode", False)
        if engine_kwargs.get("max_prefill_token_num", 8192) > 4096:
            engine_kwargs["max_prefill_token_num"] = 4096
        _orig_prepare(self, engine_kwargs)
        bc = self.backend_config
        if hasattr(bc, "eager_mode"):
            bc.eager_mode = False
        if hasattr(bc, "max_prefill_token_num") and (bc.max_prefill_token_num or 0) > 4096:
            bc.max_prefill_token_num = 4096

    LmdeployEngine._prepare_engine_kwargs = _prepare_engine_kwargs

    _patch_lmdeploy_stream_session_end(LmdeployEngine)


def _patch_lmdeploy_stream_session_end(LmdeployEngine) -> None:
    """ms-swift stream infer omits inst.async_end() for PyTorch backend; 2nd request → CUDA crash."""
    import lmdeploy
    import time
    from packaging import version

    from swift.infer_engine.protocol import (
        ChatCompletionResponseStreamChoice,
        ChatCompletionStreamResponse,
        DeltaMessage,
    )
    from swift.infer_engine.utils import InferStreamer

    if getattr(LmdeployEngine, "_sgvlm_stream_session_patched", False):
        return
    LmdeployEngine._sgvlm_stream_session_patched = True

    async def _infer_stream_async(self, inputs, generation_config, request_config):
        session_id = time.time_ns()
        kwargs = {
            "stream_output": True,
            "gen_config": generation_config,
            "sequence_start": True,
            "sequence_end": True,
        }
        if version.parse(lmdeploy.__version__) >= version.parse("0.6.5"):
            async with self.engine.model_inst(session_id) as inst:
                context = self.engine.safe_run(inst, session_id, **inputs, **kwargs)
                infer_streamer = InferStreamer(self.template)
                token_idx = 0
                async with context as gen:
                    is_finished = False
                    while not is_finished:
                        try:
                            output = await gen.__anext__()
                        except StopAsyncIteration:
                            is_finished = True
                        delta_text = infer_streamer.get_printable_text(output.token_ids, is_finished)
                        if not delta_text and not is_finished:
                            continue
                        logprobs = self._get_logprobs(
                            output.logprobs, output.token_ids[token_idx:], request_config.top_logprobs
                        )
                        token_idx = len(output.token_ids)
                        usage_info = self._get_usage_info(len(inputs["input_ids"]), output.num_token)
                        toolcall = None
                        if is_finished:
                            toolcall = self._get_toolcall(self.template.decode(output.token_ids))
                        finish_reason = self._get_finish_reason(
                            generation_config.max_new_tokens,
                            output.num_token,
                            output.status.name == "FINISH",
                        )
                        choices = [
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant", content=delta_text, tool_calls=toolcall),
                                finish_reason=finish_reason,
                                logprobs=logprobs,
                            )
                        ]
                        yield ChatCompletionStreamResponse(
                            model=self.model_name, choices=choices, usage=usage_info
                        )
                if self.engine.backend == "pytorch":
                    from lmdeploy.pytorch.engine.request import RequestType

                    await inst.req_sender.async_send(
                        RequestType.END_SESSION, dict(session_id=session_id)
                    )
            return

        context = self.engine.safe_run(session_id)
        infer_streamer = InferStreamer(self.template)
        token_idx = 0
        async with context:
            generator = await self.engine.get_generator(False, session_id)
            gen = generator.async_stream_infer(session_id=session_id, **inputs, **kwargs)
            is_finished = False
            while not is_finished:
                try:
                    output = await gen.__anext__()
                except StopAsyncIteration:
                    is_finished = True
                delta_text = infer_streamer.get_printable_text(output.token_ids, is_finished)
                if not delta_text and not is_finished:
                    continue
                logprobs = self._get_logprobs(
                    output.logprobs, output.token_ids[token_idx:], request_config.top_logprobs
                )
                token_idx = len(output.token_ids)
                usage_info = self._get_usage_info(len(inputs["input_ids"]), output.num_token)
                toolcall = None
                if is_finished:
                    toolcall = self._get_toolcall(self.template.decode(output.token_ids))
                finish_reason = self._get_finish_reason(
                    generation_config.max_new_tokens,
                    output.num_token,
                    output.status.name == "FINISH",
                )
                choices = [
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant", content=delta_text, tool_calls=toolcall),
                        finish_reason=finish_reason,
                        logprobs=logprobs,
                    )
                ]
                yield ChatCompletionStreamResponse(model=self.model_name, choices=choices, usage=usage_info)

    LmdeployEngine._infer_stream_async = _infer_stream_async


def _needs_sglang_text_prompt(model_id_or_path: str) -> bool:
    """InternVL on sglang>=0.5 rejects pre-tokenized input_ids + raw image paths."""
    name = (model_id_or_path or "").lower()
    return "internvl" in name


def _internvl_sglang_processor_output(template: Any, inputs: dict) -> Optional[dict]:
    """Build sglang InternVL ``processor_output`` payload with expanded IMG tokens."""
    ti = inputs.get("template_inputs")
    if ti is None:
        return None
    old_mode = template.mode
    try:
        template.set_mode("transformers")
        encoded = template._encode(ti)
    finally:
        template.set_mode(old_mode)
    pixel_values = encoded.get("pixel_values")
    input_ids = encoded.get("input_ids")
    if pixel_values is None or input_ids is None:
        return None
    import torch

    return {
        "format": "processor_output",
        "pixel_values": pixel_values,
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
    }


def _sglang_engine_inputs(inputs: dict, *, template: Any = None, model_id_or_path: str = "") -> dict:
    """ms-swift 4.2 passes ``images``; sglang 0.5.x ``async_generate`` expects ``image_data``."""
    engine_inputs = {k: v for k, v in inputs.items() if k != "template_inputs"}
    if template is not None and _needs_sglang_text_prompt(model_id_or_path):
        po = _internvl_sglang_processor_output(template, inputs)
        if po is not None:
            engine_inputs["input_ids"] = po["input_ids"].flatten().tolist()
            engine_inputs["image_data"] = [po]
            engine_inputs.pop("images", None)
            engine_inputs.pop("audios", None)
            engine_inputs.pop("videos", None)
            return engine_inputs
    if "images" in engine_inputs and "image_data" not in engine_inputs:
        engine_inputs["image_data"] = engine_inputs.pop("images")
    if "audios" in engine_inputs and "audio_data" not in engine_inputs:
        engine_inputs["audio_data"] = engine_inputs.pop("audios")
    if "videos" in engine_inputs and "video_data" not in engine_inputs:
        engine_inputs["video_data"] = engine_inputs.pop("videos")
    return engine_inputs


def patch_sglang_smolvlm_compat() -> None:
    """sglang 0.5.12: transformers_auto references Modality.MULTI_IMAGES (removed)."""
    try:
        import sglang.srt.multimodal.processors.transformers_auto as ta
    except ImportError:
        return
    if getattr(ta, "_sgvlm_smolvlm_patched", False):
        return
    ta._sgvlm_smolvlm_patched = True

    def _build_mm_items(self, processor_output, input_ids):
        items = self.collect_mm_items_from_processor_output(processor_output)
        modality_to_token_id = {
            ta.Modality.IMAGE: self.mm_tokens.image_token_id,
            ta.Modality.VIDEO: self.mm_tokens.video_token_id,
            ta.Modality.AUDIO: self.mm_tokens.audio_token_id,
        }
        for item in items:
            token_id = modality_to_token_id.get(item.modality)
            if token_id is not None:
                item.offsets = self.get_mm_items_offset(input_ids, token_id)
        return items

    ta.TransformersAutoMultimodalProcessor._build_mm_items = _build_mm_items


def patch_swift_sglang_engine() -> None:
    """sglang>=0.5: rename multimodal kwargs for ms-swift SglangEngine."""
    try:
        import swift.infer_engine.sglang_engine as se
        from swift.infer_engine.sglang_engine import SglangEngine
    except ImportError:
        return
    if getattr(se, "_sgvlm_sglang_patched", False):
        return
    se._sgvlm_sglang_patched = True

    async def _infer_full_async(self, inputs, generation_config, request_config):
        engine_inputs = _sglang_engine_inputs(
            inputs, template=self.template, model_id_or_path=self.model_id_or_path
        )
        output = await self.engine.async_generate(**engine_inputs, sampling_params=generation_config)
        output["prompt_token_ids"] = output.get("meta_info", {}).get("prompt_tokens") or inputs.get("input_ids")
        return self._create_chat_completion_response(output, inputs, request_config.return_details)

    async def _infer_stream_async(self, inputs, generation_config, **kwargs):
        engine_inputs = _sglang_engine_inputs(
            inputs, template=self.template, model_id_or_path=self.model_id_or_path
        )
        result_generator = await self.engine.async_generate(
            **engine_inputs, sampling_params=generation_config, stream=True
        )
        from swift.infer_engine.utils import InferStreamer

        infer_streamer = InferStreamer(self.template)
        async for output in result_generator:
            res = self._create_chat_completion_stream_response(output, infer_streamer)
            if res is None:
                continue
            yield res

    SglangEngine._infer_full_async = _infer_full_async
    SglangEngine._infer_stream_async = _infer_stream_async


def apply_model_patches(model_id_or_path: str) -> None:
    name = (model_id_or_path or "").lower()
    if re.search(r"internvl2[_-]?5|internvl2_5|deepseek", name):
        patch_remote_causal_lm()
    if "deepseek" in name:
        patch_transformers_llama_flash_compat()
        patch_deepseek_llama_attention_compat()
        patch_deepseek_dynamic_cache_compat()
        patch_deepseek_modelscope_cache()
        patch_deepseek_vl2_config_for_vllm()
    patch_lmdeploy_api_compat()
    patch_lmdeploy_consumer_gpu()
    patch_lmdeploy_qwen35_consumer_kernels()
    patch_swift_lmdeploy_engine()
    patch_swift_sglang_engine()
    patch_sglang_smolvlm_compat()
