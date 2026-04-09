#!/usr/bin/env python
"""
End-to-end prediction video for one PVSG video id:

1) Filter a Swift JSONL to that video (temp file).
2) Run qwen-bench inference:
     GT_prompt  -> metrics/qwen-bench/infer/GT-prompt/infer_swift_gt_prompt.py
     GEN_prompt -> metrics/qwen-bench/infer/GEN-prompt/infer_swift_gen_prompt.py
3) Run metrics/qwen-bench/eval/eval_sgg_metrics_with_qwen.py with --per-sample-jsonl.
4) Render an MP4 with predictions + per-frame Qwen-eval metrics (IoU on matches + Qwen P/R/F1 + SGG; no vLLM step time).

Requires: same env as ms-swift inference; eval needs scipy + vLLM Qwen judge stack.

Default artifact layout (override with --metrics-output / --metrics-summary-output if needed):
  <predicts-root>/<GT_prompt|GEN_prompt>/<video-name>/
    {run_name}.jsonl       — raw inference
    frames_metrics.jsonl   — per-frame rows + metrics (eval --per-sample-jsonl)
    general_metrics.json   — eval summary JSON

--predicts-dir is the parent of GT_prompt (i.e. the predicts_and_metrics folder).
One GPU setup for the whole pipeline (infer then eval): --cuda-visible-devices, --gpu-memory-utilization, --batch-size apply to both steps.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# .../SceneGraphVLM/visualization/video_demo_results/scripts -> parents[2] == repo root
_REPO_ROOT = Path(_SCRIPT_DIR).resolve().parents[2]
_DEMO_ROOT = Path(_SCRIPT_DIR).resolve().parent
_DEFAULT_PREDICTS = _DEMO_ROOT / "predicts_and_metrics"
_DEFAULT_VIDEOS = _DEMO_ROOT / "videos_output"
_FRAMES_METRICS_FILENAME = "frames_metrics.jsonl"
_GENERAL_METRICS_FILENAME = "general_metrics.json"

_INFER_GT = _REPO_ROOT / "metrics/qwen-bench/infer/GT-prompt/infer_swift_gt_prompt.py"
_INFER_GEN = _REPO_ROOT / "metrics/qwen-bench/infer/GEN-prompt/infer_swift_gen_prompt.py"
_EVAL_SCRIPT = _REPO_ROOT / "metrics/qwen-bench/eval/eval_sgg_metrics_with_qwen.py"

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
from build_gt_video import (
    combine_image_with_metrics_table,
    create_qwen_eval_metrics_table,
    create_video_from_frames,
    draw_scene_graph,
    filter_jsonl_by_video,
    filter_video_frames_swift,
    get_prediction_text,
    load_jsonl,
    resolve_image_path,
    sample_image_ref,
)


def _default_frame_title(prompt_mode: str) -> str:
    if prompt_mode == "GT_prompt":
        return "Prediction (GT Prompt)"
    if prompt_mode == "GEN_prompt":
        return "Prediction (GEN Prompt)"
    return "Prediction"


def _run(cmd: list[str], *, cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, cwd=str(cwd or _REPO_ROOT), env=env)


def _render_video(
    *,
    metrics_jsonl: Path,
    images_base: str,
    video_name: str,
    out_path: Path,
    title: str,
    fps: float,
    max_width: int | None,
) -> int:
    all_samples = load_jsonl(str(metrics_jsonl))
    if not all_samples:
        print("ERROR: empty metrics jsonl", file=sys.stderr)
        return 1
    if "metrics" not in all_samples[0]:
        print(
            "ERROR: each line must contain 'metrics' (use eval with --per-sample-jsonl).",
            file=sys.stderr,
        )
        return 1

    video_samples = filter_video_frames_swift(all_samples, video_name)
    if not video_samples:
        print(f"ERROR: no frames for video {video_name!r} in {metrics_jsonl}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list = []
    for i, sample in enumerate(video_samples):
        print(f"Frame {i + 1}/{len(video_samples)}...", end="\r", flush=True)
        ref = sample_image_ref(sample)
        if not ref:
            continue
        ip = resolve_image_path(images_base, ref)
        if not os.path.isfile(ip):
            print(f"\nWARNING: missing image {ip}")
            continue
        pred = get_prediction_text(sample)
        if not pred:
            continue
        try:
            img = draw_scene_graph(ip, pred, max_width=max_width, title=title, title_wrap_long=True)
        except Exception as e:
            print(f"\nERROR frame {i}: {e}")
            continue
        metrics = sample.get("metrics") if isinstance(sample.get("metrics"), dict) else {}
        try:
            w, _ = img.size
            table = create_qwen_eval_metrics_table(metrics, w)
            combined = combine_image_with_metrics_table(img, table)
        except Exception:
            combined = img
        frames.append(np.array(combined))

    print(f"\nWrote {len(frames)} frames -> {out_path}")
    if not frames:
        return 1
    create_video_from_frames(frames, str(out_path), fps=fps)
    print("Done.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Infer (GT_prompt | GEN_prompt) + qwen-bench metrics + prediction MP4 for one video id."
    )
    p.add_argument(
        "--prompt-mode",
        choices=("GT_prompt", "GEN_prompt"),
        required=True,
        help="Which ms-swift infer script to run (GT-prompt batch vs GEN-prompt temporal chaining).",
    )
    p.add_argument(
        "--annotation-file",
        type=str,
        required=True,
        help="Full Swift JSONL; only rows whose image path contains --video-name are inferred.",
    )
    p.add_argument("--video-name", type=str, required=True, help="Substring of frame path, e.g. 0004_11566980553.")
    p.add_argument("--model", type=str, required=True, help="HF id or local checkpoint path for vLLM/Transformers.")
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Basename for raw predictions JSONL: {run_name}.jsonl. "
        "Default: {video_name}_{prompt_mode} (e.g. 0004_11566980553_GT_prompt). "
        "Override only if you need a different preds filename.",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default="",
        help="Convenience: writes preds to <root>/predicts_and_metrics/ and videos to <root>/videos_output/. "
        "Ignored for a path if you pass explicit --predicts-dir or --videos-dir for that side.",
    )
    p.add_argument(
        "--predicts-dir",
        type=str,
        default="",
        help=f"predicts_and_metrics root (parent of GT_prompt/ GEN_prompt/). Default: {_DEFAULT_PREDICTS} or <results-root>/predicts_and_metrics.",
    )
    p.add_argument(
        "--videos-dir",
        type=str,
        default="",
        help=f"Root for rendered MP4 tree (default: {_DEFAULT_VIDEOS} or <results-root>/videos_output).",
    )
    p.add_argument(
        "--videos-subdir",
        type=str,
        default="",
        help="With default video layout only: subfolder under --videos-dir (default: --prompt-mode). Ignored if --video-output-dir is set.",
    )
    p.add_argument(
        "--video-output-dir",
        type=str,
        default="",
        help="Directory for the output MP4. Final file is <dir>/<output-filename> (default: {video_name}_{prompt_mode}.mp4). Overrides --videos-dir / --videos-subdir / video-id subfolders.",
    )
    p.add_argument("--images-base", type=str, default="", help="Same as infer --images-base (relative image paths).")
    p.add_argument(
        "--output-filename",
        type=str,
        default="",
        help="MP4 basename only (default: {video_name}_{prompt_mode}.mp4). Used inside --video-output-dir or under the default videos tree.",
    )
    p.add_argument(
        "--metrics-output",
        type=str,
        default="",
        help=f"Override path for per-frame metrics JSONL (default: .../<prompt-mode>/<video>/{_FRAMES_METRICS_FILENAME}).",
    )
    p.add_argument(
        "--metrics-summary-output",
        type=str,
        default="",
        help=f"Override path for eval summary JSON (default: same folder as frames metrics, {_GENERAL_METRICS_FILENAME}).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="",
        help="Header on each frame (default: Prediction (GT Prompt) or Prediction (GEN Prompt) from --prompt-mode).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output MP4 frame rate (passed to ffmpeg -framerate).",
    )
    p.add_argument("--max-width", type=int, default=1920, help="0 = no resize by width.")

    p.add_argument(
        "--skip-infer",
        action="store_true",
        help="Reuse existing raw preds JSONL in the video artifact folder (default name: {video_name}_{prompt_mode}.jsonl).",
    )
    p.add_argument("--skip-eval", action="store_true", help="Skip eval; require existing per-sample metrics JSONL.")
    p.add_argument(
        "--metrics-jsonl",
        type=str,
        default="",
        help="Deprecated alias for --metrics-output (if --metrics-output is empty, this is used).",
    )
    p.add_argument(
        "--keep-filtered-jsonl",
        action="store_true",
        help="Do not delete the temp JSONL filtered to one video (debug).",
    )
    p.add_argument("--force-infer", action="store_true", help="Pass --force to the infer script.")

    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="",
        metavar="IDS",
        help="Both steps: CUDA_VISIBLE_DEVICES for infer subprocess and --cuda-visible-devices for eval (e.g. 4).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Infer --batch-size and eval --batch-size-qwen (runs one after another on the same GPU).",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        metavar="FRAC",
        help="vLLM gpu_memory_utilization for VL infer and for Qwen judge in eval (0..1).",
    )

    # Infer (shared)
    p.add_argument("--infer-backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--torch-dtype", type=str, default="bfloat16")
    p.add_argument("--template-type", type=str, default="")
    p.add_argument("--response-prefix", type=str, default="")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--auto-obj-prefix-fallback", action="store_true")
    p.add_argument("--no-stop-on-answer-close", action="store_true")

    # GEN-prompt only
    p.add_argument(
        "--gen-prev-source",
        choices=("model", "gt"),
        default="model",
        help="GEN_prompt only: previous-frame block from model or GT (--prev-source on infer_swift_gen_prompt).",
    )

    # Eval
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--max-new-tokens-qwen", type=int, default=16)
    p.add_argument("--qwen-model-path", type=str, default="", help="Synonym judge model (eval).")

    args = p.parse_args()

    for path, label in ((_INFER_GT, "GT infer"), (_INFER_GEN, "GEN infer"), (_EVAL_SCRIPT, "eval")):
        if not path.is_file():
            print(f"ERROR: missing {label} script: {path}", file=sys.stderr)
            return 1

    run_name = args.run_name.strip() or f"{args.video_name}_{args.prompt_mode}"

    results_root = (
        Path(args.results_root.strip()).expanduser().resolve() if args.results_root.strip() else None
    )
    if args.predicts_dir.strip():
        predicts_root = Path(args.predicts_dir.strip()).expanduser().resolve()
    elif results_root is not None:
        predicts_root = (results_root / "predicts_and_metrics").resolve()
    else:
        predicts_root = _DEFAULT_PREDICTS.resolve()

    artifact_dir = (predicts_root / args.prompt_mode / args.video_name).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.videos_dir.strip():
        videos_dir = Path(args.videos_dir.strip()).resolve()
    elif results_root is not None:
        videos_dir = results_root / "videos_output"
    else:
        videos_dir = _DEFAULT_VIDEOS.resolve()

    print(f"[paths] predicts_root={predicts_root}", flush=True)
    print(f"[paths] artifact_dir={artifact_dir}", flush=True)
    pred_jsonl = artifact_dir / f"{run_name}.jsonl"

    if args.metrics_output.strip():
        metrics_jsonl = Path(args.metrics_output.strip()).expanduser().resolve()
    elif args.metrics_jsonl.strip():
        metrics_jsonl = Path(args.metrics_jsonl.strip()).expanduser().resolve()
    else:
        metrics_jsonl = artifact_dir / _FRAMES_METRICS_FILENAME
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.metrics_summary_output.strip():
        summary_path = Path(args.metrics_summary_output.strip()).expanduser().resolve()
    else:
        summary_path = metrics_jsonl.parent / _GENERAL_METRICS_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary_dir = str(summary_path.parent)
    eval_summary_name = summary_path.name

    out_name = args.output_filename.strip() or f"{args.video_name}_{args.prompt_mode}.mp4"
    if not out_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        out_name += ".mp4"
    if args.video_output_dir.strip():
        video_out_dir = Path(args.video_output_dir.strip()).expanduser().resolve()
        video_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = video_out_dir / out_name
    else:
        videos_subdir = args.videos_subdir.strip() if args.videos_subdir.strip() else args.prompt_mode
        out_path = videos_dir / videos_subdir / args.video_name / out_name
        print(f"[paths] videos_dir={videos_dir}", flush=True)

    print(f"[paths] metrics_jsonl={metrics_jsonl}", flush=True)
    print(f"[paths] metrics_summary={summary_path}", flush=True)
    print(f"[paths] video_out={out_path}", flush=True)

    max_width = args.max_width if args.max_width > 0 else None
    images_base = args.images_base or ""

    cuda_env: dict[str, str] | None = None
    if args.cuda_visible_devices.strip():
        cuda_env = {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices.strip()}
    print(
        f"[gpu] cuda_visible_devices={args.cuda_visible_devices or '(inherit)'} "
        f"batch_size={args.batch_size} gpu_memory_utilization={args.gpu_memory_utilization}",
        flush=True,
    )

    tmp_jsonl: str | None = None
    try:
        if not args.skip_infer:
            fd, tmp_jsonl = tempfile.mkstemp(suffix=".jsonl", prefix=f"pvsg_{args.video_name}_")
            os.close(fd)
            tmp_path = Path(tmp_jsonl)
            n_in, n_out = filter_jsonl_by_video(Path(args.annotation_file), tmp_path, args.video_name)
            print(f"[filter] kept {n_out}/{n_in} lines for video {args.video_name!r} -> {tmp_path}")
            if n_out == 0:
                return 1

            infer_script = _INFER_GT if args.prompt_mode == "GT_prompt" else _INFER_GEN
            cmd: list[str] = [
                sys.executable,
                str(infer_script),
                "--model",
                args.model,
                "--test-jsonl",
                str(tmp_path.resolve()),
                "--output-dir",
                str(artifact_dir),
                "--run-name",
                run_name,
                "--batch-size",
                str(args.batch_size),
                "--infer-backend",
                args.infer_backend,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--max-model-len",
                str(args.max_model_len),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--torch-dtype",
                args.torch_dtype,
            ]
            if images_base:
                cmd.extend(["--images-base", images_base])
            if args.template_type.strip():
                cmd.extend(["--template-type", args.template_type.strip()])
            if args.response_prefix.strip():
                cmd.extend(["--response-prefix", args.response_prefix.strip()])
            if args.enable_thinking:
                cmd.append("--enable-thinking")
            if args.auto_obj_prefix_fallback:
                cmd.append("--auto-obj-prefix-fallback")
            if args.no_stop_on_answer_close:
                cmd.append("--no-stop-on-answer-close")
            if args.force_infer:
                cmd.append("--force")
            if args.prompt_mode == "GEN_prompt":
                cmd.extend(["--prev-source", args.gen_prev_source])

            _run(cmd, extra_env=cuda_env)
        else:
            if not pred_jsonl.is_file():
                print(f"ERROR: --skip-infer but missing {pred_jsonl}", file=sys.stderr)
                return 1

        if not args.skip_eval:
            eval_cmd = [
                sys.executable,
                str(_EVAL_SCRIPT),
                "--pred-jsonl",
                str(pred_jsonl),
                "--output-dir",
                eval_summary_dir,
                "--output-name",
                eval_summary_name,
                "--per-sample-jsonl",
                str(metrics_jsonl),
                "--iou-thr",
                str(args.iou_thr),
                "--batch-size-qwen",
                str(args.batch_size),
                "--max-new-tokens-qwen",
                str(args.max_new_tokens_qwen),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
            if args.qwen_model_path.strip():
                eval_cmd.extend(["--qwen-model-path", args.qwen_model_path.strip()])
            if args.cuda_visible_devices.strip():
                eval_cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices.strip()])
            _run(eval_cmd, extra_env=cuda_env)
        else:
            if not metrics_jsonl.is_file():
                print(f"ERROR: --skip-eval but missing {metrics_jsonl}", file=sys.stderr)
                return 1

        frame_title = args.title.strip() or _default_frame_title(args.prompt_mode)
        return _render_video(
            metrics_jsonl=metrics_jsonl,
            images_base=images_base,
            video_name=args.video_name,
            out_path=out_path,
            title=frame_title,
            fps=float(args.fps),
            max_width=max_width,
        )
    finally:
        if tmp_jsonl and not args.keep_filtered_jsonl:
            try:
                os.unlink(tmp_jsonl)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
