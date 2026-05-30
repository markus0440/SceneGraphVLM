#!/usr/bin/env python
"""
Stack three MP4s into one layout (synced by frame time, shortest stream ends output):

  Row 1: top video (typically GT), centered in a fixed slot (letterboxed).
  Row 2: left + right videos side by side (e.g. GT_prompt vs GEN_prompt).

  Padding / letterbox color is white; the three video cells share the same pixel size
  (half of canvas width × panel height); the top cell is centered on the full canvas width.

Layout by folder name (recommended): pass VIDEO_NAME only — looks under videos_output/ for
  GT/VIDEO_NAME/VIDEO_NAME_GT.mp4,
  GT_prompt/VIDEO_NAME/VIDEO_NAME_GT_prompt.mp4,
  GEN_prompt/VIDEO_NAME/VIDEO_NAME_GEN_prompt.mp4
and writes videos_output/COMMON/VIDEO_NAME/VIDEO_NAME_COMMON.mp4

Explicit mode: --top, --bottom-left, --bottom-right, -o

Requires ffmpeg in PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# .../video_demo_results/scripts/build_common_video.py -> video_demo_results/videos_output
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_VIDEOS_ROOT = (_SCRIPT_DIR.parent / "videos_output").resolve()


def _ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found in PATH.")
    return ff


def _ffprobe(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def _video_stream(info: dict) -> dict:
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    raise ValueError("No video stream found")


def _parse_fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        s = stream.get(key) or "0/0"
        if "/" in s:
            a, b = s.split("/", 1)
            try:
                num, den = float(a), float(b)
                if den:
                    return num / den
            except ValueError:
                pass
    return 25.0


def _probe(path: str) -> tuple[int, int, float, float]:
    info = _ffprobe(path)
    vs = _video_stream(info)
    w, h = int(vs["width"]), int(vs["height"])
    fps = _parse_fps(vs)
    try:
        dur = float(info.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        dur = 0.0
    return w, h, fps, dur


def main() -> int:
    p = argparse.ArgumentParser(
        description="Merge three videos: top center (GT), bottom row (two clips).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (paths under video_demo_results/videos_output):\n"
            "  %(prog)s 0004_11566980553\n"
            "  %(prog)s 0004_11566980553 --videos-root /path/to/videos_output"
        ),
    )
    p.add_argument(
        "video_name",
        nargs="?",
        default=None,
        help=(
            "Folder name under videos_output (e.g. 0004_11566980553). "
            "Resolves GT / GT_prompt / GEN_prompt inputs and writes COMMON/<name>/<name>_COMMON.mp4. "
            "If omitted, use --top, --bottom-left, --bottom-right, and -o."
        ),
    )
    p.add_argument(
        "--videos-root",
        type=str,
        default="",
        help=(
            "Directory that contains GT/, GT_prompt/, GEN_prompt/, COMMON/ "
            f"(default: {_DEFAULT_VIDEOS_ROOT})."
        ),
    )
    p.add_argument(
        "--top",
        default=None,
        type=str,
        help="Top-center video (explicit mode only).",
    )
    p.add_argument(
        "--bottom-left",
        default=None,
        type=str,
        help="Bottom-left video (explicit mode only).",
    )
    p.add_argument(
        "--bottom-right",
        default=None,
        type=str,
        help="Bottom-right video (explicit mode only).",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        type=str,
        help="Output MP4 path (explicit mode only).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Canvas width (top and bottom row use full width).",
    )
    p.add_argument(
        "--top-height",
        type=int,
        default=720,
        help="Panel height for all three videos (same size; letterbox/pillarbox inside each cell).",
    )
    p.add_argument(
        "--bottom-height",
        type=int,
        default=720,
        help="Deprecated: ignored; use --top-height for height of every panel.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output FPS (0 = use --top video's FPS).",
    )
    args = p.parse_args()

    videos_root = (
        Path(args.videos_root).expanduser().resolve()
        if (args.videos_root or "").strip()
        else _DEFAULT_VIDEOS_ROOT
    )

    explicit = (
        args.top is not None
        and args.bottom_left is not None
        and args.bottom_right is not None
        and args.output is not None
    )
    by_name = args.video_name is not None and str(args.video_name).strip() != ""

    if by_name and explicit:
        print(
            "ERROR: pass either VIDEO_NAME or explicit --top/--bottom-left/--bottom-right/-o, not both.",
            file=sys.stderr,
        )
        return 2
    if not by_name and not explicit:
        print(
            "ERROR: provide VIDEO_NAME (folder under videos_output) or all of "
            "--top, --bottom-left, --bottom-right, --output.",
            file=sys.stderr,
        )
        return 2

    out_path: Path
    if by_name:
        name = Path(str(args.video_name).strip()).name.strip()
        if not name or name in (".", ".."):
            print("ERROR: invalid VIDEO_NAME.", file=sys.stderr)
            return 1
        top_p = str(videos_root / "GT" / name / f"{name}_GT.mp4")
        bl_p = str(videos_root / "GT_prompt" / name / f"{name}_GT_prompt.mp4")
        br_p = str(videos_root / "GEN_prompt" / name / f"{name}_GEN_prompt.mp4")
        out_path = (videos_root / "COMMON" / name / f"{name}_COMMON.mp4").resolve()
    else:
        assert args.top is not None and args.bottom_left is not None
        assert args.bottom_right is not None and args.output is not None
        top_p = os.path.abspath(args.top)
        bl_p = os.path.abspath(args.bottom_left)
        br_p = os.path.abspath(args.bottom_right)
        out_path = Path(args.output).expanduser().resolve()

    for path, tag in ((top_p, "top"), (bl_p, "bottom-left"), (br_p, "bottom-right")):
        if not os.path.isfile(path):
            print(f"ERROR: input not a file ({tag}): {path}", file=sys.stderr)
            return 1

    W = max(320, args.width)
    Ch = max(120, args.top_height)
    half_left = W // 2
    half_right = W - half_left
    fps = float(args.fps) if args.fps and args.fps > 0 else None

    _, _, fps_top, _ = _probe(top_p)
    if fps is None:
        fps = fps_top if fps_top > 0 else 25.0

    # Same cell size (half_left x Ch) for all three; top cell centered on full width W.
    # fps -> scale -> pad (white); setpts; shortest on stacks
    lines = [
        f"[0:v]fps={fps},scale={half_left}:{Ch}:force_original_aspect_ratio=decrease,"
        f"pad={half_left}:{Ch}:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1,setpts=PTS-STARTPTS[gtc]",
        f"[gtc]pad={W}:{Ch}:(ow-iw)/2:(oh-ih)/2:color=white[gt]",
        f"[1:v]fps={fps},scale={half_left}:{Ch}:force_original_aspect_ratio=decrease,"
        f"pad={half_left}:{Ch}:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1,setpts=PTS-STARTPTS[bl]",
        f"[2:v]fps={fps},scale={half_right}:{Ch}:force_original_aspect_ratio=decrease,"
        f"pad={half_right}:{Ch}:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1,setpts=PTS-STARTPTS[br]",
        # Options on one filter are ':'-separated; ',' starts the next filter in the chain.
        "[bl][br]hstack=inputs=2:shortest=1[bot]",
        "[gt][bot]vstack=inputs=2:shortest=1[stack]",
        "[stack]format=yuv420p[outv]",
    ]

    # Semicolons separate filter *chains*; newlines alone make ffmpeg treat the rest as garbage.
    graph = ";".join(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ffconcat", delete=False, encoding="utf-8"
    ) as gf:
        gf.write(graph)
        graph_path = gf.name

    try:
        cmd = [
            _ffmpeg(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            top_p,
            "-i",
            bl_p,
            "-i",
            br_p,
            "-filter_complex_script",
            graph_path,
            "-map",
            "[outv]",
            "-an",
            "-r",
            str(fps),
            str(out_path),
        ]
        print("+ ffmpeg -i ... -filter_complex_script", graph_path, "->", out_path, flush=True)
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.unlink(graph_path)
        except OSError:
            pass

    print(f"Done -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
