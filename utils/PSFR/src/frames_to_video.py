import re
import os
import shutil
import argparse
import tempfile
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional


# ----------------------------- backend detection -----------------------------

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ----------------------------- helpers ---------------------------------------

_NUMERIC_RE = re.compile(r"^(\d+)$")


def _natural_key(p: Path):
    """Sort helper: numeric stems by integer, else lexicographic path."""

    m = _NUMERIC_RE.match(p.stem)
    return (0, int(m.group(1))) if m else (1, p.name.lower())

def _discover_sequence(frames_dir: str, ext: str) -> Tuple[List[Path], Optional[int], Optional[int]]:
    """
    Returns: (sorted_paths, start_number, pad_width)
      - start_number/pad_width set only if all names are strictly numeric AND contiguous.
    """

    d = Path(frames_dir)
    if not d.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    candidates = sorted(
        [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == f".{ext.lower()}"],
        key=_natural_key
    )
    if not candidates:
        raise FileNotFoundError(f"No *.{ext} frames in: {frames_dir}")

    indices: List[int] = []
    widths: List[int] = []
    numeric_paths: List[Path] = []

    for p in candidates:
        m = _NUMERIC_RE.match(p.stem)
        if m:
            s = m.group(1)
            indices.append(int(s))
            widths.append(len(s))
            numeric_paths.append(p)

    # All numeric?
    if numeric_paths and len(numeric_paths) == len(candidates):
        # Already sorted with _natural_key
        sorted_paths = candidates
        sorted_idx = indices
        start = sorted_idx[0]
        contiguous = all(sorted_idx[i] == start + i for i in range(len(sorted_idx)))
        if contiguous:
            from collections import Counter
            pad_width = Counter(widths).most_common(1)[0][0]
            return sorted_paths, start, pad_width
        else:
            return sorted_paths, None, None
    else:
        # Non-numeric naming
        return candidates, None, None


def _ffmpeg_codec_params(codec: str, crf: int, preset: str, pix_fmt: str):
    codec = codec.lower()
    if codec == "h264":
        return "libx264", ["-crf", str(int(crf)), "-preset", str(preset)], pix_fmt
    if codec == "vp9":
        return "libvpx-vp9", ["-crf", str(int(crf)), "-b:v", "0"], pix_fmt
    if codec == "mpeg4":
        qscale = "2" if crf <= 20 else "5"
        return "mpeg4", ["-qscale:v", qscale], pix_fmt
    raise ValueError(f"Unsupported codec: {codec}. Choose from: h264, vp9, mpeg4")


def _quote_for_ffmpeg(p: Path) -> str:
    # Use POSIX path and escape single quotes
    s = p.as_posix().replace("'", "'\\''")
    return f"file '{s}'\n"


# ----------------------------- ffmpeg composers ------------------------------

def compose_with_ffmpeg_concat(
    paths: List[Path],
    out_video: str,
    fps: float,
    overwrite: bool,
    codec: str,
    crf: int,
    preset: str,
    pix_fmt: str
) -> None:
    """
    Concat demuxer + explicit per-frame durations, encoded as CFR at --fps.
    Total length = len(paths) / fps exactly.
    """

    if fps <= 0:
        raise ValueError("--fps must be > 0")
    dt = 1.0 / float(fps)

    vcodec, extra, pix_f = _ffmpeg_codec_params(codec, crf, preset, pix_fmt)
    outp = Path(out_video)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Build ffconcat list
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False)
    list_path = tmp.name
    try:
        tmp.write("ffconcat version 1.0\n")
        for p in paths:
            tmp.write(_quote_for_ffmpeg(p))  # file '...'
            tmp.write(f"duration {dt:.16f}\n")  # seconds per image
        # ffmpeg ignores the last 'duration' unless you repeat the last file
        tmp.write(_quote_for_ffmpeg(paths[-1]))
        tmp.flush()
        tmp.close()

        # IMPORTANT:
        # -vsync cfr  -> constant frame rate synthesis from timestamps
        # -r <fps>    -> target CFR
        # (Do NOT combine -r with -vsync vfr)
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-y" if overwrite else "-n",
            "-safe", "0",
            "-f", "concat",
            "-i", list_path,
            "-vsync", "cfr",
            "-r", str(float(fps)),
            "-c:v", vcodec,
            *extra,
            "-pix_fmt", pix_f,
            "-movflags", "+faststart",
            out_video
        ]

        preview = ", ".join(p.name for p in paths[:3]) + (", ..." if len(paths) > 3 else "")
        print(f"[ffmpeg concat+CFR] {len(paths)} frames ({preview}) → {out_video} @ {fps} fps")
        subprocess.run([c for c in cmd if c], check=True)
    finally:
        try:
            os.unlink(list_path)
        except Exception:
            pass


# ----------------------------- OpenCV composer -------------------------------

def compose_with_cv2(
    paths: List[Path],
    out_video: str,
    fps: float = 24.0,
    overwrite: bool = False
) -> None:
    import cv2  # lazy import

    outp = Path(out_video)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_video}. Use --overwrite to replace.")

    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {paths[0]}")
    H, W = first.shape[:2]

    suffix = outp.suffix.lower()
    if suffix in (".mp4", ".m4v", ".mov"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix in (".avi",):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif suffix in (".webm",):
        fourcc = cv2.VideoWriter_fourcc(*"VP90")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(outp), fourcc, float(fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_video}")

    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read frame: {p}")
        if im.shape[1] != W or im.shape[0] != H:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        writer.write(im)
    writer.release()

    preview = ", ".join(p.name for p in paths[:3])
    if len(paths) > 3:
        preview += ", ..."
    print(f"[opencv] {len(paths)} frames ({preview}) → {out_video} @ {fps} fps")


# ----------------------------- unified wrapper -------------------------------

def images_to_video(
    frames_dir: str,
    out_video: str,
    fps: float = 24.0,
    ext: str = "png",
    overwrite: bool = False,
    codec: str = "h264",
    crf: int = 18,
    preset: str = "medium",
    pix_fmt: str = "yuv420p"
) -> None:
    """
    If contiguous numeric sequence -> ffmpeg pattern (%0Nd).
    Else if ffmpeg available -> ffmpeg concat list (handles gaps/non-numeric).
    Else -> OpenCV.
    """
    paths, start_number, pad_width = _discover_sequence(frames_dir, ext)

    if have_ffmpeg():
        try:            
            # Non-contiguous or non-numeric: concat list
            compose_with_ffmpeg_concat(
                paths, out_video, fps, overwrite, codec, crf, preset, pix_fmt
            )
            return
        except Exception as e:
            print(f"[ffmpeg path] {e}. Falling back to OpenCV...")

    # Last resort
    compose_with_cv2(paths, out_video, fps, overwrite)


# ----------------------------------- CLI ------------------------------------

def main():
    p = argparse.ArgumentParser(description="Compose a video from a folder of frames.")
    p.add_argument("frames_dir", help="Directory with frames (e.g., 000001.png, 000002.png, ...)")
    p.add_argument("out_video", help="Output video file (.mp4, .webm, .avi, ...)")
    p.add_argument("--fps", type=float, default=24.0, help="Target frames per second (default: 24).")
    p.add_argument("--ext", choices=["png", "jpg", "jpeg", "webp"], default="png", help="Frame image extension (default: png).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    p.add_argument("--codec", choices=["h264", "vp9", "mpeg4"], default="h264", help="Video codec for ffmpeg path (default: h264/libx264).")
    p.add_argument("--crf", type=int, default=18, help="Constant Rate Factor for ffmpeg (lower = better).")
    p.add_argument("--preset", default="medium", help="ffmpeg x264 preset (ultrafast..placebo).")
    p.add_argument("--pix-fmt", default="yuv420p", help="ffmpeg pixel format (default: yuv420p).")
    args = p.parse_args()

    images_to_video(
        frames_dir=args.frames_dir,
        out_video=args.out_video,
        fps=args.fps,
        ext=args.ext,
        overwrite=args.overwrite,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        pix_fmt=args.pix_fmt
    )


if __name__ == "__main__":
    main()