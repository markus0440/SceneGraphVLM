import shutil
import argparse
import subprocess
import typing as tp
from pathlib import Path


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_with_ffmpeg(
    video: str,
    out_dir: str,
    max_fps: tp.Optional[float] = 4.0,
    ext: str = "png",
    overwrite: bool = False
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = (out / f"%06d.{ext}").as_posix()

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video]

    # capping FPS without upsampling: select frames spaced by >= 1/max_fps seconds.
    if max_fps and max_fps > 0:
        # passing the filter string WITHOUT shell quotes when using subprocess list
        vf_expr = f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,1.0/{max_fps})"
        cmd += ["-vf", f"select={vf_expr}", "-vsync", "vfr"]
    else:
        # no cap -> just dumping all frames (variable-rate sync to avoid duplicates)
        cmd += ["-vsync", "vfr"]

    # starting numbering at 1, zero-padded to 6 digits via pattern
    cmd += ["-start_number", "1"]
    if overwrite:
        cmd.insert(1, "-y")

    cmd += [pattern]

    subprocess.run(cmd, check=True)


def extract_with_cv2(
    video: str,
    out_dir: str,
    max_fps: tp.Optional[float] = 4.0,
    ext: str = "png",
    overwrite: bool = False
) -> None:
    """
    Slower fallback using OpenCV, only if ffmpeg isn't available.
    """

    import cv2  # lazy import

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}.")

    step = None if (not max_fps or max_fps <= 0) else 1.0 / max_fps
    next_t = -1e9 if step else None  # forcing first frame save if capping

    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # preferring timestamp in seconds when available
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = (t_ms / 1000.0) if t_ms and t_ms > 0 else None

        save = False
        if step is None:
            save = True
        else:
            if t is not None:
                if t >= next_t:
                    save = True
                    next_t = t + step
            else:
                # fallback: approximate by frame index and native fps
                fps = cap.get(cv2.CAP_PROP_FPS)

                if not fps or fps <= 0:
                    fps = 30.0  # guessing
                
                approx_t = (idx - 1) / fps
                if approx_t >= next_t:
                    save = True
                    next_t = approx_t + step

        if save:
            out_path = out / f"{idx:06d}.{ext}"

            if overwrite or not out_path.exists():
                cv2.imwrite(str(out_path), frame)
            
            idx += 1

    cap.release()


def extract_frames(
    video: str,
    out_dir: str,
    max_fps: float = 4.0,
    ext: str = "png",
    overwrite: bool = False
) -> None:
    """
    Convert `video` into a folder of frames named 000001.png, 000002.png, ...
    Caps output FPS to `max_fps` (no upsampling). Set (max_fps <= 0) to disable capping.
    """

    if have_ffmpeg():
        extract_with_ffmpeg(video, out_dir, max_fps, ext, overwrite)
    else:
        print("ffmpeg not found. Falling back to OpenCV...")
        extract_with_cv2(video, out_dir, max_fps, ext, overwrite)


def main():
    p = argparse.ArgumentParser(description="Extract video frames with an FPS cap (no upsampling).")
    p.add_argument("video", help="Path to input video (.mp4, .mov, ...)")
    p.add_argument("out_dir", help="Output directory for frames")
    p.add_argument("--max-fps", type=float, default=4.0, help="Maximum frames per second to export (default: 4). Use <= 0 to disable capping.")
    p.add_argument("--ext", choices=["png", "jpg", "webp"], default="png", help="Image format (default: png).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite frames if present.")
    args = p.parse_args()

    extract_frames(args.video, args.out_dir, args.max_fps, args.ext, args.overwrite)


if __name__ == "__main__":
    main()