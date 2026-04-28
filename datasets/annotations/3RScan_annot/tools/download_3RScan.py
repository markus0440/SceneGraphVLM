#!/usr/bin/env python
"""
Download 3RScan ``sequence.zip`` per scan and extract RGB-D frames.

This is a slimmed-down variant of the original 3RScan downloader. It only fetches
``sequence.zip`` (color/depth/poses) and skips scene-level mesh / segmentation
annotation files: per-frame GT scene graphs are already provided in
``datasets/annotations/3RScan_annot/annotations/{train,test}_annotations.json``.

By default the script downloads only scans listed in the resplit text files
shipped with the annotations (``train_resplit_scans.txt`` +
``test_resplit_scans.txt``). Pass ``--scan_list`` to override, or ``--id`` to
download a single scan.

**Run** (from the SceneGraphVLM repository root):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/3RScan_annot/tools/download_3RScan.py

Layout produced::

  datasets/frames/3RScan_frames/<scan_id>/
      frame-000000.color.jpg
      frame-000000.depth.pgm
      frame-000000.pose.txt
      ...
      _info.txt
      .sequence_extracted   # marker file, signals "do not redownload"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from http.client import IncompleteRead
from pathlib import Path
from typing import Iterable, List

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        class _DummyPbar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def update(self, n=0):
                return None

            def set_postfix_str(self, *args, **kwargs):
                return None

        if iterable is None:
            return _DummyPbar()
        return iterable

    setattr(tqdm, "write", print)


BASE_URL = "http://campar.in.tum.de/public_datasets/3RScan/"
DATA_URL = BASE_URL + "Dataset/"
TOS_URL = "http://campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf"
FRAMES_ZIP = "sequence.zip"

ID_REGEX = re.compile(r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}")
EXTRACT_MARKER = ".sequence_extracted"

_TOOLS_DIR = Path(__file__).resolve().parent
_ANNOT_DIR_DEFAULT = _TOOLS_DIR.parent / "annotations"
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]

DEFAULT_FRAMES_DIR = str(_REPO_ROOT_DEFAULT / "datasets" / "frames" / "3RScan_frames")
DEFAULT_TRAIN_LIST = str(_ANNOT_DIR_DEFAULT / "train_resplit_scans.txt")
DEFAULT_TEST_LIST = str(_ANNOT_DIR_DEFAULT / "test_resplit_scans.txt")


def read_scan_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Scan list not found: {path}")
    scans: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            match = ID_REGEX.search(line)
            if not match:
                continue
            scan_id = match.group()
            if scan_id not in seen:
                seen.add(scan_id)
                scans.append(scan_id)
    return scans


def is_frames_extracted(scan_id: str, frames_root: Path) -> bool:
    marker = frames_root / scan_id / EXTRACT_MARKER
    return marker.is_file()


def download_file_robust(
    url: str,
    out_file: Path,
    desc: str = "",
    max_retries: int = 5,
    backoff: int = 2,
    chunk_size: int = 8192,
    timeout: int = 30,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_file.with_suffix(out_file.suffix + ".part")

    attempt = 0
    while attempt < max_retries:
        try:
            existing = tmp_path.stat().st_size if tmp_path.exists() else 0
            req = urllib.request.Request(url)
            if existing > 0:
                req.add_header("Range", f"bytes={existing}-")

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_length = resp.getheader("Content-Length")
                total = None
                if content_length is not None:
                    total = int(content_length) + (existing if existing > 0 else 0)

                mode = "ab" if existing > 0 else "wb"
                with open(tmp_path, mode) as fh:
                    with tqdm(
                        total=total,
                        initial=existing,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=desc or out_file.name,
                        leave=False,
                        dynamic_ncols=True,
                    ) as pbar:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            fh.write(chunk)
                            pbar.update(len(chunk))

            if content_length is not None:
                expected = int(content_length) + (existing if existing > 0 else 0)
                actual = tmp_path.stat().st_size
                if actual != expected:
                    raise IOError(
                        f"Size mismatch for {out_file.name}: got {actual}, expected {expected}"
                    )

            os.replace(tmp_path, out_file)
            return

        except (urllib.error.ContentTooShortError, IncompleteRead, IOError, urllib.error.URLError) as exc:
            attempt += 1
            wait = backoff ** attempt
            tqdm.write(f"[retry {attempt}/{max_retries}] {out_file.name}: {exc}; waiting {wait}s")
            time.sleep(wait)
            continue

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def download_and_extract_frames(
    scan_id: str, frames_root: Path, keep_sequence_zip: bool = False
) -> None:
    scene_dir = frames_root / scan_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    if is_frames_extracted(scan_id, frames_root):
        return

    zip_path = scene_dir / FRAMES_ZIP
    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        url = DATA_URL + scan_id + "/" + FRAMES_ZIP
        download_file_robust(url, zip_path, desc=f"{scan_id[:8]}:{FRAMES_ZIP}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(scene_dir)

    marker = scene_dir / EXTRACT_MARKER
    marker.write_text("ok\n", encoding="utf-8")

    if not keep_sequence_zip:
        try:
            zip_path.unlink()
        except OSError:
            pass


def collect_scans(args: argparse.Namespace) -> List[str]:
    if args.id:
        return [args.id.strip()]

    if args.scan_list:
        custom_list = read_scan_list(Path(args.scan_list))
        return custom_list

    train_list = Path(args.train_list)
    test_list = Path(args.test_list)
    train_scans = read_scan_list(train_list)
    test_scans = read_scan_list(test_list)
    seen = set()
    merged: List[str] = []
    for scan_id in train_scans + test_scans:
        if scan_id in seen:
            continue
        seen.add(scan_id)
        merged.append(scan_id)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download 3RScan sequence.zip per scan (frames only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frames_dir",
        default=DEFAULT_FRAMES_DIR,
        help="Output root for extracted RGB-D sequences (one folder per scan_id).",
    )
    parser.add_argument(
        "--train_list",
        default=DEFAULT_TRAIN_LIST,
        help="Path to text file with train scan ids (resplit).",
    )
    parser.add_argument(
        "--test_list",
        default=DEFAULT_TEST_LIST,
        help="Path to text file with test scan ids (resplit).",
    )
    parser.add_argument(
        "--scan_list",
        default="",
        help="Optional: path to a custom scan list (overrides --train_list/--test_list).",
    )
    parser.add_argument(
        "--id",
        default="",
        help="Optional: download a single scan id (overrides all lists).",
    )
    parser.add_argument(
        "--keep_sequence_zip",
        action="store_true",
        help="Keep sequence.zip on disk after extraction (off by default).",
    )
    args = parser.parse_args()

    print("You confirm that you agreed to the 3RScan terms of use:")
    print(TOS_URL)
    print("***")

    scans = collect_scans(args)
    if not scans:
        print("No scans selected, nothing to do.", file=sys.stderr)
        return 1

    frames_root = Path(args.frames_dir).resolve()
    frames_root.mkdir(parents=True, exist_ok=True)

    print(f"Frames root: {frames_root}")
    print(f"Scans queued: {len(scans)}")

    with tqdm(scans, desc="Scans", unit="scan", dynamic_ncols=True) as bar:
        for scan_id in bar:
            bar.set_postfix_str(scan_id[:8], refresh=False)
            try:
                download_and_extract_frames(
                    scan_id=scan_id,
                    frames_root=frames_root,
                    keep_sequence_zip=args.keep_sequence_zip,
                )
            except Exception as exc:
                tqdm.write(f"[error] {scan_id}: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
