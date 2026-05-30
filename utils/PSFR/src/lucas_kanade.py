# standart imports
import sys
import pathlib
import typing as tp

# cv & related imports
import numpy as np
import cv2 as cv

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR / "src"))

# other imports
from utils import build_patches, filter_patch_points


def lucas_kanade_track(
    old_img_gray: np.ndarray,
    new_img_gray: np.ndarray,
    old_points_xy: np.ndarray,
    nw: int = 5, nh: int = 5,
    centroidal: bool = True,
    max_error: float = 15.0,
    **lk_kwargs
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (tracked_indices_in_old_points, new_positions_xy_for_those_indices)
    """
    if old_points_xy is None or old_points_xy.size == 0:
        return np.empty((0,), np.int32), np.empty((0, 2), np.float32)

    if old_points_xy.dtype != np.float32:
        old_points_xy = old_points_xy.astype(np.float32)

    H, W = old_img_gray.shape[:2]
    patches = build_patches(W, H, nw, nh, centroidal)

    tracked_flag = np.zeros(old_points_xy.shape[0], dtype=np.uint8)
    new_positions = np.zeros_like(old_points_xy, dtype=np.float32)

    for p in patches:
        old_roi = old_img_gray[p.y: p.y + p.h, p.x: p.x + p.w]
        new_roi = new_img_gray[p.y: p.y + p.h, p.x: p.x + p.w]
        shift = np.array([p.x, p.y], dtype=np.float32)

        idx_local, pts_local = filter_patch_points(p, old_points_xy)
        if idx_local.size == 0:
            continue

        pts_local_w = (pts_local - shift).reshape(-1, 1, 2)
        new_pts, status, err = cv.calcOpticalFlowPyrLK(old_roi, new_roi, pts_local_w, None, **lk_kwargs)
        if new_pts is None or status is None:
            continue

        status = status.reshape(-1).astype(bool)
        err = err.reshape(-1) if err is not None else np.full(status.shape, np.inf, dtype=np.float32)
        good = np.where(status & (err <= max_error))[0]

        for li in good:
            gi = idx_local[li]
            if tracked_flag[gi] == 0:
                new_positions[gi] = new_pts[li].reshape(2) + shift
                tracked_flag[gi] = 1

    tracked_idx = np.where(tracked_flag == 1)[0].astype(np.int32)
    return tracked_idx, new_positions[tracked_idx]