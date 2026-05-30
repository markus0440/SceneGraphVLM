# standart imports
import sys
import pathlib
import typing as tp

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR / "src"))

# cv & related imports
import numpy as np
import cv2 as cv

# other imports
from utils import build_patches, remove_duplicate_points, sigmoid


# --------- local helpers (kept here to avoid cross-import cycles) ---------

def _bilinear_sample(map2d: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = map2d.shape[:2]

    x = np.clip(points[:, 0], 0, w - 1 - 1e-6)
    y = np.clip(points[:, 1], 0, h - 1 - 1e-6)

    x0 = np.floor(x).astype(np.int32); x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(np.int32); y1 = np.clip(y0 + 1, 0, h - 1)

    dx = x - x0; dy = y - y0

    Ia = map2d[y0, x0]
    Ib = map2d[y0, x1]
    Ic = map2d[y1, x0]
    Id = map2d[y1, x1]

    return (Ia * (1 - dx) * (1 - dy) + Ib * dx * (1 - dy) + Ic * (1 - dx) * dy + Id * dx * dy)

def _corner_response_map(gray: np.ndarray, blockSize: int, gradientSize: int, use_harris: bool, k: float) -> np.ndarray:
    gradientSize = int(max(3, gradientSize) | 1)  # odd
    blockSize = int(max(2, blockSize))

    if use_harris:
        return cv.cornerHarris(gray, blockSize=blockSize, ksize=gradientSize, k=float(k))
    
    return cv.cornerMinEigenVal(gray, blockSize=blockSize, ksize=gradientSize)

def round_robin_interleave_scored(lists_scored: tp.List[np.ndarray], max_total: tp.Optional[int] = None) -> np.ndarray:
    out, rank, nlists = [], 0, len(lists_scored)

    while True:
        added = False

        for li in lists_scored:
            if rank < len(li):
                out.append(li[rank]); added = True

                if max_total is not None and len(out) >= max_total:
                    return np.vstack(out).astype(np.float32)

        if not added: break

        rank += 1

    return np.vstack(out).astype(np.float32) if out else np.empty((0, 3), np.float32)


# ------------------------------ API ------------------------------

def shi_tomasi(img_gray_or_bgr: np.ndarray, need_grey: bool = True, **kwargs) -> np.ndarray:
    gray = cv.cvtColor(img_gray_or_bgr, cv.COLOR_BGR2GRAY) if (need_grey and img_gray_or_bgr.ndim == 3) else img_gray_or_bgr
    pts = cv.goodFeaturesToTrack(gray, **kwargs)
    if pts is None: return np.empty((0, 2), np.float32)
    return pts.reshape(-1, 2).astype(np.float32)

def shi_tomasi_with_patching(
    img_gray_or_bgr: np.ndarray,
    nw: int = 5, nh: int = 5,
    centroidal: bool=True,
    need_grey: bool=True,
    max_corners_patch: int = 10,
    maxCorners: int = 25,
    dedup_radius: float = 5.0,
    **kwargs
) -> np.ndarray:
    gray = cv.cvtColor(img_gray_or_bgr, cv.COLOR_BGR2GRAY) if (need_grey and img_gray_or_bgr.ndim == 3) else img_gray_or_bgr
    H, W = gray.shape[:2]
    patches = build_patches(W, H, nw, nh, centroidal=centroidal)
    gathered = []

    for p in patches:
        mask = np.zeros((H, W), np.uint8)
        mask[p.y: p.y + p.h, p.x: p.x + p.w] = 255
        cand = cv.goodFeaturesToTrack(gray, mask=mask, maxCorners=max_corners_patch, **kwargs)

        if cand is not None:
            gathered.append(cand.reshape(-1, 2).astype(np.float32))

    if not gathered:
        return np.empty((0, 2), np.float32)

    all_pts = np.vstack(gathered)
    return remove_duplicate_points(all_pts, radius=dedup_radius).astype(np.float32)[:maxCorners]

def shi_tomasi_with_patching_and_sorting(
    img_gray_or_bgr: np.ndarray,
    nw: int = 5, nh: int = 5,
    centroidal: bool = True,
    need_grey: bool = True,
    max_corners_patch: int = 10,
    maxCorners: int = 25,
    dedup_radius: float = 5.0,
    custom_sort: tp.Callable = round_robin_interleave_scored,
    **kwargs
) -> np.ndarray:
    gray = cv.cvtColor(img_gray_or_bgr, cv.COLOR_BGR2GRAY) if (need_grey and img_gray_or_bgr.ndim == 3) else img_gray_or_bgr
    H, W = gray.shape[:2]

    patches = build_patches(W, H, nw, nh, centroidal=centroidal)
    per_patch_pts: tp.List[np.ndarray] = []

    for p in patches:
        mask = np.zeros((H, W), np.uint8)
        mask[p.y: p.y + p.h, p.x: p.x + p.w] = 255
        cand = cv.goodFeaturesToTrack(gray, mask=mask, maxCorners=max_corners_patch, **kwargs)
        per_patch_pts.append(np.empty((0, 2), np.float32) if cand is None else cand.reshape(-1, 2).astype(np.float32))

    if all(len(c) == 0 for c in per_patch_pts):
        return np.empty((0, 2), np.float32)

    response_map = _corner_response_map(
        gray,
        blockSize=int(kwargs.get("blockSize", 3)),
        gradientSize=int(kwargs.get("gradientSize", 3)),
        use_harris=bool(kwargs.get("useHarrisDetector", False)),
        k=float(kwargs.get("k", 0.04))
    )

    lists_scored: tp.List[np.ndarray] = []
    for cand in per_patch_pts:
        if len(cand) == 0:
            lists_scored.append(np.empty((0, 3), np.float32))
            continue

        scores = _bilinear_sample(response_map, cand)
        order = np.argsort(-scores)
        lists_scored.append(np.column_stack([scores[order], cand[order]]).astype(np.float32))

    merged = custom_sort(lists_scored)
    xy = merged[:, 1:3] if merged.size else np.empty((0, 2), np.float32)
    xy = remove_duplicate_points(xy, radius=dedup_radius)
    return xy.astype(np.float32)[:maxCorners]

def plain_sorting_by_quality(lists_scored: tp.List[np.ndarray], stable: bool=False) -> np.ndarray:
    arr = np.vstack(lists_scored).astype(np.float32, copy=False)
    key = arr[:, 0]
    order = np.argsort(-key, kind=("mergesort" if stable else "quicksort"))
    return arr[order]

def custom_sorting_by_quality(lists_scored: tp.List[np.ndarray], temperature: float = 0.2, stable: bool=False) -> np.ndarray:
    keys = []

    for _list in lists_scored:
        for i, (score, _x, _y) in enumerate(_list):
            keys.append(score - temperature * i * (1 - sigmoid(score - 0.3)))

    arr = np.vstack(lists_scored).astype(np.float32, copy=False)
    order = np.argsort(-np.array(keys), kind=("mergesort" if stable else "quicksort"))
    return arr[order]