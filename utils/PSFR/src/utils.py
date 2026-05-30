# standart imports
import sys
import math
import pathlib
import functools
import typing as tp
from dataclasses import dataclass

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR / "src"))

# cv & related imports
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# other imports
from constants import PAD_LEN


_INTERP = {
    "nearest": cv.INTER_NEAREST,
    "linear":  cv.INTER_LINEAR,
    "cubic":   cv.INTER_CUBIC,
    "area":    cv.INTER_AREA,
    "lanczos": cv.INTER_LANCZOS4,
}


# -------------------------- small helpers --------------------------

def with_default(value: tp.Any, default: tp.Any) -> tp.Any:
    return value if value is not None else default

def round_point(point: tp.Tuple[float, float]) -> tp.Tuple[int, int]:
    return (round(float(point[0])), round(float(point[1])))

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-z)))

def pad_integer(number: int, pad_length: int = PAD_LEN) -> str:
    s = str(int(number))
    return '0' * max(0, pad_length - len(s)) + s

def read_image_from_file(path: str) -> np.ndarray:
    return cv.imread(path, cv.IMREAD_COLOR)

def save_image_to_file(img: np.ndarray, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(path, img)

def make_grey(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        grey = img.copy()
    else:
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if grey.dtype != np.uint8:
        grey = cv.normalize(grey, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    return grey

def hex_to_rgb(hex_color: str) -> tp.Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (r, g, b)

def resize_image(
    img: np.ndarray,
    width: tp.Optional[int] = None,
    height: tp.Optional[int] = None,
    keep_aspect: bool = False,
    interpolation: str = "linear"
) -> np.ndarray:
    """
    Resize image deterministically BEFORE feature detection/tracking.
    If keep_aspect=True and both width/height are set -> scales by min ratio (no padding).
    """

    if width is None and height is None:
        return img

    H, W = img.shape[:2]
    inter = _INTERP.get(str(interpolation).lower(), cv.INTER_LINEAR)

    if keep_aspect:
        # determine target size via single scale (fit into WxH, no padding)
        sx = (width / W) if width else math.inf
        sy = (height / H) if height else math.inf
        s = min(sx, sy)

        if not math.isfinite(s):  # only one side provided
            s = sx if math.isfinite(sx) else sy
        
        newW = max(1, int(round(W * s)))
        newH = max(1, int(round(H * s)))
        return cv.resize(img, (newW, newH), interpolation=inter)
    else:
        # exact target; fallback to original for None sides
        targetW = int(width)  if width  is not None else W
        targetH = int(height) if height is not None else H
        return cv.resize(img, (targetW, targetH), interpolation=inter)


# -------------------------- drawing --------------------------

def add_circles(
    img: np.ndarray,
    points: np.ndarray,
    radius: int = 5,
    color_hex: str = "#FF0000",
    bgr: bool = False,
    thickness: int = -1,
    inplace: bool = False,
    **kwargs
) -> np.ndarray:
    if points is None or points.size == 0:
        return img if inplace else img.copy()

    out = img if inplace else img.copy()

    color = hex_to_rgb(color_hex)
    if bgr: color = color[::-1]

    for (x, y) in points.astype(np.float32):
        cv.circle(out, (int(x), int(y)), radius=radius, color=color, thickness=thickness, **kwargs)
    
    return out

def add_line(
    img: np.ndarray,
    start: tp.Tuple[int, int],
    end: tp.Tuple[int, int],
    color_hex: str = "#FF0000",
    alpha: float = 1.0,
    thickness: int = 2,
    inplace: bool = False,
    **kwargs
) -> np.ndarray:

    color_rgb = hex_to_rgb(color_hex)
    color_bgr = color_rgb[::-1]

    if inplace and alpha != 1.0:
        raise ValueError("inplace=True conflicts with alpha!=1.0")

    if inplace:
        cv.line(img, start, end, color=color_bgr, thickness=thickness, **kwargs)
        return img

    overlay = img.copy()
    cv.line(overlay, start, end, color=color_bgr, thickness=thickness, **kwargs)
    return cv.addWeighted(overlay, alpha, img, 1 - alpha, 0) if alpha != 1.0 else overlay

def add_rect(
    img: np.ndarray,
    rect: tp.Tuple[int, int, int, int],
    color_hex: str = "#FF0000",
    alpha: float = 0.3,
    thickness: int = -1,
    inplace: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Draw rectangle; if thickness == -1 or alpha != 1.0 -> filled with optional transparency.
    rect = (x, y, w, h)
    """

    x, y, w, h = rect
    color_bgr = hex_to_rgb(color_hex)[::-1]

    if thickness == -1 or alpha != 1.0:
        base = img if inplace else img.copy()
        overlay = base.copy()
        cv.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, thickness=-1, **kwargs)
        
        if alpha != 1.0:
            return cv.addWeighted(overlay, alpha, base, 1 - alpha, 0)
        return overlay
    else:
        out = img if inplace else img.copy()
        cv.rectangle(out, (x, y), (x + w, y + h), color_bgr, thickness=thickness, **kwargs)
        return out


# -------------------------- patching --------------------------

@dataclass(frozen=True)
class Patch:
    x: int
    y: int
    w: int
    h: int

@functools.lru_cache
def build_patches(W: int, H: int, nw: int, nh: int, centroidal: bool = True) -> tp.List[Patch]:
    pw, ph = max(1, W // nw), max(1, H // nh)
    base = [Patch(i * pw, j * ph, pw, ph) for j in range(nh) for i in range(nw)]

    if centroidal and (nw > 1 and nh > 1):
        for j in range(nh - 1):
            for i in range(nw - 1):
                x = int((i + 0.5) * pw); y = int((j + 0.5) * ph)
                x = max(0, min(x, W - pw)); y = max(0, min(y, H - ph))
                base.append(Patch(x, y, pw, ph))
    return base

def filter_patch_points(patch: Patch, points: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    if points is None or points.size == 0:
        return np.empty((0,), np.int32), np.empty((0,2), np.float32)

    px, py = points[:,0], points[:,1]
    mask = (
        (patch.x <= px) & (px <= patch.x + patch.w) &
        (patch.y <= py) & (py <= patch.y + patch.h)
    )
    idx = np.where(mask)[0]
    return idx.astype(np.int32), points[idx]

def separate_on_patches(points: np.ndarray, patches: tp.List[Patch]) -> tp.List[np.ndarray]:
    return [filter_patch_points(p, points)[1] for p in patches]

def remove_duplicate_points(points: np.ndarray, radius: float = 3.0) -> np.ndarray:
    if points is None or points.size == 0:
        return np.empty((0,2), np.float32)

    q = np.floor(points / max(1.0, float(radius))).astype(np.int32)
    seen, keep = set(), []

    for i, (qx, qy) in enumerate(q):
        key = (int(qx), int(qy))
        if key in seen: continue
        seen.add(key); keep.append(i)

    return points[np.array(keep, dtype=np.int32)]

def remove_duplicate_points_from_second_array(old_points: np.ndarray, new_points: np.ndarray, radius: float = 5.0) -> np.ndarray:
    if new_points is None or new_points.size == 0:
        return np.empty((0,2), np.float32)

    old_q = np.floor(old_points / max(1.0, float(radius))).astype(np.int32) if old_points.size else np.empty((0,2), np.int32)
    new_q = np.floor(new_points / max(1.0, float(radius))).astype(np.int32)
    seen = { (int(qx), int(qy)) for qx, qy in old_q }
    keep = []

    for i, (qx, qy) in enumerate(new_q):
        key = (int(qx), int(qy))
        if key in seen: continue
        seen.add(key); keep.append(i)

    return new_points[np.array(keep, dtype=np.int32)]


# -------------------------- display --------------------------

def show_numpy_image(img: np.ndarray) -> None:
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def show_numpy_images(images: tp.List[np.ndarray], titles: tp.Optional[tp.List[str]], cmap=None):
    n = len(images)
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (ax, im) in enumerate(zip(axes, images)):
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap or "gray")
        else:
            ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(titles[i] if titles else f"Image {i+1}")
    
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout(); plt.show()