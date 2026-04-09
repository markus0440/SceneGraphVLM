# standart imports
import sys
import json
import shutil
import pathlib
import typing as tp
from tqdm import tqdm

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR / "src"))

# cv & related imports
import numpy as np
import cv2 as cv

# other imports
from utils import (
    resize_image,
    make_grey,
    remove_duplicate_points_from_second_array,
    build_patches,
    filter_patch_points,

    # visualization
    add_circles,
    add_rect,
    save_image_to_file,
)
from shi_tomasi import shi_tomasi_with_patching_and_sorting, custom_sorting_by_quality
from lucas_kanade import lucas_kanade_track


def select_keyframes_from_frames(
    frames_dir: pathlib.Path,
    out_dir: pathlib.Path,
    key_frames_dir: pathlib.Path,
    patching: dict,
    shi: dict,
    lk: dict,
    selection: dict,
    preprocess: tp.Optional[dict] = None,
    visualization: tp.Optional[dict] = None,
    frame_paths: tp.Optional[tp.List[pathlib.Path]] = None,
) -> dict:
    """
    Select keyframes from a directory (frames_dir) or from an explicit list of paths (frame_paths).
    When frame_paths is given, frames are read from those paths and no report is written / no keyframes copied.
    """
    use_paths = frame_paths is not None
    if use_paths:
        frames = sorted([pathlib.Path(p) for p in frame_paths], key=lambda p: p.name)
    else:
        frames = sorted([p for p in pathlib.Path(frames_dir).glob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
        out_dir.mkdir(parents=True, exist_ok=True)
        if selection.get("copy_keyframes", True):
            key_frames_dir.mkdir(parents=True, exist_ok=True)

    # preprocess
    preprocess = preprocess or {}
    rz = (preprocess.get("resize") or {})
    rz_w = rz.get("width")
    rz_h = rz.get("height")
    keep_aspect = bool(preprocess.get("keep_aspect", False))
    interp = str(preprocess.get("interpolation", "linear"))

    def _pre(img):
        return resize_image(img, width=rz_w, height=rz_h, keep_aspect=keep_aspect, interpolation=interp)
    
    # visualization (disabled when using frame_paths to avoid writing per-video viz)
    visualization = visualization or {}
    viz_enabled = bool(visualization.get("enabled", False)) and not use_paths
    viz_dir = pathlib.Path(visualization.get("dir", "data/output/viz"))
    tracked_color = str(visualization.get("tracked_color", "#00FF00"))
    new_color = str(visualization.get("new_color", "#0066FF"))
    drop_color = str(visualization.get("patch_drop_color", "#FF0000"))
    drop_alpha = float(visualization.get("patch_drop_alpha", 0.3))
    pt_radius = int(visualization.get("point_radius", 4))
    pt_thick  = int(visualization.get("point_thickness", -1))

    if viz_enabled:
        viz_dir.mkdir(parents=True, exist_ok=True)

    def save_viz(
        img_color: np.ndarray,
        tracked_xy: np.ndarray,
        new_xy: np.ndarray,
        dropping_patch_ids: tp.Optional[tp.List[int]],
        frame_name: str
    ) -> None:
        if not viz_enabled: return

        vis = img_color.copy()

        # highlight dropping patches
        if dropping_patch_ids:
            for pid in dropping_patch_ids:
                p = patches[pid]
                vis = add_rect(vis, (p.x, p.y, p.w, p.h), color_hex=drop_color, alpha=drop_alpha, thickness=-1)

        # draw points: tracked = green; new = blue
        if tracked_xy is not None and tracked_xy.size:
            vis = add_circles(vis, tracked_xy, radius=pt_radius, color_hex=tracked_color, bgr=True, thickness=pt_thick)

        if new_xy is not None and new_xy.size:
            vis = add_circles(vis, new_xy, radius=pt_radius, color_hex=new_color, bgr=True, thickness=pt_thick)

        save_image_to_file(vis, str(viz_dir / frame_name))
    
    if not frames:
        if use_paths:
            return {"keyframes": [], "num_keyframes": 0, "config": {}, "events": []}
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # static params
    nw, nh = int(patching["nw"]), int(patching["nh"])
    centroidal = bool(patching.get("centroidal", True))
    patches = build_patches(1, 1, nw, nh, centroidal)  # dummy init; real dims per first image
    CORNERS_LIMIT = int(shi["CORNERS_LIMIT_PER_IMAGE"])

    # read first frame
    first = cv.imread(str(frames[0]), cv.IMREAD_COLOR)
    H, W = first.shape[:2]
    first = _pre(first)
    patches = build_patches(W, H, nw, nh, centroidal)

    grey0 = make_grey(first)
    base_pts = shi_tomasi_with_patching_and_sorting(
        grey0,
        nw=nw, nh=nh, centroidal=centroidal, need_grey=False,
        max_corners_patch=int(shi["max_corners_patch"]),
        maxCorners=CORNERS_LIMIT,
        dedup_radius=float(shi["dedup_radius"]),
        qualityLevel=float(shi["qualityLevel"]),
        minDistance=float(shi["minDistance"]),
        useHarrisDetector=bool(shi["useHarrisDetector"]),
        blockSize=int(shi["blockSize"]),
        gradientSize=int(shi["gradientSize"]),
        k=float(shi["k"]),
        custom_sort=custom_sorting_by_quality
    )

    # frame 0 visualization: no drop overlay, all corners are "new"
    save_viz(first, tracked_xy=np.empty((0, 2), np.float32), new_xy=base_pts, dropping_patch_ids=None, frame_name=frames[0].name)

    # per-patch initial counts (at "reseed" start)
    def counts_per_patch(points_xy: np.ndarray) -> np.ndarray:
        counts = np.zeros(len(patches), dtype=np.int32)

        for pid, p in enumerate(patches):
            idx, _ = filter_patch_points(p, points_xy)
            counts[pid] = int(idx.size)

        return counts

    init_counts = counts_per_patch(base_pts)
    keyframes = [frames[0].name]  # start-of-segment is often a key anchor
    events = []

    # LK params
    winSize = tuple(int(x) for x in lk["winSize"])
    criteria_flag, criteria_count, criteria_eps = lk["criteria"]
    lk_kwargs = dict(
        winSize=winSize,
        maxLevel=int(lk["maxLevel"]),
        criteria=(int(criteria_flag), int(criteria_count), float(criteria_eps))
    )

    tau = float(selection["retention_tau"])
    min_k = int(selection["min_patches_k"])

    for i in tqdm(range(1, len(frames))):
        prev = _pre(cv.imread(str(frames[i - 1]), cv.IMREAD_COLOR))
        curr = _pre(cv.imread(str(frames[i]), cv.IMREAD_COLOR))
        prevg, currg = make_grey(prev), make_grey(curr)

        tracked_idx, tracked_xy = lucas_kanade_track(
            prevg, currg, base_pts,
            nw=nw, nh=nh, centroidal=centroidal,
            max_error=float(lk["max_error"]),
            **lk_kwargs
        )

        # retention per patch = (#tracked from base) / (#initial at reseed)
        tracked_mask = np.zeros(base_pts.shape[0], dtype=np.uint8)
        tracked_mask[tracked_idx] = 1

        per_patch_tracked = np.zeros(len(patches), dtype=np.int32)
        for pid, p in enumerate(patches):
            idx, _ = filter_patch_points(p, base_pts)  # which base indices belong to this patch
            if idx.size:
                per_patch_tracked[pid] = int(tracked_mask[idx].sum())

        # avoid div-by-zero: only consider patches with init>0
        low_retention_patches = 0
        ratios = []
        dropping = []
        for pid in range(len(patches)):
            if init_counts[pid] > 0:
                r = per_patch_tracked[pid] / float(init_counts[pid])
                ratios.append((pid, r))

                if r < tau:
                    low_retention_patches += 1
                    dropping.append(pid)

        mark_key = (low_retention_patches >= min_k)

        if mark_key:
            keyframes.append(frames[i].name)

            # reseed: keep currently tracked + refill to CORNERS_LIMIT
            refill = max(0, CORNERS_LIMIT - int(tracked_xy.shape[0]))
            new_pts = shi_tomasi_with_patching_and_sorting(
                currg, nw=nw, nh=nh, centroidal=centroidal, need_grey=False,
                max_corners_patch=int(shi["max_corners_patch"]),
                maxCorners=refill,
                dedup_radius=float(shi["dedup_radius"]),
                qualityLevel=float(shi["qualityLevel"]),
                minDistance=float(shi["minDistance"]),
                useHarrisDetector=bool(shi["useHarrisDetector"]),
                blockSize=int(shi["blockSize"]),
                gradientSize=int(shi["gradientSize"]),
                k=float(shi["k"]),
                custom_sort=custom_sorting_by_quality
            )

            # visualization
            save_viz(curr, tracked_xy=tracked_xy, new_xy=new_pts, dropping_patch_ids=dropping, frame_name=frames[i].name)

            new_pts = remove_duplicate_points_from_second_array(tracked_xy, new_pts, radius=float(shi["dedup_radius"]))
            base_pts = np.vstack([tracked_xy, new_pts]) if new_pts.size else tracked_xy
            init_counts = counts_per_patch(base_pts)  # reset denominators

            events.append({
                "frame": frames[i].name,
                "low_retention_patches": int(low_retention_patches),
                "ratios": [float(r) for _pid, r in ratios]
            })
        else:
            # carry forward without reseed; next step uses current base_pts = tracked_xy + refill (NOT reseeding here)
            refill = max(0, CORNERS_LIMIT - int(tracked_xy.shape[0]))
            new_pts = shi_tomasi_with_patching_and_sorting(
                currg, nw=nw, nh=nh, centroidal=centroidal, need_grey=False,
                max_corners_patch=int(shi["max_corners_patch"]),
                maxCorners=refill,
                dedup_radius=float(shi["dedup_radius"]),
                qualityLevel=float(shi["qualityLevel"]),
                minDistance=float(shi["minDistance"]),
                useHarrisDetector=bool(shi["useHarrisDetector"]),
                blockSize=int(shi["blockSize"]),
                gradientSize=int(shi["gradientSize"]),
                k=float(shi["k"]),
                custom_sort=custom_sorting_by_quality
            )

            # visualization
            save_viz(curr, tracked_xy=tracked_xy, new_xy=new_pts, dropping_patch_ids=dropping, frame_name=frames[i].name)
            
            new_pts = remove_duplicate_points_from_second_array(tracked_xy, new_pts, radius=float(shi["dedup_radius"]))
            base_pts = np.vstack([tracked_xy, new_pts]) if new_pts.size else tracked_xy
            # keep init_counts unchanged until we actually mark a keyframe

    # Save artifacts (skip when using frame_paths)
    report = {
        "keyframes": keyframes,
        "num_keyframes": len(keyframes),
        "config": {
            "nw": nw, "nh": nh, "tau": tau, "min_patches_k": min_k,
            "CORNERS_LIMIT": CORNERS_LIMIT
        },
        "events": events
    }
    if not use_paths:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "keyframes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if selection.get("copy_keyframes", True):
            for name in keyframes:
                src = frames_dir / name
                dst = key_frames_dir / name
                print(src, dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
        (out_dir / "keyframes.txt").write_text("\n".join(keyframes) + "\n")
    return report