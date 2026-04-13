#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVSG (Panoptic Video Scene Graph) ==> TOON SFT JSON + resized frames (640x480).

Reads ``pvsg.json`` (video-level objects/relations + split lists) and, per frame,
loads RGB frames and panoptic masks from the PVSG data tree, derives boxes from
masks, and writes TOON scene graphs.

**Layout (defaults are repo-relative):**

- Annotations: ``datasets/annotations/PVSG_annot/annotations/pvsg.json``
- Source frames/masks: ``datasets/annotations/PVSG_annot/OpenPVSG/{vidor,epic_kitchen,ego4d}/``
  with ``{source}/frames/{video_id}/*.png`` and ``{source}/masks/...``
- Output JSON: ``datasets/annotations/PVSG_annot/data_sft_original/{train,test}_annotations_toon_sft.json``
  (PVSG ``split["val"]`` in ``pvsg.json`` is written as ``test_*.json``.)
- Output images: ``datasets/frames/PVSG_frames/train_images/`` and ``test_images/`` (same mapping).

Each sample ``image_path`` is **relative to the SceneGraphVLM repository root** (POSIX ``/``).

**Run** (repository root = SceneGraphVLM):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/PVSG_annot/tools/prepare_original_pvsg_sft.py

After layout or naming changes, re-run the same command to regenerate
``train_annotations_toon_sft.json``, ``test_annotations_toon_sft.json``, and frames under
``PVSG_frames/`` so ``image_path`` entries match on-disk paths.

Optional: ``--sources``, ``--splits``, ``--limit_videos``, ``--limit_frames``, ``--only_video``,
``--num_workers``, ``--repo_root``, ``--pvsg_data_root``, ``--annotation``, ``--export_root``,
``--images_out`` (all paths may be repo-relative or absolute).
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

# <repo>/datasets/annotations/PVSG_annot/tools/<this>.py
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]

ANNOTATION_REL = "datasets/annotations/PVSG_annot/annotations/pvsg.json"
PVSG_DATA_ROOT_REL = "datasets/annotations/PVSG_annot/OpenPVSG"
EXPORT_ROOT_REL = "datasets/annotations/PVSG_annot/data_sft_original"
IMAGES_OUT_REL = "datasets/frames/PVSG_frames"

SUPPORTED_SOURCES = ["vidor", "epic_kitchen", "ego4d"]

TARGET_WIDTH = 640
TARGET_HEIGHT = 480


def resolve_under_repo(repo_root: Path, path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / path_arg).resolve()


def path_for_json(repo_root: Path, file_path: Path) -> str:
    try:
        return str(file_path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(file_path.resolve()).replace("\\", "/")


def split_images_subdir(split_name: str) -> str:
    """train ==> train_images; PVSG val split ==> test_images."""
    if split_name == "train":
        return "train_images"
    if split_name == "val":
        return "test_images"
    return f"{split_name}_images"


def split_annotation_file_stem(split_name: str) -> str:
    """Output JSON basename prefix: train ==> train; PVSG val split ==> test."""
    if split_name == "val":
        return "test"
    return split_name


# ==================== DATA STRUCTURES AND TYPE ALIASES ====================

VideoID = str
FrameID = str
ObjectID = int
BoundingBox = List[int]  # [x_min, y_min, x_max, y_max]
TimeInterval = Tuple[int, int]  # [start_frame, end_frame]
RelationshipTuple = Tuple[ObjectID, ObjectID, str, List[TimeInterval]]

# ==================== HELPER FUNCTIONS ====================


def determine_data_source(video_id: VideoID) -> str:
    """
    Identify the dataset source from video ID patterns.
    
    Args:
        video_id: Unique video identifier
        
    Returns:
        Dataset source name ('vidor', 'epic_kitchen', or 'ego4d')
    """
    if video_id.startswith("P"):
        return "epic_kitchen"
    if "-" in video_id:
        return "ego4d"
    return "vidor"


def safe_integer_conversion(value: Any) -> Optional[int]:
    """
    Safely convert input to integer.
    
    Args:
        value: Input value to convert
        
    Returns:
        Integer value or None if conversion fails
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def normalize_relationship_predicate(predicate: str) -> str:
    """
    Normalize relationship predicates to hyphen-separated format.
    
    Args:
        predicate: Original predicate string (e.g., "in front of")
        
    Returns:
        Normalized predicate (e.g., "in-front-of")
    """
    predicate = (predicate or "").strip()
    predicate = " ".join(predicate.split())  # Remove extra whitespace
    return predicate.replace(" ", "-")


def decode_panoptic_rgb_mask(mask_array: np.ndarray) -> np.ndarray:
    """
    Decode COCO panoptic mask encoding: id = R + G * 256 + B * 256²
    
    Args:
        mask_array: RGB mask array of shape (H, W, 3) with uint8 dtype
        
    Returns:
        Integer ID mask of shape (H, W) with int64 dtype
    """
    mask_array = mask_array.astype(np.uint32)
    return (mask_array[..., 0] + 
            (mask_array[..., 1] << 8) + 
            (mask_array[..., 2] << 16)).astype(np.int64)


def load_segmentation_mask(mask_path: Path) -> Optional[np.ndarray]:
    """
    Load segmentation mask, supporting both ID maps and RGB panoptic masks.
    
    Args:
        mask_path: Path to mask image file
        
    Returns:
        Integer mask array or None if loading fails
    """
    try:
        mask_data = np.array(Image.open(mask_path))
        if mask_data.ndim == 3 and mask_data.shape[-1] == 3:
            return decode_panoptic_rgb_mask(mask_data)
        return mask_data.astype(np.int64)
    except Exception as error:
        return None


def extract_bounding_boxes_from_mask(mask: np.ndarray) -> List[Tuple[ObjectID, BoundingBox]]:
    """
    Extract bounding boxes for all object instances in a segmentation mask.
    
    Args:
        mask: Integer mask array where values > 0 represent object IDs
        
    Returns:
        List of (object_id, bounding_box) tuples
    """
    if mask is None:
        return []
    
    unique_ids = np.unique(mask)
    object_ids = unique_ids[unique_ids > 0]  # Exclude background (0)
    
    bounding_boxes = []
    for instance_id in object_ids.tolist():
        # Find coordinates of all pixels belonging to this instance
        y_coords, x_coords = np.where(mask == instance_id)
        if y_coords.size == 0:
            continue
            
        # Calculate axis-aligned bounding box
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        bounding_boxes.append((instance_id, [x_min, y_min, x_max, y_max]))
    
    return bounding_boxes


def locate_mask_file(data_root: Path, source: str, 
                    video_id: VideoID, frame_id: FrameID) -> Optional[Path]:
    """
    Locate mask file using multiple layout conventions.
    
    Args:
        data_root: Root directory of PVSG data
        source: Dataset source name
        video_id: Video identifier
        frame_id: Frame identifier
        
    Returns:
        Path to mask file or None if not found
    """
    # Mirror layout: masks organized per video
    mirror_path = data_root / source / "masks" / video_id / f"{frame_id}.png"
    
    # Flat layout: all masks in single directory
    flat_path = data_root / source / "masks" / f"{frame_id}.png"
    
    if mirror_path.is_file():
        return mirror_path
    if flat_path.is_file():
        return flat_path
    return None


def parse_relationship_annotations(raw_annotations: List[Any]) -> List[RelationshipTuple]:
    """
    Parse and normalize relationship annotations from PVSG JSON format.
    
    Args:
        raw_annotations: Raw relationship data from PVSG JSON
        
    Returns:
        List of normalized relationship tuples (subject_id, object_id, predicate, intervals)
    """
    normalized_relationships = []
    
    if not isinstance(raw_annotations, (list, tuple)):
        return normalized_relationships
    
    for annotation in raw_annotations:
        if isinstance(annotation, dict):
            # Extract from dictionary format
            subject_id = annotation.get("subject_tid", 
                                       annotation.get("subject_id", 
                                                     annotation.get("subject")))
            object_id = annotation.get("object_tid", 
                                      annotation.get("object_id", 
                                                    annotation.get("object")))
            predicate = annotation.get("predicate", annotation.get("pred"))
            
            if None in (subject_id, object_id, predicate):
                continue
            
            subject_id_int = safe_integer_conversion(subject_id)
            object_id_int = safe_integer_conversion(object_id)
            
            if None in (subject_id_int, object_id_int):
                continue
            
            # Parse temporal intervals
            temporal_intervals = []
            if isinstance(annotation.get("intervals"), (list, tuple)):
                for interval in annotation["intervals"]:
                    if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                        start = safe_integer_conversion(interval[0])
                        end = safe_integer_conversion(interval[1])
                        if start is not None and end is not None:
                            temporal_intervals.append((start, end))
            else:
                start = safe_integer_conversion(annotation.get("begin_fid", 
                                                              annotation.get("begin")))
                end = safe_integer_conversion(annotation.get("end_fid", 
                                                            annotation.get("end")))
                if start is not None and end is not None:
                    temporal_intervals.append((start, end))
            
            normalized_relationships.append(
                (subject_id_int, object_id_int, str(predicate).strip(), temporal_intervals)
            )
            continue
        
        if isinstance(annotation, (list, tuple)) and len(annotation) >= 3:
            # Extract from list format [subject_id, object_id, predicate, intervals]
            subject_id, object_id, predicate = annotation[0], annotation[1], annotation[2]
            subject_id_int = safe_integer_conversion(subject_id)
            object_id_int = safe_integer_conversion(object_id)
            
            if None in (subject_id_int, object_id_int, predicate):
                continue
            
            temporal_intervals = []
            if len(annotation) >= 4 and isinstance(annotation[3], (list, tuple)):
                for interval in annotation[3]:
                    if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                        start = safe_integer_conversion(interval[0])
                        end = safe_integer_conversion(interval[1])
                        if start is not None and end is not None:
                            temporal_intervals.append((start, end))
            
            normalized_relationships.append(
                (subject_id_int, object_id_int, str(predicate).strip(), temporal_intervals)
            )
    
    return normalized_relationships


def is_relationship_active(frame_index: int, intervals: List[TimeInterval]) -> bool:
    """
    Determine if relationship is active at given frame index.
    
    Args:
        frame_index: Frame number to check
        intervals: List of temporal intervals when relationship is valid
        
    Returns:
        True if relationship is active at specified frame
    """
    if not intervals:
        return True  # No intervals specified: relationship is always active
    
    for start_frame, end_frame in intervals:
        if start_frame <= frame_index <= end_frame:
            return True
    return False


# ==================== TOON FORMAT GENERATION ====================


def generate_toon_format_annotation(objects: List[Dict], 
                                   relationships: List[Dict]) -> str:
    """
    Generate scene graph annotation in TOON format.
    
    Format:
        obj[N]{id,name,x1,y1,x2,y2}:
          id,name,x1,y1,x2,y2
          ...
        rel[M]{subj,pred,obj}:
          subj,pred,obj
          ...
    
    Args:
        objects: List of object dictionaries with keys: obj_id, name, bbox
        relationships: List of relationship dictionaries with keys: sub_id, pred, obj_id
        
    Returns:
        TOON format annotation string
    """
    annotation_lines = []
    
    # Object section
    annotation_lines.append(f"obj[{len(objects)}]{{id,name,x1,y1,x2,y2}}:")
    for obj in objects:
        obj_id = obj["obj_id"]
        name = obj["name"]
        x1, y1, x2, y2 = obj["bbox"]
        annotation_lines.append(f"  {obj_id},{name},{x1},{y1},{x2},{y2}")
    
    # Relationship section
    annotation_lines.append(f"rel[{len(relationships)}]{{subj,pred,obj}}:")
    for rel in relationships:
        subject_id = rel["sub_id"]
        predicate = normalize_relationship_predicate(rel["pred"])
        object_id = rel["obj_id"]
        annotation_lines.append(f"  {subject_id},{predicate},{object_id}")
    
    return "\n".join(annotation_lines)


def scale_bounding_boxes_in_toon(toon_annotation: str, 
                                 scale_x: float, scale_y: float) -> str:
    """
    Scale bounding box coordinates in TOON annotation according to image resizing.
    
    Args:
        toon_annotation: TOON format annotation string
        scale_x: Horizontal scaling factor (target_width / original_width)
        scale_y: Vertical scaling factor (target_height / original_height)
        
    Returns:
        TOON annotation with scaled bounding box coordinates
    """
    annotation_lines = toon_annotation.split('\n')
    scaled_lines = []
    
    for line in annotation_lines:
        stripped_line = line.strip()
        
        # Identify object lines (format: id,name,x1,y1,x2,y2)
        if (stripped_line and ',' in stripped_line and 
            not stripped_line.startswith('obj[') and 
            not stripped_line.startswith('rel[')):
            parts = stripped_line.split(',')
            if len(parts) == 6:
                try:
                    obj_id, name, x1_str, y1_str, x2_str, y2_str = parts
                    x1, y1, x2, y2 = map(float, (x1_str, y1_str, x2_str, y2_str))
                    
                    # Scale coordinates
                    x1_scaled = int(round(x1 * scale_x))
                    y1_scaled = int(round(y1 * scale_y))
                    x2_scaled = int(round(x2 * scale_x))
                    y2_scaled = int(round(y2 * scale_y))
                    
                    # Preserve original indentation
                    indentation = line[:len(line) - len(line.lstrip())]
                    scaled_line = f"{indentation}{obj_id},{name},{x1_scaled},{y1_scaled},{x2_scaled},{y2_scaled}"
                    scaled_lines.append(scaled_line)
                    continue
                except (ValueError, IndexError):
                    pass
        
        scaled_lines.append(line)
    
    return '\n'.join(scaled_lines)


# ==================== VIDEO PROCESSING ====================


def process_single_video(
    video_info: Tuple[VideoID, Dict, str],
    data_root: Path,
    images_out_root: Path,
    repo_root: Path,
    split_name: str,
    max_frames_per_video: int,
) -> Tuple[List[Dict], int, int]:
    """
    Process all frames in a single video.

    Args:
        video_info: (video_id, video_metadata, source)
        data_root: Root with {source}/frames and {source}/masks (OpenPVSG)
        images_out_root: Root for resized PNGs (train_images/ or test_images/)
        repo_root: Repository root for relative paths in JSON
        split_name: 'train' or 'val' (val uses test_images / test_*.json on disk)
        max_frames_per_video: Cap frames per video (0 = all)
    """
    video_id, video_metadata, source = video_info
    
    # Build object ID to category name mapping
    object_id_to_name = {}
    for obj_annotation in video_metadata.get("objects", []):
        obj_id = safe_integer_conversion(
            obj_annotation.get("object_id", obj_annotation.get("id"))
        )
        if obj_id is None:
            continue
        category_name = obj_annotation.get("category", obj_annotation.get("name"))
        if category_name:
            object_id_to_name[obj_id] = category_name
    
    if not object_id_to_name:
        return [], 0, 0
    
    # Parse relationship annotations
    normalized_relationships = parse_relationship_annotations(
        video_metadata.get("relations", [])
    )
    
    # Verify frame directory existence
    frames_directory = data_root / source / "frames" / video_id
    if not frames_directory.exists():
        return [], 0, 0
    
    # Collect frame files
    frame_files = sorted(frames_directory.glob("*.png"))
    if max_frames_per_video > 0:
        frame_files = frame_files[:max_frames_per_video]
    
    if not frame_files:
        return [], 0, 0
    
    split_dir = split_images_subdir(split_name)
    scaled_images_dir = images_out_root / split_dir / source / "frames" / video_id
    scaled_images_dir.mkdir(parents=True, exist_ok=True)
    
    processed_samples = []
    frames_processed = 0
    frames_skipped = 0
    
    for frame_path in frame_files:
        frame_id = frame_path.stem
        frame_index = safe_integer_conversion(frame_id)
        
        # Locate corresponding mask file
        mask_path = locate_mask_file(data_root, source, video_id, frame_id)
        if not mask_path:
            frames_skipped += 1
            continue
        
        # Load segmentation mask
        segmentation_mask = load_segmentation_mask(mask_path)
        if segmentation_mask is None:
            frames_skipped += 1
            continue
        
        # Extract object bounding boxes from mask
        object_boxes = extract_bounding_boxes_from_mask(segmentation_mask)
        if not object_boxes:
            frames_skipped += 1
            continue
        
        # Filter objects present in annotation metadata
        present_objects = []
        for instance_id, bounding_box in object_boxes:
            object_name = object_id_to_name.get(instance_id)
            if object_name:
                present_objects.append((instance_id, bounding_box, object_name))
        
        if not present_objects:
            frames_skipped += 1
            continue
        
        # Renumber objects with sequential local IDs
        local_objects = []
        global_to_local_id_mapping = {}
        for local_id, (instance_id, bounding_box, object_name) in enumerate(present_objects, start=1):
            global_to_local_id_mapping[instance_id] = local_id
            local_objects.append({
                "obj_id": local_id,
                "name": object_name,
                "bbox": bounding_box
            })
        
        # Reconstruct relationships active in current frame
        frame_relationships = []
        seen_relationships = set()
        
        for subject_id, object_id, predicate, intervals in normalized_relationships:
            # Check temporal validity
            if (frame_index is not None and 
                not is_relationship_active(frame_index, intervals)):
                continue
            
            subject_local_id = global_to_local_id_mapping.get(subject_id)
            object_local_id = global_to_local_id_mapping.get(object_id)
            
            # Validate relationship
            if (subject_local_id is None or object_local_id is None or 
                subject_local_id == object_local_id):
                continue
            
            relationship_key = (subject_local_id, object_local_id, predicate)
            if relationship_key in seen_relationships:
                continue
            
            seen_relationships.add(relationship_key)
            frame_relationships.append({
                "sub_id": subject_local_id,
                "pred": predicate,
                "obj_id": object_local_id
            })
        
        # Resize image and adjust bounding boxes
        try:
            original_image = Image.open(frame_path)
            original_width, original_height = original_image.size
            
            # Calculate scaling factors
            width_scale = TARGET_WIDTH / original_width if original_width > 0 else 1.0
            height_scale = TARGET_HEIGHT / original_height if original_height > 0 else 1.0
            
            # Generate TOON format annotation
            toon_annotation = generate_toon_format_annotation(
                local_objects, frame_relationships
            )
            
            # Scale bounding box coordinates if image is resized
            if width_scale != 1.0 or height_scale != 1.0:
                toon_annotation = scale_bounding_boxes_in_toon(
                    toon_annotation, width_scale, height_scale
                )
            
            # Save resized image
            output_image_path = scaled_images_dir / frame_path.name
            if (original_width == TARGET_WIDTH and 
                original_height == TARGET_HEIGHT):
                # Copy original if already at target size
                shutil.copy2(frame_path, output_image_path)
            else:
                resized_image = original_image.resize(
                    (TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR
                )
                resized_image.save(output_image_path)
            
            processed_samples.append(
                {
                    "image_id": f"{video_id}_{frame_id}",
                    "image_path": path_for_json(repo_root, output_image_path),
                    "answer_toon": toon_annotation,
                }
            )
            
            frames_processed += 1
            
        except Exception:
            frames_skipped += 1
            continue
    
    return processed_samples, frames_processed, frames_skipped


def video_processing_wrapper(args_tuple):
    """
    Wrapper function for parallel video processing.
    
    Args:
        args_tuple: Tuple of arguments for process_single_video
        
    Returns:
        Result from process_single_video
    """
    return process_single_video(*args_tuple)


# ==================== MAIN PROCESSING PIPELINE ====================


def execute_dataset_preparation(args: argparse.Namespace) -> None:
    """Load pvsg.json, process videos, write TOON JSON and resized frames."""
    repo_root = Path(args.repo_root).resolve()
    data_root = resolve_under_repo(repo_root, args.pvsg_data_root)
    export_root = resolve_under_repo(repo_root, args.export_root)
    images_out_root = resolve_under_repo(repo_root, args.images_out)
    annotation_path = resolve_under_repo(repo_root, args.annotation)

    if not data_root.exists():
        print(f"Error: PVSG data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    if not annotation_path.exists():
        print(f"Error: Annotation file not found: {annotation_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Repository root: {repo_root}")
    print(f"PVSG data root: {data_root}")
    print(f"Export (JSON): {export_root}")
    print(f"Images out: {images_out_root}")
    print(f"Annotation file: {annotation_path}")

    # Load dataset annotations
    print(f"Loading annotations from {annotation_path}")
    with open(annotation_path, 'r', encoding='utf-8') as annotation_file:
        dataset_annotations = json.load(annotation_file)
    
    # Parse source and split specifications
    selected_sources = [
        source.strip() for source in args.sources.split(',') 
        if source.strip() in SUPPORTED_SOURCES
    ]
    selected_splits = [
        split.strip() for split in args.splits.split(',') 
        if split.strip() in ['train', 'val']
    ]
    
    if not selected_sources:
        print("Error: No valid sources specified", file=sys.stderr)
        sys.exit(1)
    
    if not selected_splits:
        print("Error: No valid splits specified", file=sys.stderr)
        sys.exit(1)
    
    export_root.mkdir(parents=True, exist_ok=True)
    images_out_root.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset split
    splits_progress_bar = tqdm(
        selected_splits,
        desc="Processing dataset splits",
        unit="split",
        position=0,
        leave=True,
        ncols=80
    )
    
    for split_name in splits_progress_bar:
        splits_progress_bar.set_description(f"Processing split: {split_name}")
        
        # Collect videos for current split
        videos_to_process = []
        for source in selected_sources:
            split_data = dataset_annotations.get("split", {}).get(source, {})
            video_ids = split_data.get(split_name, [])
            
            for video_id in video_ids:
                if args.only_video and video_id != args.only_video:
                    continue
                
                # Find video metadata
                video_metadata = next(
                    (video for video in dataset_annotations.get("data", []) 
                     if video.get("video_id") == video_id), 
                    None
                )
                if not video_metadata:
                    continue
                
                # Determine data source
                video_source = determine_data_source(video_id)
                if video_source not in selected_sources:
                    continue
                
                videos_to_process.append((video_id, video_metadata, video_source))
        
        # Apply video limit if specified
        if args.limit_videos > 0:
            videos_to_process = videos_to_process[:args.limit_videos]
        
        splits_progress_bar.set_postfix({
            "videos": len(videos_to_process),
            "split": split_name
        })
        
        if not videos_to_process:
            continue
        
        processing_tasks = [
            (
                video_info,
                data_root,
                images_out_root,
                repo_root,
                split_name,
                args.limit_frames,
            )
            for video_info in videos_to_process
        ]
        
        # Execute video processing (parallel or sequential)
        all_processed_samples = []
        total_frames_processed = 0
        total_frames_skipped = 0
        
        if args.num_workers > 1:
            with Pool(processes=min(args.num_workers, len(processing_tasks))) as pool:
                video_iterator = pool.imap(video_processing_wrapper, processing_tasks)
                for results, frames_processed, frames_skipped in tqdm(
                    video_iterator,
                    total=len(processing_tasks),
                    desc=f"Processing videos ({split_name})",
                    unit="video",
                    position=1,
                    leave=False,
                    ncols=80
                ):
                    if results:
                        all_processed_samples.extend(results)
                    total_frames_processed += frames_processed
                    total_frames_skipped += frames_skipped
        else:
            for task in tqdm(
                processing_tasks,
                total=len(processing_tasks),
                desc=f"Processing videos ({split_name})",
                unit="video",
                position=1,
                leave=False,
                ncols=80
            ):
                results, frames_processed, frames_skipped = video_processing_wrapper(task)
                if results:
                    all_processed_samples.extend(results)
                total_frames_processed += frames_processed
                total_frames_skipped += frames_skipped
        
        # Save processed annotations
        if all_processed_samples:
            out_stem = split_annotation_file_stem(split_name)
            output_path = export_root / f"{out_stem}_annotations_toon_sft.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(
                    all_processed_samples,
                    output_file,
                    ensure_ascii=False,
                    indent=2,
                )

            splits_progress_bar.write(
                f"Split '{split_name}': "
                f"{len(all_processed_samples):,} samples, "
                f"{total_frames_skipped:,} frames skipped -> "
                f"{path_for_json(repo_root, output_path)}"
            )
        else:
            splits_progress_bar.write(
                f"Split '{split_name}': No samples processed"
            )
    
    splits_progress_bar.close()
    print("\nDataset preparation completed successfully.")
    print(f"JSON (relative to repo): {path_for_json(repo_root, export_root)}/")
    print(f"Images (relative to repo): {path_for_json(repo_root, images_out_root)}/")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="PVSG ==> TOON SFT JSON + 640x480 frames (repo-relative paths).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    argument_parser.add_argument(
        "--repo_root",
        default=str(_REPO_ROOT_DEFAULT),
        help="SceneGraphVLM repository root (used to resolve relative paths)",
    )
    argument_parser.add_argument(
        "--pvsg_data_root",
        default=PVSG_DATA_ROOT_REL,
        help="OpenPVSG root (frames/masks live under {source}/frames, {source}/masks)",
    )
    argument_parser.add_argument(
        "--export_root",
        default=EXPORT_ROOT_REL,
        help="Output directory for {train,test}_annotations_toon_sft.json (val split -> test_*)",
    )
    argument_parser.add_argument(
        "--images_out",
        default=IMAGES_OUT_REL,
        help="Output root for resized PNGs (train_images/, test_images/)",
    )
    argument_parser.add_argument(
        "--annotation",
        default=ANNOTATION_REL,
        help="Path to pvsg.json (relative to repo_root or absolute)",
    )
    argument_parser.add_argument(
        "--sources", 
        default="vidor,epic_kitchen,ego4d",
        help="Comma-separated list of data sources to process"
    )
    argument_parser.add_argument(
        "--splits", 
        default="train,val",
        help="Comma-separated list of dataset splits to process"
    )
    argument_parser.add_argument(
        "--limit_videos", 
        type=int, 
        default=0,
        help="Maximum number of videos to process (0 for all)"
    )
    argument_parser.add_argument(
        "--limit_frames", 
        type=int, 
        default=0,
        help="Maximum frames per video to process (0 for all)"
    )
    argument_parser.add_argument(
        "--only_video", 
        default="",
        help="Process only specified video ID"
    )
    argument_parser.add_argument(
        "--num_workers", 
        type=int, 
        default=cpu_count(),
        help="Number of parallel processing workers"
    )
    
    parsed_args = argument_parser.parse_args()
    execute_dataset_preparation(parsed_args)


if __name__ == "__main__":
    main()