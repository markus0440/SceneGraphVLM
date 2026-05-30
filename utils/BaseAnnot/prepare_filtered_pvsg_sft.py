#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter PVSG TOON annotations (keyframes + train/test symmetry).

Reads ``train_annotations_toon_sft.json`` and ``test_annotations_toon_sft.json`` from an input
folder (e.g. output of ``prepare_original_pvsg_sft.py``). Applies:

1. Per-video keyframe selection when object sets or relation *counts* change vs last kept frame.
2. Iterative pruning so train and test share object categories and relation types (overlap filter).

Writes the same filenames as in the input folder (e.g. ``train_annotations_toon_sft.json``,
``test_annotations_toon_sft.json``) into ``--output_dir`` — filtered content, familiar names.
All ``image_path`` values in the output are **relative to the SceneGraphVLM repository root**
(POSIX ``/``). Pixels are not touched.

**Run** (repository root = SceneGraphVLM):

  cd /path/to/SceneGraphVLM
  python utils/BaseAnnot/prepare_filtered_pvsg_sft.py

  python utils/BaseAnnot/prepare_filtered_pvsg_sft.py \\
    --input_dir datasets/annotations/PVSG_annot/data_sft_original \\
    --output_dir datasets/annotations/PVSG_annot/data_sft_base_annot
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# utils/BaseAnnot/<this>.py -> repo root = parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = "datasets/annotations/PVSG_annot/data_sft_original"
DEFAULT_OUTPUT_DIR = "datasets/annotations/PVSG_annot/data_sft_base_annot"
DEFAULT_TRAIN_JSON = "train_annotations_toon_sft.json"
DEFAULT_TEST_JSON = "test_annotations_toon_sft.json"


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


def normalize_image_path_repo_relative(image_path: str, repo_root: Path) -> str:
    p = Path(image_path)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p.resolve()).replace("\\", "/")
    return str(Path(image_path).as_posix())

# ===================== ФУНКЦИИ ФИЛЬТРАЦИИ =====================

def extract_video_id_from_image_id(image_id):
    """Извлекает video_id из image_id. Формат: {video_id}_{frame_idx}"""
    if not image_id:
        return None
    # image_id имеет формат: video_id_frame_idx
    # Например: "0001_4164158586_123" -> video_id = "0001_4164158586", frame_idx = 123
    # Или: "P01_123" -> video_id = "P01", frame_idx = 123
    # Или: "1019_3988584418_0" -> video_id = "1019_3988584418", frame_idx = 0
    
    # Ищем последнее подчеркивание и число после него
    parts = image_id.split("_")
    if len(parts) < 2:
        return None
    
    # Последняя часть должна быть числом (frame_idx)
    # Все остальные части - это video_id
    if parts[-1].isdigit():
        video_id = "_".join(parts[:-1])
        return video_id if video_id else None
    
    # Если последняя часть не число, возвращаем все кроме последней
    if len(parts) > 1:
        video_id = "_".join(parts[:-1])
        return video_id if video_id else None
    
    return None

def extract_frame_number_from_image_id(image_id):
    """Извлекает номер кадра из image_id. Формат: {video_id}_{frame_idx}"""
    if not image_id:
        return None
    # image_id имеет формат: video_id_frame_idx
    parts = image_id.split("_")
    if len(parts) < 2:
        return None
    
    # Последняя часть должна быть числом (frame_idx)
    if parts[-1].isdigit():
        try:
            return int(parts[-1])
        except:
            return None
    
    return None

def extract_scene_graph_info_from_toon(annotation_lines):
    """
    Извлекает информацию о scene graph из TOON аннотации.
    
    Returns:
        tuple: (object_names_set, num_relations)
    """
    object_names = set()
    num_relations = 0  # По умолчанию 0
    
    in_obj_block = False
    in_rel_block = False
    found_rel_header = False  # Флаг, что нашли заголовок rel[...]
    
    for line in annotation_lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        if stripped.startswith("obj[") and "{id,name,x1,y1,x2,y2}" in stripped:
            in_obj_block = True
            in_rel_block = False
            continue
        elif stripped.startswith("rel[") and "{subj,pred,obj}" in stripped:
            in_obj_block = False
            in_rel_block = True
            found_rel_header = True
            # Извлекаем количество отношений из заголовка
            rel_match = re.match(r'rel\[(\d+)\]\{.*\}', stripped)
            if rel_match:
                try:
                    num_relations = int(rel_match.group(1))
                except:
                    num_relations = 0
            else:
                num_relations = 0
            continue
        
        if in_obj_block and "," in stripped:
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 2:
                obj_name = parts[1]
                if obj_name:  # Пропускаем пустые имена
                    object_names.add(obj_name)
        
        if in_rel_block and "," in stripped:
            # Если не нашли заголовок или заголовок показал 0, подсчитываем вручную
            if not found_rel_header or num_relations == 0:
                parts = [p.strip() for p in stripped.split(",")]
                if len(parts) >= 3:
                    num_relations += 1
    
    return object_names, num_relations

def select_unique_frames_by_changes(video_scenes):
    """
    Выбирает уникальные кадры на основе изменений в разметке PVSG.
    Включает первый кадр и все кадры, где изменились объекты или количество отношений.
    """
    if not video_scenes:
        return []
    
    selected_indices = [0]  # Всегда включаем первый кадр
    
    first_info = extract_scene_graph_info_from_toon(video_scenes[0].get('annotation_lines', []))
    first_object_names, first_num_relations = first_info
    
    if first_object_names is None:
        first_object_names = set()
    if first_num_relations is None:
        first_num_relations = 0
    
    last_selected_object_names = first_object_names
    last_selected_num_relations = first_num_relations
    
    for i in range(1, len(video_scenes)):
        current_info = extract_scene_graph_info_from_toon(video_scenes[i].get('annotation_lines', []))
        current_object_names, current_num_relations = current_info
        
        if current_object_names is None:
            current_object_names = set()
        if current_num_relations is None:
            current_num_relations = 0
        
        # Проверяем изменения: объекты или количество отношений
        objects_changed = (current_object_names != last_selected_object_names)
        relations_changed = (current_num_relations != last_selected_num_relations)
        
        if objects_changed or relations_changed:
            selected_indices.append(i)
            last_selected_object_names = current_object_names
            last_selected_num_relations = current_num_relations
    
    return selected_indices

def filter_scenes_by_changes(scenes, extract_video_id_func, extract_frame_number_func):
    """Фильтрует сцены по изменениям в разметке."""
    videos = defaultdict(list)
    for scene in scenes:
        # Используем image_id для извлечения video_id и frame_number
        image_id = scene.get('image_id', '')
        video_id = extract_video_id_func(image_id)
        if video_id:
            videos[video_id].append(scene)
    
    filtered_scenes = []
    for video_id, video_scenes in videos.items():
        video_scenes_with_frame = []
        for scene in video_scenes:
            image_id = scene.get('image_id', '')
            frame_num = extract_frame_number_func(image_id)
            if frame_num is not None:
                video_scenes_with_frame.append((frame_num, scene))
        video_scenes_with_frame.sort(key=lambda x: x[0])
        sorted_scenes = [scene for _, scene in video_scenes_with_frame]
        selected_indices = select_unique_frames_by_changes(sorted_scenes)
        for idx in selected_indices:
            filtered_scenes.append(sorted_scenes[idx])
    
    return filtered_scenes

def extract_objects_and_relations_from_toon(annotation_lines):
    """Извлекает объекты и типы отношений из TOON аннотации."""
    objects = set()
    relation_types = set()
    
    in_obj_block = False
    in_rel_block = False
    
    for line in annotation_lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        if stripped.startswith("obj[") and "{id,name,x1,y1,x2,y2}" in stripped:
            in_obj_block = True
            in_rel_block = False
            continue
        elif stripped.startswith("rel[") and "{subj,pred,obj}" in stripped:
            in_obj_block = False
            in_rel_block = True
            continue
        
        if in_obj_block and "," in stripped:
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 2:
                obj_name = parts[1]
                objects.add(obj_name)
        
        if in_rel_block and "," in stripped:
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 3:
                pred = parts[1]
                relation_types.add(pred)
    
    return objects, relation_types

def filter_scenes_by_overlap(train_scenes, test_scenes):
    """Итеративная фильтрация по перекрытию объектов и типов отношений."""
    def scene_has_unique_items(scene, unique_objects, unique_relation_types):
        objects, relation_types = extract_objects_and_relations_from_toon(scene.get('annotation_lines', []))
        has_unique_object = bool(objects & unique_objects)
        has_unique_relation_type = bool(relation_types & unique_relation_types)
        return has_unique_object or has_unique_relation_type
    
    def collect_objects_and_relations(scenes):
        objects_all = set()
        relation_types_all = set()
        for scene in scenes:
            objects, relation_types = extract_objects_and_relations_from_toon(scene.get('annotation_lines', []))
            objects_all.update(objects)
            relation_types_all.update(relation_types)
        return objects_all, relation_types_all
    
    current_train = train_scenes
    current_test = test_scenes
    iteration = 0
    
    while True:
        iteration += 1
        if iteration > 100:
            break
        
        train_objects_all, train_relation_types_all = collect_objects_and_relations(current_train)
        test_objects_all, test_relation_types_all = collect_objects_and_relations(current_test)
        
        objects_train_only = train_objects_all - test_objects_all
        objects_test_only = test_objects_all - train_objects_all
        relation_types_train_only = train_relation_types_all - test_relation_types_all
        relation_types_test_only = test_relation_types_all - train_relation_types_all
        
        total_unique = (len(objects_train_only) + len(objects_test_only) + 
                       len(relation_types_train_only) + len(relation_types_test_only))
        
        if total_unique == 0:
            break
        
        train_filtered = [
            scene for scene in current_train
            if not scene_has_unique_items(scene, objects_train_only, relation_types_train_only)
        ]
        
        test_filtered = [
            scene for scene in current_test
            if not scene_has_unique_items(scene, objects_test_only, relation_types_test_only)
        ]
        
        if len(train_filtered) == len(current_train) and len(test_filtered) == len(current_test):
            break
        
        current_train = train_filtered
        current_test = test_filtered
    
    return current_train, current_test

# ===================== ЗАГРУЗКА И ОБРАБОТКА =====================

def load_scenes_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load samples from prepare_original_pvsg_sft-style JSON array."""
    scenes: List[Dict[str, Any]] = []

    if not path.is_file():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return scenes

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        image_path = item.get("image_path", "")
        answer_toon = item.get("answer_toon", "")

        if not answer_toon:
            continue

        annotation_lines = answer_toon.split("\n")

        scenes.append(
            {
                "image_id": item.get("image_id", ""),
                "image_path": image_path,
                "annotation_lines": annotation_lines,
            }
        )

    return scenes

def process_filtered_scenes(
    scenes: List[Dict[str, Any]],
    out_json_file: Path,
    split_name: str,
    repo_root: Path,
) -> int:
    """Write filtered samples; ``image_path`` normalized relative to repo root."""
    samples: List[Dict[str, Any]] = []

    pbar = tqdm(
        total=len(scenes),
        desc=f"[{split_name.upper():<5}]",
        unit="scene",
        mininterval=1.0,
        ncols=120,
    )

    for scene in scenes:
        image_path = scene.get("image_path", "")
        annotation_lines = scene.get("annotation_lines", [])

        if not image_path or not annotation_lines:
            pbar.update(1)
            continue

        toon_str = "\n".join(annotation_lines)
        rel_img = normalize_image_path_repo_relative(image_path, repo_root)

        samples.append(
            {
                "image_id": scene.get("image_id", ""),
                "image_path": rel_img,
                "answer_toon": toon_str,
            }
        )
        pbar.update(1)

    pbar.close()

    out_json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return len(samples)


# ===================== MAIN =====================


def _pct(part: int, whole: int) -> str:
    if not whole:
        return "n/a"
    return f"{part / whole * 100:.1f}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PVSG TOON JSON folder -> filtered JSON (same filenames) in output dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo_root",
        default=str(_REPO_ROOT_DEFAULT),
        help="SceneGraphVLM repo root (resolves relative paths)",
    )
    p.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help="Folder with train/test *_annotations_toon_sft.json from prepare_original_pvsg_sft.py",
    )
    p.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for filtered train/test JSON (same basenames as --train_json / --test_json)",
    )
    p.add_argument(
        "--train_json",
        default=DEFAULT_TRAIN_JSON,
        help="Train annotation filename inside --input_dir",
    )
    p.add_argument(
        "--test_json",
        default=DEFAULT_TEST_JSON,
        help="Test annotation filename inside --input_dir",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    input_dir = resolve_under_repo(repo_root, args.input_dir)
    output_dir = resolve_under_repo(repo_root, args.output_dir)

    train_in = input_dir / args.train_json
    test_in = input_dir / args.test_json
    train_out = output_dir / args.train_json
    test_out = output_dir / args.test_json

    print("=" * 70)
    print("PVSG filtered SFT JSON (keyframe + train/test vocabulary overlap)")
    print("=" * 70)
    print(f"Repository root: {repo_root}")
    print(f"Input dir:  {path_for_json(repo_root, input_dir)}")
    print(f"Output dir: {path_for_json(repo_root, output_dir)}")

    if not input_dir.is_dir():
        print(f"[ERROR] input_dir is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not train_in.is_file() or not test_in.is_file():
        print(
            f"[ERROR] Need both:\n  {train_in}\n  {test_in}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nLoading JSON...")
    train_scenes_original = load_scenes_from_json(train_in)
    test_scenes_original = load_scenes_from_json(test_in)
    print(f"Train: {len(train_scenes_original):,} samples")
    print(f"Test:  {len(test_scenes_original):,} samples")

    print("\nKeyframe filter (label change)...")
    train_scenes_filtered = filter_scenes_by_changes(
        train_scenes_original,
        extract_video_id_from_image_id,
        extract_frame_number_from_image_id,
    )
    test_scenes_filtered = filter_scenes_by_changes(
        test_scenes_original,
        extract_video_id_from_image_id,
        extract_frame_number_from_image_id,
    )
    print(f"After keyframe filter — Train: {len(train_scenes_filtered):,}")
    print(f"After keyframe filter — Test:  {len(test_scenes_filtered):,}")

    print("\nOverlap filter (shared object names & relation types)...")
    train_scenes_final, test_scenes_final = filter_scenes_by_overlap(
        train_scenes_filtered,
        test_scenes_filtered,
    )
    print(f"After overlap filter — Train: {len(train_scenes_final):,}")
    print(f"After overlap filter — Test:  {len(test_scenes_final):,}")

    print("\nWriting TRAIN:")
    train_count = process_filtered_scenes(
        train_scenes_final,
        train_out,
        "train",
        repo_root,
    )

    print("\nWriting TEST:")
    test_count = process_filtered_scenes(
        test_scenes_final,
        test_out,
        "test",
        repo_root,
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Original Train / Test: {len(train_scenes_original):,} / {len(test_scenes_original):,}")
    print(
        f"After keyframe: {len(train_scenes_filtered):,} / {len(test_scenes_filtered):,} "
        f"({_pct(len(train_scenes_filtered), len(train_scenes_original))} / "
        f"{_pct(len(test_scenes_filtered), len(test_scenes_original))})"
    )
    print(
        f"After overlap: {len(train_scenes_final):,} / {len(test_scenes_final):,} "
        f"({_pct(len(train_scenes_final), len(train_scenes_original))} / "
        f"{_pct(len(test_scenes_final), len(test_scenes_original))})"
    )
    print(f"\nSaved {train_count} train -> {path_for_json(repo_root, train_out)}")
    print(f"Saved {test_count} test  -> {path_for_json(repo_root, test_out)}")
    print("[SUCCESS] Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
