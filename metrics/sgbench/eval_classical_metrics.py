#!/usr/bin/env python3
"""
Compute classical scene-graph Recall@K / Mean-Recall@K metrics (as in OED)
from a VLM inference JSONL that contains TOON-formatted `content` (GT) and
`predict` (model output).

The evaluation logic mirrors OED's BasicSceneGraphEvaluator so that numbers
are directly comparable with Table results in the OED paper.

Usage:
    python metrics/sgbench/eval_classical_metrics.py \
        --pred-jsonl metrics/results/checkpoints-inference/sft/AG/<file>.jsonl \
        --output-dir metrics/results/checkpoints-metrics/sft/AG \
        --output-name <file>-classical-metrics.json \
        --mode sgdet \
        --iou-thr 0.5

Modes (same semantics as in OED / STTran / etc.):
    sgdet   - evaluates detection + classification + predicates
    sgcls   - uses GT boxes, evaluates object classes + predicates
    predcls - uses GT boxes AND GT object classes, evaluates only predicates
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Vocabulary (identical to OED/AG)
# ---------------------------------------------------------------------------
OBJ_CLASSES = (
    "background", "person", "bag", "bed", "blanket", "book", "box", "broom",
    "chair", "closet/cabinet", "clothes", "cup/glass/bottle", "dish", "door",
    "doorknob", "doorway", "floor", "food", "groceries", "laptop", "light",
    "medicine", "mirror", "paper/notebook", "phone/camera", "picture",
    "pillow", "refrigerator", "sandwich", "shelf", "shoe", "sofa/couch",
    "table", "television", "towel", "vacuum", "window",
)
OBJ_NAME_TO_IDX = {name: i for i, name in enumerate(OBJ_CLASSES)}

ATTENTION_CLS = ("looking_at", "not_looking_at", "unsure")
SPATIAL_CLS = ("above", "beneath", "in_front_of", "behind", "on_the_side_of", "in")
CONTACTING_CLS = (
    "carrying", "covered_by", "drinking_from", "eating",
    "have_it_on_the_back", "holding", "leaning_on", "lying_on",
    "not_contacting", "other_relationship", "sitting_on", "standing_on",
    "touching", "twisting", "wearing", "wiping", "writing_on",
)

NUM_ATTN = len(ATTENTION_CLS)       # 3
NUM_SPATIAL = len(SPATIAL_CLS)       # 6
NUM_CONTACT = len(CONTACTING_CLS)    # 17
NUM_REL = NUM_ATTN + NUM_SPATIAL + NUM_CONTACT  # 26

ATTN_NAME_TO_LOCAL = {n: i for i, n in enumerate(ATTENTION_CLS)}
SPAT_NAME_TO_LOCAL = {n: i for i, n in enumerate(SPATIAL_CLS)}
CONT_NAME_TO_LOCAL = {n: i for i, n in enumerate(CONTACTING_CLS)}

# Global predicate index: attn 0-2, spatial 3-8, contacting 9-25
PREDICATE_CLASSES = ATTENTION_CLS + SPATIAL_CLS + CONTACTING_CLS


# ---------------------------------------------------------------------------
# TOON parser
# ---------------------------------------------------------------------------
_RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_RE_OBJ_LINE = re.compile(
    r"^\s*(\d+)\s*,\s*([^,]+?)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$",
    re.MULTILINE,
)
_RE_REL_LINE = re.compile(
    r"^\s*(\d+)\s*,\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]\s*,\s*(\d+)\s*$",
    re.MULTILINE,
)


def _norm_obj_name(name: str) -> str:
    return name.strip().lower().replace("-", " ").replace("_", " ")


def _build_obj_lookup() -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, canonical in enumerate(OBJ_CLASSES):
        lookup[canonical] = idx
        lookup[_norm_obj_name(canonical)] = idx
    return lookup


_OBJ_LOOKUP = _build_obj_lookup()


def _obj_name_to_idx(name: str) -> int:
    n = _norm_obj_name(name)
    if n in _OBJ_LOOKUP:
        return _OBJ_LOOKUP[n]
    for canon, idx in _OBJ_LOOKUP.items():
        if n in canon or canon in n:
            return idx
    return 0  # background fallback


def _parse_label_list(text: str) -> List[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def parse_toon(text: str) -> Optional[Dict[str, Any]]:
    """Parse a TOON scene graph from text. Returns None on failure."""
    text = _RE_THINK.sub("", text)
    m = _RE_ANSWER.search(text)
    if not m:
        return None
    body = m.group(1)

    objects: Dict[int, Dict[str, Any]] = {}
    for mo in _RE_OBJ_LINE.finditer(body):
        oid = int(mo.group(1))
        name = mo.group(2).strip()
        x1, y1, x2, y2 = int(mo.group(3)), int(mo.group(4)), int(mo.group(5)), int(mo.group(6))
        objects[oid] = {"name": name, "cls_idx": _obj_name_to_idx(name), "box": [x1, y1, x2, y2]}

    rel_pairs: List[Dict[str, Any]] = []
    for mr in _RE_REL_LINE.finditer(body):
        subj_id, obj_id = int(mr.group(1)), int(mr.group(5))
        attn_labels = _parse_label_list(mr.group(2))
        spat_labels = _parse_label_list(mr.group(3))
        cont_labels = _parse_label_list(mr.group(4))
        rel_pairs.append({
            "subj_id": subj_id,
            "obj_id": obj_id,
            "attention": attn_labels,
            "spatial": spat_labels,
            "contacting": cont_labels,
        })

    if not objects:
        return None

    return {"objects": objects, "rel_pairs": rel_pairs}


# ---------------------------------------------------------------------------
# Convert parsed TOON → OED-compatible gt_entry / pred_entry
# ---------------------------------------------------------------------------

def toon_to_gt_entry(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Build gt_entry dict expected by OED's evaluate_from_dict."""
    id_list = sorted(parsed["objects"].keys())
    id_to_pos = {oid: pos for pos, oid in enumerate(id_list)}

    gt_classes = np.array([parsed["objects"][oid]["cls_idx"] for oid in id_list], dtype=np.int64)
    gt_boxes = np.array([parsed["objects"][oid]["box"] for oid in id_list], dtype=np.float64)

    gt_relations: List[List[int]] = []
    for rp in parsed["rel_pairs"]:
        subj_pos = id_to_pos.get(rp["subj_id"])
        obj_pos = id_to_pos.get(rp["obj_id"])
        if subj_pos is None or obj_pos is None:
            continue

        for a in rp["attention"]:
            if a in ATTN_NAME_TO_LOCAL:
                gt_relations.append([subj_pos, obj_pos, ATTN_NAME_TO_LOCAL[a]])

        for s in rp["spatial"]:
            if s in SPAT_NAME_TO_LOCAL:
                gt_relations.append([obj_pos, subj_pos, NUM_ATTN + SPAT_NAME_TO_LOCAL[s]])

        for c in rp["contacting"]:
            if c in CONT_NAME_TO_LOCAL:
                gt_relations.append([subj_pos, obj_pos, NUM_ATTN + NUM_SPATIAL + CONT_NAME_TO_LOCAL[c]])

    gt_relations_arr = np.array(gt_relations, dtype=np.int64) if gt_relations else np.zeros((0, 3), dtype=np.int64)

    return {
        "gt_classes": gt_classes,
        "gt_relations": gt_relations_arr,
        "gt_boxes": gt_boxes,
    }


def toon_to_pred_entry(
    parsed: Dict[str, Any],
    mode: str = "sgdet",
    gt_parsed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build pred_entry dict expected by OED's evaluate_from_dict.

    For predcls/sgcls modes, GT boxes (and optionally GT classes) are used
    instead of predicted ones, following OED convention.
    """
    id_list = sorted(parsed["objects"].keys())
    id_to_pos = {oid: pos for pos, oid in enumerate(id_list)}

    pred_classes = np.array([parsed["objects"][oid]["cls_idx"] for oid in id_list], dtype=np.int64)
    pred_boxes = np.array([parsed["objects"][oid]["box"] for oid in id_list], dtype=np.float64)
    obj_scores = np.ones(len(id_list), dtype=np.float64)

    if mode in ("predcls", "sgcls") and gt_parsed is not None:
        gt_id_list = sorted(gt_parsed["objects"].keys())
        pred_boxes = np.array([gt_parsed["objects"][oid]["box"] for oid in gt_id_list], dtype=np.float64)
        if mode == "predcls":
            pred_classes = np.array([gt_parsed["objects"][oid]["cls_idx"] for oid in gt_id_list], dtype=np.int64)
        obj_scores = np.ones(len(gt_id_list), dtype=np.float64)

    num_pairs = len(parsed["rel_pairs"])
    if num_pairs == 0:
        empty_rels = np.zeros((0, 2), dtype=np.int64)
        rels_i = np.zeros((0, 2), dtype=np.int64)
        rel_scores = np.zeros((0, NUM_REL), dtype=np.float64)
        return {
            "pred_boxes": pred_boxes,
            "pred_classes": pred_classes,
            "pred_rel_inds": rels_i,
            "obj_scores": obj_scores,
            "rel_scores": rel_scores,
        }

    pair_idx = np.zeros((num_pairs, 2), dtype=np.int64)
    attn_dist = np.zeros((num_pairs, NUM_ATTN), dtype=np.float64)
    spat_dist = np.zeros((num_pairs, NUM_SPATIAL), dtype=np.float64)
    cont_dist = np.zeros((num_pairs, NUM_CONTACT), dtype=np.float64)

    for i, rp in enumerate(parsed["rel_pairs"]):
        subj_pos = id_to_pos.get(rp["subj_id"], 0)
        obj_pos = id_to_pos.get(rp["obj_id"], 0)
        pair_idx[i] = [subj_pos, obj_pos]

        for a in rp["attention"]:
            if a in ATTN_NAME_TO_LOCAL:
                attn_dist[i, ATTN_NAME_TO_LOCAL[a]] = 1.0

        eps = 1e-4
        for j, s in enumerate(rp["spatial"]):
            if s in SPAT_NAME_TO_LOCAL:
                spat_dist[i, SPAT_NAME_TO_LOCAL[s]] = 1.0 - j * eps

        for j, c in enumerate(rp["contacting"]):
            if c in CONT_NAME_TO_LOCAL:
                cont_dist[i, CONT_NAME_TO_LOCAL[c]] = 1.0 - j * eps

    rels_i = np.concatenate([pair_idx, pair_idx[:, ::-1], pair_idx], axis=0)

    scores_1 = np.concatenate([attn_dist, np.zeros((num_pairs, NUM_SPATIAL)), np.zeros((num_pairs, NUM_CONTACT))], axis=1)
    scores_2 = np.concatenate([np.zeros((num_pairs, NUM_ATTN)), spat_dist, np.zeros((num_pairs, NUM_CONTACT))], axis=1)
    scores_3 = np.concatenate([np.zeros((num_pairs, NUM_ATTN)), np.zeros((num_pairs, NUM_SPATIAL)), cont_dist], axis=1)
    rel_scores = np.concatenate([scores_1, scores_2, scores_3], axis=0)

    return {
        "pred_boxes": pred_boxes,
        "pred_classes": pred_classes,
        "pred_rel_inds": rels_i,
        "obj_scores": obj_scores,
        "rel_scores": rel_scores,
    }


# ---------------------------------------------------------------------------
# Evaluation core (self-contained, mirrors OED evaluate_recall.py)
# ---------------------------------------------------------------------------

def _box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes. Returns [N, M] array."""
    b1 = torch.tensor(boxes1, dtype=torch.float64)
    b2 = torch.tensor(boxes2, dtype=torch.float64)
    area1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    return iou.numpy()


def _intersect_2d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """[m1, n] x [m2, n] → [m1, m2] bool — True where rows match."""
    return (x1[..., None] == x2.T[None, ...]).all(1)


def _argsort_desc(scores: np.ndarray) -> np.ndarray:
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def _triplet(predicates, relations, classes, boxes, predicate_scores=None, class_scores=None):
    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))
    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))
    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh):
    keeps = _intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for _ in range(pred_boxes.shape[0])]

    for gt_ind, gt_box, keep_inds in zip(
        np.where(gt_has_match)[0],
        gt_boxes[gt_has_match],
        keeps[gt_has_match],
    ):
        boxes = pred_boxes[keep_inds]
        sub_iou = _box_iou_numpy(gt_box[None, :4], boxes[:, :4])[0]
        obj_iou = _box_iou_numpy(gt_box[None, 4:], boxes[:, 4:])[0]
        inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))

    return pred_to_gt


def evaluate_from_dict(gt_entry, pred_entry, method, iou_thresh=0.5):
    """Run recall evaluation for one frame. Returns dict of R@K values."""
    gt_rels = gt_entry["gt_relations"]
    gt_boxes = gt_entry["gt_boxes"].astype(float)
    gt_classes = gt_entry["gt_classes"]

    pred_rel_inds = pred_entry["pred_rel_inds"]
    rel_scores = pred_entry["rel_scores"]
    pred_boxes = pred_entry["pred_boxes"].astype(float)
    pred_classes = pred_entry["pred_classes"]
    obj_scores = pred_entry["obj_scores"]

    if gt_rels.shape[0] == 0:
        return None

    if pred_rel_inds.shape[0] == 0 or rel_scores.shape[0] == 0:
        zero_recall = {k: 0.0 for k in [10, 20, 50, 100]}
        zero_per_class = {k: [(0, int(np.sum(gt_rels[:, 2] == c))) for c in range(NUM_REL)] for k in [10, 20, 50, 100]}
        return {"recall": zero_recall, "per_class": zero_per_class}

    if method == "no":
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = _argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
    else:
        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1)))
        predicate_scores = rel_scores.max(1)

    gt_triplets, gt_triplet_boxes, _ = _triplet(
        gt_rels[:, 2], gt_rels[:, :2], gt_classes, gt_boxes
    )
    pred_triplets, pred_triplet_boxes, rel_scores_out = _triplet(
        pred_rels[:, 2], pred_rels[:, :2], pred_classes, pred_boxes,
        predicate_scores, obj_scores,
    )

    sorted_scores = rel_scores_out.prod(1)
    order = sorted_scores.argsort()[::-1]
    pred_triplets = pred_triplets[order]
    pred_triplet_boxes = pred_triplet_boxes[order]

    pred_to_gt = _compute_pred_matches(
        gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, iou_thresh
    )

    results: Dict[int, float] = {}
    per_class: Dict[int, List] = {}
    for k in [10, 20, 50, 100]:
        match = reduce(np.union1d, pred_to_gt[:k]) if len(pred_to_gt) > 0 else np.array([])
        results[k] = float(len(match)) / float(gt_rels.shape[0])

        recall_hit = [0] * NUM_REL
        recall_count = [0] * NUM_REL
        for idx_m in match:
            label = int(gt_rels[int(idx_m), 2])
            recall_hit[label] += 1
        for idx_g in range(gt_rels.shape[0]):
            label = int(gt_rels[idx_g, 2])
            recall_count[label] += 1
        per_class[k] = [(recall_hit[c], recall_count[c]) for c in range(NUM_REL)]

    return {"recall": results, "per_class": per_class}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute classical R@K / mR@K scene-graph metrics from VLM inference JSONL.",
    )
    parser.add_argument("--pred-jsonl", required=True, type=Path,
                        help="Inference JSONL with `content` (GT) and `predict` (model) fields.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", required=True,
                        help="Output JSON filename, e.g. model-classical-metrics.json")
    parser.add_argument("--mode", choices=("sgdet", "sgcls", "predcls"), default="sgdet",
                        help="Evaluation mode (default: sgdet)")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N samples (0 = all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name

    print(f"[config] pred_jsonl={args.pred_jsonl}")
    print(f"[config] mode={args.mode}  iou_thr={args.iou_thr}")

    samples: List[Dict[str, Any]] = []
    with args.pred_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"[data] {len(samples)} samples loaded")

    recall_with: Dict[int, List[float]] = {10: [], 20: [], 50: [], 100: []}
    recall_no: Dict[int, List[float]] = {10: [], 20: [], 50: [], 100: []}
    per_class_with: Dict[int, np.ndarray] = {k: np.zeros((NUM_REL, 2)) for k in [10, 20, 50, 100]}
    per_class_no: Dict[int, np.ndarray] = {k: np.zeros((NUM_REL, 2)) for k in [10, 20, 50, 100]}

    n_valid = 0
    n_skip_gt = 0
    n_skip_pred = 0
    n_skip_no_gt_rels = 0

    t0 = time.time()
    for i, sample in enumerate(samples):
        gt_text = sample.get("content", "")
        pred_text = sample.get("predict", "")

        gt_parsed = parse_toon(gt_text)
        if gt_parsed is None:
            n_skip_gt += 1
            continue

        pred_parsed = parse_toon(pred_text)
        if pred_parsed is None:
            n_skip_pred += 1
            continue

        gt_entry = toon_to_gt_entry(gt_parsed)
        if gt_entry["gt_relations"].shape[0] == 0:
            n_skip_no_gt_rels += 1
            continue

        pred_entry = toon_to_pred_entry(pred_parsed, mode=args.mode, gt_parsed=gt_parsed)

        for method, recall_dict, pc_dict in [
            ("with", recall_with, per_class_with),
            ("no", recall_no, per_class_no),
        ]:
            result = evaluate_from_dict(gt_entry, pred_entry, method=method, iou_thresh=args.iou_thr)
            if result is None:
                continue
            for k in [10, 20, 50, 100]:
                recall_dict[k].append(result["recall"][k])
                for c in range(NUM_REL):
                    pc_dict[k][c, 0] += result["per_class"][k][c][0]
                    pc_dict[k][c, 1] += result["per_class"][k][c][1]

        n_valid += 1
        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{len(samples)} ...")

    elapsed = time.time() - t0
    print(f"[done] {n_valid} valid samples evaluated in {elapsed:.1f}s")
    print(f"  skipped: gt_parse_fail={n_skip_gt}  pred_parse_fail={n_skip_pred}  no_gt_rels={n_skip_no_gt_rels}")

    output: Dict[str, Any] = {
        "predictions_file": str(args.pred_jsonl),
        "mode": args.mode,
        "iou_thr": args.iou_thr,
        "num_samples": len(samples),
        "num_valid": n_valid,
        "num_skip_gt_parse": n_skip_gt,
        "num_skip_pred_parse": n_skip_pred,
        "num_skip_no_gt_rels": n_skip_no_gt_rels,
        "eval_time_sec": round(elapsed, 2),
    }

    print()
    for label, recall_dict, pc_dict in [
        ("with_constraint", recall_with, per_class_with),
        ("no_constraint", recall_no, per_class_no),
    ]:
        print(f"====================== {label} ({args.mode}) ============================")
        for k in [10, 20, 50, 100]:
            vals = recall_dict[k]
            r = np.mean(vals) if vals else 0.0
            output[f"{label}/R@{k}"] = round(float(r), 6)
            print(f"  R@{k}: {r:.4f}")

        for k in [10, 20, 50, 100]:
            per_cls = pc_dict[k]
            cls_recalls = []
            for c in range(NUM_REL):
                if per_cls[c, 1] > 0:
                    cls_recalls.append(per_cls[c, 0] / per_cls[c, 1])
                else:
                    cls_recalls.append(0.0)
            mr = np.mean(cls_recalls)
            output[f"{label}/mR@{k}"] = round(float(mr), 6)
            print(f"  mR@{k}: {mr:.4f}")

        print()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
