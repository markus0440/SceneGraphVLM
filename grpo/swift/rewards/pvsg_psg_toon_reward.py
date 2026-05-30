import os
import re
import math
from functools import lru_cache
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from swift.rewards import ORM, orms  # type: ignore

# ============================================================
# PVSG reward ORMs for Swift
# ============================================================

# --------------------------
# Config (env-overridable)
# --------------------------
IMG_W = int(os.getenv("IMG_W", "640"))
IMG_H = int(os.getenv("IMG_H", "480"))

MAX_OBJS = int(os.getenv("MAX_OBJS", "200"))
MAX_RELS = int(os.getenv("MAX_RELS", "400"))

PARSE_CACHE = int(os.getenv("PARSE_CACHE", "8192"))
ASSIGN_CACHE = int(os.getenv("ASSIGN_CACHE", "4096"))

SEM_WEIGHT = float(os.getenv("SEM_WEIGHT", "1.0"))
IOU_WEIGHT = float(os.getenv("IOU_WEIGHT", "2.0"))
BOX_L1_WEIGHT = float(os.getenv("BOX_L1_WEIGHT", "5.0"))

FORMAT_REWARD_WEIGHT = float(os.getenv("FORMAT_REWARD_WEIGHT", "0.5"))
NODE_REWARD_WEIGHT = float(os.getenv("NODE_REWARD_WEIGHT", "1.5"))
EDGE_REWARD_WEIGHT = float(os.getenv("EDGE_REWARD_WEIGHT", "3.0"))
EDGE_PRECISION_REWARD_WEIGHT = float(os.getenv("EDGE_PRECISION_REWARD_WEIGHT", "1.0"))
EDGE_F1_REWARD_WEIGHT = float(os.getenv("EDGE_F1_REWARD_WEIGHT", "2.0"))

EDGE_IOU_THRESH = float(os.getenv("EDGE_IOU_THRESH", "0.5"))
OBJ_IOU_THRESH = float(os.getenv("OBJ_IOU_THRESH", "0.5"))

EDGE_HALLUC_PENALTY_WEIGHT = float(os.getenv("EDGE_FP_PENALTY_WEIGHT", "1.0"))
EDGE_HALLUC_PENALTY_POWER = float(os.getenv("EDGE_FP_PENALTY_POWER", "2.0"))
OBJ_HALLUC_PENALTY_WEIGHT = float(os.getenv("OBJ_HALLUC_PENALTY_WEIGHT", "1.0"))


# Hungarian solver
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# --------------------------
# spaCy similarity (disabled by default, exact-match fallback)
# --------------------------
_HAS_SPACY = False
_NLP = None


def _normalize_text(s: str) -> str:
    s = s.replace("_", " ").replace("-", " ").replace("/", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _canonical_label(s: str) -> str:
    return str(s).strip().lower()


_CANON_LABEL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _is_valid_canonical_label(s: str) -> bool:
    return bool(_CANON_LABEL_RE.fullmatch(_canonical_label(s)))


def _category_key(s: str) -> str:
    s = _canonical_label(s)
    return s if _is_valid_canonical_label(s) else ""


@lru_cache(maxsize=8192)
def _doc_for(token: str):
    if not _HAS_SPACY or _NLP is None:
        return None
    return _NLP(token)


@lru_cache(maxsize=16384)
def semantic_similarity(a: str, b: str) -> float:
    ca = _category_key(a)
    cb = _category_key(b)
    if not ca or not cb:
        return 0.0
    if not _HAS_SPACY or _NLP is None:
        return 1.0 if ca == cb else 0.0
    da = _doc_for(ca)
    db = _doc_for(cb)
    if da is None or db is None:
        return 1.0 if ca == cb else 0.0
    try:
        sim = float(da.similarity(db))
        if math.isnan(sim) or math.isinf(sim):
            return 0.0
        return max(0.0, min(1.0, sim))
    except Exception:
        return 1.0 if ca == cb else 0.0


# --------------------------
# Helpers: completion extraction
# --------------------------
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
_ANSWER_FULL_RE = re.compile(r"\A<answer>\n(?P<body>.*)\n</answer>\Z", re.S)
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
_TRAILING_SPECIAL_RE = re.compile(
    r'(?:\s|["\'])*(?:<\|im_end\|>|<\|endoftext\|>|<\|end\|>)+(?:\s|["\'])*\Z'
)

_OBJ_HDR_RE = re.compile(r"obj\[(\d+)\]\{.*?\}:", re.S | re.I)
_REL_HDR_RE = re.compile(r"rel\[(\d+)\]\{.*?\}:", re.S | re.I)

_STRICT_OBJ_HDR_RE = re.compile(r"\A\s*obj\[(\d+)\]\{id,name,x1,y1,x2,y2\}:\s*\Z", re.I)
_STRICT_REL_HDR_RE = re.compile(r"\A\s*rel\[(\d+)\]\{subj,pred,obj\}:\s*\Z", re.I)

_LABEL_TOKEN = r"([a-z0-9]+(?:-[a-z0-9]+)*)"

_STRICT_OBJ_LINE_RE = re.compile(
    rf"\A\s*(\d+)\s*,\s*{_LABEL_TOKEN}\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\Z",
    re.I,
)

_STRICT_REL_LINE_RE = re.compile(
    rf"\A\s*(\d+)\s*,\s*{_LABEL_TOKEN}\s*,\s*(\d+)\s*\Z",
    re.I,
)


def _strip_trailing_special_tokens(text: str) -> str:
    return _TRAILING_SPECIAL_RE.sub("", text.strip()).strip()


def _prepare_for_strict_parse(raw: str) -> str:
    raw = _strip_trailing_special_tokens(raw.strip())
    raw = _THINK_RE.sub("", raw).strip()
    return raw


def _coerce_completion_text(c: Any) -> str:
    if isinstance(c, (list, tuple)) and c and isinstance(c[0], dict) and "content" in c[0]:
        return str(c[0]["content"])
    if isinstance(c, dict) and "content" in c:
        return str(c["content"])
    return str(c)


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"<answer>(.*)", text, re.S)
    return m2.group(1).strip() if m2 else text.strip()


def _has_answer_tags(text: str) -> bool:
    return bool(_ANSWER_RE.search(text))


Obj = Dict[str, Any]
Rel = Dict[str, Any]
ObjId = int



def _canonical_predicate(s: str) -> str:
    return _category_key(s)


def _build_gt_rel_index(gt_rels, gt_boxes, gt_names):
    idx = defaultdict(list)
    for j, gr in enumerate(gt_rels):
        gs, go = gr["subject"], gr["object"]
        gp = _canonical_predicate(str(gr.get("predicate", "")))
        if gs not in gt_boxes or go not in gt_boxes or not gp:
            continue

        sub_name = _category_key(gt_names.get(gs, str(gs)))
        obj_name = _category_key(gt_names.get(go, str(go)))
        if not sub_name or not obj_name:
            continue

        idx[(gp, sub_name, obj_name)].append((j, gt_boxes[gs], gt_boxes[go]))
    return idx


# --------------------------
# Parsing: PVSG TOON
# --------------------------
def _parse_scene_graph_pvsg_meta(text: str) -> Tuple[List[Obj], List[Rel], Dict[str, Any]]:
    pred_objs: List[Obj] = []
    pred_rels: List[Rel] = []
    id_map: Dict[int, Obj] = {}

    m_obj = _OBJ_HDR_RE.search(text)
    m_rel = _REL_HDR_RE.search(text)
    declared_n = int(m_obj.group(1)) if m_obj else None
    declared_m = int(m_rel.group(1)) if m_rel else None

    has_obj_header = m_obj is not None
    has_rel_header = m_rel is not None

    obj_match = re.search(r"obj\[\d+\]\{.*?\}:(.*?)(?:\n\s*rel\[\d+\]|\Z)", text, re.S | re.I)
    if obj_match:
        for raw in obj_match.group(1).strip().splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                oid = int(parts[0])
                name = parts[1]
                x1, y1, x2, y2 = map(int, parts[2:6])
            except Exception:
                continue
            obj = {"id": oid, "name": name, "bbox": [x1, y1, x2, y2]}
            id_map[oid] = obj
            pred_objs.append(obj)

    n_rel_lines_total = 0
    n_rel_lines_invalid_ref = 0
    n_rel_lines_invalid_pred = 0

    rel_match = re.search(r"rel\[\d+\]\{.*?\}:(.*)$", text, re.S | re.I)
    if rel_match:
        for raw in rel_match.group(1).strip().splitlines():
            line = raw.strip()
            if not line:
                continue
            n_rel_lines_total += 1

            mr = _STRICT_REL_LINE_RE.fullmatch(line)
            if not mr:
                n_rel_lines_invalid_pred += 1
                continue

            subj = int(mr.group(1))
            pred = _canonical_predicate(mr.group(2))
            obj = int(mr.group(3))

            if not pred:
                n_rel_lines_invalid_pred += 1
                continue
            if subj not in id_map or obj not in id_map or subj == obj:
                n_rel_lines_invalid_ref += 1
                continue

            pred_rels.append({"subject": subj, "predicate": pred, "object": obj})

    meta = {
        "declared_n": declared_n,
        "declared_m": declared_m,
        "has_obj_header": has_obj_header,
        "has_rel_header": has_rel_header,
        "n_obj_parsed": len(pred_objs),
        "n_rel_parsed": len(pred_rels),
        "n_rel_lines_total": n_rel_lines_total,
        "n_rel_lines_invalid_ref": n_rel_lines_invalid_ref,
        "n_rel_lines_invalid_pred": n_rel_lines_invalid_pred,
    }
    return pred_objs, pred_rels, meta


def _parse_scene_graph_pvsg_strict(answer_text: str) -> Tuple[List[Obj], List[Rel], Dict[str, Any]]:
    lines = answer_text.splitlines()
    if not lines:
        raise ValueError("empty answer body")
    if any(not ln.strip() for ln in lines):
        raise ValueError("blank lines are not allowed inside <answer>")

    m_obj = _STRICT_OBJ_HDR_RE.fullmatch(lines[0])
    if not m_obj:
        raise ValueError("bad object header")
    declared_n = int(m_obj.group(1))

    rel_header_idx = None
    declared_m = None
    for i in range(1, len(lines)):
        m_rel = _STRICT_REL_HDR_RE.fullmatch(lines[i])
        if m_rel:
            if rel_header_idx is not None:
                raise ValueError("multiple rel headers")
            rel_header_idx = i
            declared_m = int(m_rel.group(1))

    if rel_header_idx is None or declared_m is None:
        raise ValueError("missing rel header")

    obj_lines = lines[1:rel_header_idx]
    rel_lines = lines[rel_header_idx + 1:]

    if len(obj_lines) != declared_n:
        raise ValueError(f"declared_n={declared_n}, actual_obj_lines={len(obj_lines)}")
    if len(rel_lines) != declared_m:
        raise ValueError(f"declared_m={declared_m}, actual_rel_lines={len(rel_lines)}")

    objs: List[Obj] = []
    rels: List[Rel] = []
    seen_obj_ids = set()

    for ln in obj_lines:
        mo = _STRICT_OBJ_LINE_RE.fullmatch(ln)
        if not mo:
            raise ValueError(f"bad object line: {ln!r}")

        oid = int(mo.group(1))
        name = mo.group(2).strip()
        x1 = int(mo.group(3))
        y1 = int(mo.group(4))
        x2 = int(mo.group(5))
        y2 = int(mo.group(6))

        if not name:
            raise ValueError("empty object name")
        if oid in seen_obj_ids:
            raise ValueError(f"duplicate object id: {oid}")

        seen_obj_ids.add(oid)
        objs.append({"id": oid, "name": name, "bbox": [x1, y1, x2, y2]})

    obj_id_set = set(seen_obj_ids)
    expected_ids = set(range(1, declared_n + 1))
    if obj_id_set != expected_ids:
        raise ValueError(f"object ids must be exactly 1..N, got {sorted(obj_id_set)}")

    seen_rel_keys = set()
    for ln in rel_lines:
        mr = _STRICT_REL_LINE_RE.fullmatch(ln)
        if not mr:
            raise ValueError(f"bad relation line: {ln!r}")

        subj = int(mr.group(1))
        pred = _canonical_predicate(mr.group(2))
        obj = int(mr.group(3))

        if not pred:
            raise ValueError("empty predicate")
        if subj not in obj_id_set or obj not in obj_id_set:
            raise ValueError(f"relation references unknown object ids: {ln!r}")
        if subj == obj:
            raise ValueError(f"self-loop is not allowed: {ln!r}")

        key = (subj, pred, obj)
        if key in seen_rel_keys:
            raise ValueError(f"duplicate relation: {ln!r}")
        seen_rel_keys.add(key)

        rels.append({"subject": subj, "predicate": pred, "object": obj})

    meta = {
        "declared_n": declared_n,
        "declared_m": declared_m,
        "has_obj_header": True,
        "has_rel_header": True,
        "n_obj_parsed": len(objs),
        "n_rel_parsed": len(rels),
        "n_rel_lines_total": len(rel_lines),
        "n_rel_lines_invalid_ref": 0,
        "n_rel_lines_invalid_pred": 0,
    }
    return objs, rels, meta

def _pair_stats_cached(gt_text: str, pr_text: str) -> Dict[str, Any]:
    gt_objs, gt_rels = _parse_text_cached(gt_text)
    pr_objs, pr_rels = _parse_text_cached(pr_text)

    gt_by_id = {o["id"]: o for o in gt_objs}
    pr_by_id = {o["id"]: o for o in pr_objs}

    pairs = _assign_pairs_cached(gt_text, pr_text)

    good_obj = 0
    node_sem_sum = 0.0
    node_box_sum = 0.0

    denom_box = max(1e-9, IOU_WEIGHT + BOX_L1_WEIGHT)

    for gt_id, pr_id in pairs:
        gt = gt_by_id.get(gt_id)
        pr = pr_by_id.get(pr_id)
        if not gt or not pr:
            continue

        gt_name = str(gt.get("name", gt.get("id", "")))
        pr_name = str(pr.get("name", pr.get("id", "")))

        sem = semantic_similarity(gt_name, pr_name)
        node_sem_sum += sem * NODE_REWARD_WEIGHT

        iou = _iou(gt["bbox"], pr["bbox"])
        l1n = _box_l1_norm(gt["bbox"], pr["bbox"], IMG_W, IMG_H)
        l1_score = math.exp(-l1n)
        blend = (IOU_WEIGHT * iou + BOX_L1_WEIGHT * l1_score) / denom_box
        node_box_sum += blend * NODE_REWARD_WEIGHT

        if _category_key(gt_name) == _category_key(pr_name) and iou >= OBJ_IOU_THRESH:
            good_obj += 1

    edge_matched, edge_n_gt, edge_n_pr = _count_edge_matches_one_to_one_cached(gt_text, pr_text)

    return {
        "gt_objs": gt_objs,
        "pr_objs": pr_objs,
        "gt_rels": gt_rels,
        "pr_rels": pr_rels,
        "gt_by_id": gt_by_id,
        "pr_by_id": pr_by_id,
        "pairs": pairs,
        "n_gt_objs": len(gt_objs),
        "n_pr_objs": len(pr_objs),
        "n_gt_rels": len(gt_rels),
        "n_pr_rels": len(pr_rels),
        "good_obj": good_obj,
        "node_sem_sum": float(node_sem_sum),
        "node_box_sum": float(node_box_sum),
        "edge_matched": int(edge_matched),
        "edge_n_gt": int(edge_n_gt),
        "edge_n_pr": int(edge_n_pr),
    }

def _get_pair_stats_if_valid(c: Any, sol: Any) -> Optional[Dict[str, Any]]:
    if not _is_strict_valid_completion(c):
        return None

    gt_text = _graph_text(sol)
    pr_text = _graph_text(c)
    return _pair_stats_cached(gt_text, pr_text)

@lru_cache(maxsize=PARSE_CACHE)
def _parse_text_meta_cached(answer_text: str) -> Tuple[List[Obj], List[Rel], Dict[str, Any]]:
    return _parse_scene_graph_pvsg_meta(answer_text)


def _parse_text_cached(text: str) -> Tuple[List[Obj], List[Rel]]:
    objs, rels, _ = _parse_text_meta_cached(text)
    return objs, rels


@lru_cache(maxsize=PARSE_CACHE)
def _graph_text_from_raw(raw_text: str) -> str:
    txt = _extract_answer(raw_text).strip()
    return _strip_trailing_special_tokens(txt)


def _graph_text(x: Any) -> str:
    return _graph_text_from_raw(_coerce_completion_text(x))


def _parse_any_graph(x: Any) -> Tuple[List[Obj], List[Rel]]:
    return _parse_text_cached(_graph_text(x))


# --------------------------
# Geometry
# --------------------------
def _valid_bbox(bb: Sequence[int], w: int = IMG_W, h: int = IMG_H) -> bool:
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(v) for v in bb]
    except Exception:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    return True


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _giou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    iou = inter / union

    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    area_c = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    if area_c <= 0:
        return float(iou)
    return float(iou - (area_c - union) / area_c)


def _box_l1_norm(a: Sequence[int], b: Sequence[int], w: int = IMG_W, h: int = IMG_H) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (
        abs(ax1 - bx1) / max(1.0, float(w))
        + abs(ay1 - by1) / max(1.0, float(h))
        + abs(ax2 - bx2) / max(1.0, float(w))
        + abs(ay2 - by2) / max(1.0, float(h))
    )


# --------------------------
# Hungarian matching
# --------------------------
def _cost_function(pr: Obj, gt: Obj) -> float:
    sem_sim = semantic_similarity(str(pr.get("name", pr.get("id", ""))), str(gt.get("name", gt.get("id", ""))))
    giou = _giou(pr["bbox"], gt["bbox"])
    l1n = _box_l1_norm(pr["bbox"], gt["bbox"], IMG_W, IMG_H)
    return SEM_WEIGHT * (1.0 - sem_sim) + IOU_WEIGHT * (1.0 - giou) + BOX_L1_WEIGHT * l1n


def _assign_objects(gt_objs: List[Obj], pr_objs: List[Obj]) -> List[Dict[str, Any]]:
    if not gt_objs or not pr_objs:
        return []

    n_gt = len(gt_objs)
    n_pr = len(pr_objs)
    cost_mat: List[List[float]] = [[0.0] * n_gt for _ in range(n_pr)]
    for i in range(n_pr):
        for j in range(n_gt):
            cost_mat[i][j] = float(_cost_function(pr_objs[i], gt_objs[j]))

    if _HAS_SCIPY:
        import numpy as np
        cm = np.asarray(cost_mat, dtype=float)
        row_ind, col_ind = linear_sum_assignment(cm)
        out: List[Dict[str, Any]] = []
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if 0 <= r < n_pr and 0 <= c < n_gt:
                out.append({"gt": gt_objs[c], "pr": pr_objs[r], "cost": float(cm[r, c])})
        return out

    used: set[int] = set()
    out: List[Dict[str, Any]] = []
    for j in range(n_gt):
        best_i: Optional[int] = None
        best_cost = float("inf")
        for i in range(n_pr):
            if i in used:
                continue
            c = cost_mat[i][j]
            if c < best_cost:
                best_cost = c
                best_i = i
        if best_i is not None:
            used.add(best_i)
            out.append({"gt": gt_objs[j], "pr": pr_objs[best_i], "cost": float(best_cost)})
    return out


@lru_cache(maxsize=ASSIGN_CACHE)
def _assign_pairs_cached(gt_text: str, pr_text: str) -> Tuple[Tuple[ObjId, ObjId], ...]:
    gt_objs, _ = _parse_text_cached(gt_text)
    pr_objs, _ = _parse_text_cached(pr_text)
    if not gt_objs or not pr_objs:
        return tuple()
    assigns = _assign_objects(gt_objs, pr_objs)
    return tuple((a["gt"]["id"], a["pr"]["id"]) for a in assigns)


def _count_good_object_matches(gt_text: str, pr_text: str) -> Tuple[int, int, int]:
    gt_objs, _ = _parse_text_cached(gt_text)
    pr_objs, _ = _parse_text_cached(pr_text)
    if not gt_objs or not pr_objs:
        return 0, len(gt_objs), len(pr_objs)

    pairs = _assign_pairs_cached(gt_text, pr_text)
    gt_by_id = {o["id"]: o for o in gt_objs}
    pr_by_id = {o["id"]: o for o in pr_objs}

    good = 0
    for gt_id, pr_id in pairs:
        gt = gt_by_id.get(gt_id)
        pr = pr_by_id.get(pr_id)
        if not gt or not pr:
            continue
        if _category_key(str(gt.get("name", ""))) != _category_key(str(pr.get("name", ""))):
            continue
        if _iou(gt["bbox"], pr["bbox"]) < OBJ_IOU_THRESH:
            continue
        good += 1

    return good, len(gt_objs), len(pr_objs)


@lru_cache(maxsize=ASSIGN_CACHE)
def _count_edge_matches_one_to_one_cached(gt_text: str, pr_text: str) -> Tuple[int, int, int]:
    gt_objs, gt_rels = _parse_text_cached(gt_text)
    pr_objs, pr_rels = _parse_text_cached(pr_text)

    n_gt = len(gt_rels)
    n_pr = len(pr_rels)
    if n_gt <= 0 or n_pr <= 0:
        return 0, n_gt, n_pr

    gt_boxes = {o["id"]: o["bbox"] for o in gt_objs}
    pr_boxes = {o["id"]: o["bbox"] for o in pr_objs}
    gt_names = {o["id"]: str(o.get("name", o.get("id", ""))) for o in gt_objs}
    pr_names = {o["id"]: str(o.get("name", o.get("id", ""))) for o in pr_objs}

    cand: Dict[int, List[Tuple[int, float]]] = {}

    gt_index = _build_gt_rel_index(gt_rels, gt_boxes, gt_names)

    for i, pr in enumerate(pr_rels):
        ps, po = pr["subject"], pr["object"]
        pp = _category_key(str(pr.get("predicate", "")))
        if ps not in pr_boxes or po not in pr_boxes or not pp:
            continue

        pr_sub_name = _category_key(pr_names.get(ps, str(ps)))
        pr_obj_name = _category_key(pr_names.get(po, str(po)))
        if not pr_sub_name or not pr_obj_name:
            continue

        matches_here = []
        for j, gt_sub_box, gt_obj_box in gt_index.get((pp, pr_sub_name, pr_obj_name), []):
            iou_sub = _iou(gt_sub_box, pr_boxes[ps])
            iou_obj = _iou(gt_obj_box, pr_boxes[po])
            if iou_sub < EDGE_IOU_THRESH or iou_obj < EDGE_IOU_THRESH:
                continue
            matches_here.append((j, float(iou_sub + iou_obj)))

        if matches_here:
            matches_here.sort(key=lambda x: x[1], reverse=True)
            cand[i] = matches_here

    if not cand:
        return 0, n_gt, n_pr

    gt_to_pr: Dict[int, int] = {}

    def _dfs(pr_idx: int, seen_gt: set) -> bool:
        for gt_idx, _score in cand.get(pr_idx, []):
            if gt_idx in seen_gt:
                continue
            seen_gt.add(gt_idx)
            if gt_idx not in gt_to_pr or _dfs(gt_to_pr[gt_idx], seen_gt):
                gt_to_pr[gt_idx] = pr_idx
                return True
        return False

    matched = 0
    for pr_idx in range(n_pr):
        if _dfs(pr_idx, set()):
            matched += 1
    return matched, n_gt, n_pr


# --------------------------
# Graph validity (format reward)
# --------------------------
def _graph_valid(objs: List[Obj], rels: List[Rel]) -> bool:
    if not objs or len(objs) > MAX_OBJS or len(rels) > MAX_RELS:
        return False

    ids = [o.get("id") for o in objs]
    if len(ids) != len(set(ids)):
        return False

    for o in objs:
        bb = o.get("bbox", None)
        if not isinstance(bb, list) or len(bb) != 4 or not _valid_bbox(bb, IMG_W, IMG_H):
            return False

    idset = set(ids)
    for r in rels:
        if r.get("subject") not in idset or r.get("object") not in idset:
            return False
        pred = r.get("predicate", "")
        if not isinstance(pred, str) or not pred.strip():
            return False

    return True


@lru_cache(maxsize=PARSE_CACHE)
def _strict_valid_from_raw(raw_text: str) -> bool:
    try:
        raw = _prepare_for_strict_parse(raw_text)
        m = _ANSWER_FULL_RE.fullmatch(raw)
        if m is None:
            return False
        answer_body = m.group("body")
        objs, rels, _ = _parse_scene_graph_pvsg_strict(answer_body)
        return _graph_valid(objs, rels)
    except Exception:
        return False


def _is_strict_valid_completion(raw_completion: Any) -> bool:
    return _strict_valid_from_raw(_coerce_completion_text(raw_completion))


# --------------------------
# ORMs
# --------------------------
class FormatRewardORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        for c in completions:
            try:
                score = 1.0 if _is_strict_valid_completion(c) else 0.0
                rewards.append(score * FORMAT_REWARD_WEIGHT)
            except Exception:
                rewards.append(0.0)
        return rewards


class NodeAccRewardORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out: List[float] = []
        for c, sol in zip(completions, solution):
            try:
                stats = _get_pair_stats_if_valid(c, sol)
                if stats is None:
                    out.append(0.0)
                    continue

                n_gt = stats["n_gt_objs"]
                n_pr = stats["n_pr_objs"]
                if n_gt <= 0 or n_pr <= 0 or not stats["pairs"]:
                    out.append(0.0)
                    continue

                score = stats["node_sem_sum"] / max(1, max(n_gt, n_pr))
                out.append(float(score))
            except Exception:
                out.append(0.0)
        return out


class NodeBoxRewardORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out: List[float] = []
        for c, sol in zip(completions, solution):
            try:
                stats = _get_pair_stats_if_valid(c, sol)
                if stats is None:
                    out.append(0.0)
                    continue

                n_gt = stats["n_gt_objs"]
                n_pr = stats["n_pr_objs"]
                if n_gt <= 0 or n_pr <= 0 or not stats["pairs"]:
                    out.append(0.0)
                    continue

                score = stats["node_box_sum"] / max(1, max(n_gt, n_pr))
                out.append(float(score))
            except Exception:
                out.append(0.0)
        return out


class EdgeHardRewardORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out: List[float] = []
        for c, sol in zip(completions, solution):
            try:
                stats = _get_pair_stats_if_valid(c, sol)
                if stats is None:
                    out.append(0.0)
                    continue

                n_gt = stats["edge_n_gt"]
                n_pr = stats["edge_n_pr"]
                if n_gt <= 0 or n_pr <= 0:
                    out.append(0.0)
                    continue

                recall = float(stats["edge_matched"]) / float(max(1, n_gt))
                out.append(float(recall * EDGE_REWARD_WEIGHT))
            except Exception:
                out.append(0.0)
        return out


class EdgePrecisionRewardORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out: List[float] = []
        for c, sol in zip(completions, solution):
            try:
                stats = _get_pair_stats_if_valid(c, sol)
                if stats is None:
                    out.append(0.0)
                    continue

                n_gt = stats["edge_n_gt"]
                n_pr = stats["edge_n_pr"]
                if n_gt <= 0 or n_pr <= 0:
                    out.append(0.0)
                    continue

                precision = float(stats["edge_matched"]) / float(max(1, n_pr))
                out.append(float(precision * EDGE_PRECISION_REWARD_WEIGHT))
            except Exception:
                out.append(0.0)
        return out


class EdgeF1RewardORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out: List[float] = []
        for c, sol in zip(completions, solution):
            try:
                stats = _get_pair_stats_if_valid(c, sol)
                if stats is None:
                    out.append(0.0)
                    continue

                n_gt = stats["edge_n_gt"]
                n_pr = stats["edge_n_pr"]
                if n_gt <= 0 or n_pr <= 0:
                    out.append(0.0)
                    continue

                matched = float(stats["edge_matched"])
                precision = matched / float(max(1, n_pr))
                recall = matched / float(max(1, n_gt))
                f1 = 0.0 if precision + recall <= 1e-12 else 2.0 * precision * recall / (precision + recall)
                out.append(float(f1 * EDGE_F1_REWARD_WEIGHT))
            except Exception:
                out.append(0.0)
        return out


class ObjHallucinationPenaltyORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out: List[float] = []
        for c, sol in zip(completions, solution):
            try:
                stats = _get_pair_stats_if_valid(c, sol)
                if stats is None:
                    out.append(0.0)
                    continue

                n_pr = stats["n_pr_objs"]
                if n_pr <= 0:
                    out.append(0.0)
                    continue

                halluc_frac = max(0.0, float(n_pr - stats["good_obj"])) / float(n_pr)
                penalty = -OBJ_HALLUC_PENALTY_WEIGHT * halluc_frac
                out.append(float(penalty))
            except Exception:
                out.append(0.0)
        return out


class EdgeHallucinationPenaltyORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out: List[float] = []
        for c, sol in zip(completions, solution):
            try:
                stats = _get_pair_stats_if_valid(c, sol)
                if stats is None:
                    out.append(0.0)
                    continue

                n_gt = stats["edge_n_gt"]
                n_pr = stats["edge_n_pr"]
                if n_gt <= 0 or n_pr <= 0:
                    out.append(0.0)
                    continue

                fp = max(0, int(stats["edge_n_pr"]) - int(stats["edge_matched"]))
                fp_frac = float(fp) / float(max(1, n_pr))
                penalty = -EDGE_HALLUC_PENALTY_WEIGHT * (fp_frac ** EDGE_HALLUC_PENALTY_POWER)
                out.append(float(penalty))
            except Exception:
                out.append(0.0)
        return out


class DiagFracNoRelORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        out: List[float] = []
        for c in completions:
            try:
                pr_text = _graph_text(c)
                _, rels, _ = _parse_text_meta_cached(pr_text)
                out.append(1.0 if len(rels) == 0 else 0.0)
            except Exception:
                out.append(1.0)
        return out


class DiagNumPredObjsORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        out: List[float] = []
        for c in completions:
            try:
                pr_text = _graph_text(c)
                objs, _, meta = _parse_text_meta_cached(pr_text)
                out.append(float(meta.get("n_obj_parsed", len(objs))))
            except Exception:
                out.append(0.0)
        return out


class DiagNumPredRelsORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        out: List[float] = []
        for c in completions:
            try:
                pr_text = _graph_text(c)
                _, rels, meta = _parse_text_meta_cached(pr_text)
                out.append(float(meta.get("n_rel_parsed", len(rels))))
            except Exception:
                out.append(0.0)
        return out


class DiagFracInvalidRelRefORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        out: List[float] = []
        for c in completions:
            try:
                pr_text = _graph_text(c)
                _, _, meta = _parse_text_meta_cached(pr_text)
                total = int(meta.get("n_rel_lines_total", 0))
                bad = int(meta.get("n_rel_lines_invalid_ref", 0)) + int(meta.get("n_rel_lines_invalid_pred", 0))
                out.append(float(bad) / float(max(1, total)) if total > 0 else 0.0)
            except Exception:
                out.append(0.0)
        return out


class DiagHasAnswerTagsORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        out: List[float] = []
        for c in completions:
            try:
                raw = _coerce_completion_text(c).strip()
                out.append(1.0 if _has_answer_tags(raw) else 0.0)
            except Exception:
                out.append(0.0)
        return out


# --------------------------
# Registry
# --------------------------
orms["sgg_format_reward"] = FormatRewardORM
orms["sgg_node_acc_reward"] = NodeAccRewardORM
orms["sgg_node_box_reward"] = NodeBoxRewardORM
orms["sgg_edge_hard_reward"] = EdgeHardRewardORM
orms["sgg_edge_precision_reward"] = EdgePrecisionRewardORM
orms["sgg_edge_f1_reward"] = EdgeF1RewardORM
orms["sgg_obj_hallucination_penalty"] = ObjHallucinationPenaltyORM
orms["sgg_edge_hallucination_penalty"] = EdgeHallucinationPenaltyORM

orms["sgg_diag_frac_no_rel"] = DiagFracNoRelORM
orms["sgg_diag_num_pred_objs"] = DiagNumPredObjsORM
orms["sgg_diag_num_pred_rels"] = DiagNumPredRelsORM
orms["sgg_diag_frac_invalid_rel"] = DiagFracInvalidRelRefORM
orms["sgg_diag_has_answer_tags"] = DiagHasAnswerTagsORM
