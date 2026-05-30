#!/usr/bin/env python3
"""AG rel_pairs parser, cleaner, and path helpers for annotation preparation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


IMG_W = 640
IMG_H = 480

OBJ_HEADER_RE = re.compile(r"^\s*obj\[(\d+)\]\{id,name,x1,y1,x2,y2\}:\s*$")
RELPAIRS_HEADER_RE = re.compile(
    r"^\s*rel_pairs\[(\d+)\]\{subj,attention,spatial,contacting,obj\}:\s*$"
)
ANSWER_CLOSE_RE = re.compile(r"^\s*</answer>\s*$")
OBJ_LINE_RE = re.compile(
    r"^\s*(\d+)\s*,\s*([^,]+?)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$"
)


@dataclass
class Obj:
    obj_id: int
    name: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class RelPair:
    subj: int
    attention: List[str]
    spatial: List[str]
    contacting: List[str]
    obj: int


def split_relpair_line(text: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in text.strip():
        if ch == "[":
            depth += 1
            cur.append(ch)
        elif ch == "]":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def parse_label_list(token: str) -> List[str]:
    token = token.strip()
    if not token.startswith("[") or not token.endswith("]"):
        raise ValueError(f"Bad AG label-list token: {token!r}")
    inner = token[1:-1].strip()
    if not inner:
        return []
    return [x.strip() for x in inner.split(",") if x.strip()]


def parse_ag_answer(text: str) -> Tuple[List[Optional[Obj]], List[Optional[RelPair]]]:
    lines = text.splitlines()
    obj_idx = None
    rel_idx = None
    for i, line in enumerate(lines):
        if obj_idx is None and OBJ_HEADER_RE.match(line):
            obj_idx = i
        if rel_idx is None and RELPAIRS_HEADER_RE.match(line):
            rel_idx = i
    if obj_idx is None or rel_idx is None or rel_idx < obj_idx:
        raise ValueError("Cannot find AG obj/rel_pairs block")

    objs: List[Optional[Obj]] = []
    for line in lines[obj_idx + 1 : rel_idx]:
        if not line.strip():
            continue
        match = OBJ_LINE_RE.match(line)
        if not match:
            objs.append(None)
            continue
        oid, name, x1, y1, x2, y2 = match.groups()
        objs.append(Obj(int(oid), name.strip(), int(x1), int(y1), int(x2), int(y2)))

    rels: List[Optional[RelPair]] = []
    for line in lines[rel_idx + 1 :]:
        if ANSWER_CLOSE_RE.match(line):
            break
        if not line.strip():
            continue
        try:
            parts = split_relpair_line(line)
            if len(parts) != 5:
                rels.append(None)
                continue
            rels.append(
                RelPair(
                    subj=int(parts[0]),
                    attention=parse_label_list(parts[1]),
                    spatial=parse_label_list(parts[2]),
                    contacting=parse_label_list(parts[3]),
                    obj=int(parts[4]),
                )
            )
        except Exception:
            rels.append(None)
    return objs, rels


def bbox_valid(obj: Obj, width: int = IMG_W, height: int = IMG_H) -> bool:
    return 0 <= obj.x1 < obj.x2 <= width and 0 <= obj.y1 < obj.y2 <= height


def render_ag_answer(objs: List[Obj], rels: List[RelPair]) -> str:
    out = ["<answer>", f"      obj[{len(objs)}]{{id,name,x1,y1,x2,y2}}:"]
    for obj in objs:
        out.append(f"        {obj.obj_id},{obj.name},{obj.x1},{obj.y1},{obj.x2},{obj.y2}")
    out.append(f"      rel_pairs[{len(rels)}]{{subj,attention,spatial,contacting,obj}}:")
    for rel in rels:
        out.append(
            f"        {rel.subj},[{','.join(rel.attention)}],[{','.join(rel.spatial)}],"
            f"[{','.join(rel.contacting)}],{rel.obj}"
        )
    out.append("</answer>")
    return "\n".join(out) + "\n"


def clean_ag_solution_text(text: str, width: int = IMG_W, height: int = IMG_H) -> Tuple[str, Dict[str, int]]:
    objs_raw, rels_raw = parse_ag_answer(text)
    stats = {
        "objects_total": 0,
        "objects_removed_bad_bbox": 0,
        "objects_removed_malformed": 0,
        "rels_total": 0,
        "rels_removed_missing_obj": 0,
        "rels_removed_malformed": 0,
    }

    objs: List[Obj] = []
    for obj in objs_raw:
        if obj is None:
            stats["objects_removed_malformed"] += 1
            continue
        stats["objects_total"] += 1
        if not bbox_valid(obj, width=width, height=height):
            stats["objects_removed_bad_bbox"] += 1
            continue
        objs.append(obj)

    old_to_new: Dict[int, int] = {}
    for new_id, obj in enumerate(objs, start=1):
        old_to_new[obj.obj_id] = new_id
        obj.obj_id = new_id

    rels: List[RelPair] = []
    for rel in rels_raw:
        if rel is None:
            stats["rels_removed_malformed"] += 1
            continue
        stats["rels_total"] += 1
        if rel.subj not in old_to_new or rel.obj not in old_to_new:
            stats["rels_removed_missing_obj"] += 1
            continue
        rels.append(
            RelPair(
                subj=old_to_new[rel.subj],
                attention=rel.attention,
                spatial=rel.spatial,
                contacting=rel.contacting,
                obj=old_to_new[rel.obj],
            )
        )

    return render_ag_answer(objs, rels), stats


def ag_relation_count(text: str) -> int:
    _objs, rels = parse_ag_answer(text)
    return sum(1 for rel in rels if rel is not None)


def normalize_ag_workspace_image_path(image_path: str, split: str) -> str:
    raw = image_path.strip().replace("\\", "/")
    match = re.search(r"AG_frames/(?:train_images|test_images)/(.*)$", raw)
    if match:
        tail = match.group(1)
    else:
        match = re.search(r"(?:^|/)(?:train_images|test_images)/(.*)$", raw)
        tail = match.group(1) if match else raw.lstrip("/")
    return f"/workspace/datasets/frames/AG_frames/{split}_images/{tail}".replace("//", "/")
