#!/usr/bin/env python
"""
=============================================================================
РАСЧЁТ МЕТРИК ДЛЯ PSG-SGG: ФОРМУЛЫ + QWEN ТОЛЬКО ДЛЯ СИНОНИМОВ
=============================================================================

Метрики считаются строго по формулам; Qwen3-4B-Instruct (судья синонимов) вызывается только
в «спорных» случаях (несовпадение названия объекта или предиката отношения)
и возвращает 1 (синонимы) или 0 (нет).

ЛОГИКА:
1) IoU и матчинг объектов — венгерский алгоритм (оптимальное назначение по IoU).
2) Для объектов: Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2/(1/P + 1/R).
   TP/FP/FN считаются по одному-одному матчу после проверки имени: точное совпадение
   или синонимы (при несовпадении — один вызов Qwen: «термин1 и термин2 синонимы? Ответ: 1 или 0»).
3) Аналогично для отношений: матчинг по парам (subject, object) после маппинга pred_id->gt_id,
   предикаты — точное совпадение или Qwen только при несовпадении.
4) Qwen вызывается только при несовпадении строк (объект или предикат), не для полного графа.

АРХИТЕКТУРА:
- Вход: model_prediction/{split}_pred_{model_name}_step{step}.jsonl
- Выход: model_metrics/{split}_metrics_{model_name}_step{step}.jsonl

ПРИМЕР ЗАПУСКА:
  python eval_psg_sgg_qwen_metrics.py --data-root /path/to/data --split test \\
    --model-name InternVL3-1B --ckpt-step 1890-merged --iou-thr 0.5 \\
    --batch-size-qwen 32 [--qwen-model-path /path/to/Qwen3-4B-Instruct-2507]

Зависимость: scipy (для linear_sum_assignment).
=============================================================================
"""

import csv
import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ------------------------ ПАРАМЕТРЫ ПО УМОЛЧАНИЮ ------------------------ #

QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# ------------------------ ВРЕМЯ ------------------------ #

def stats_with_phys(times: np.ndarray, unit: str = "s") -> Dict:
    n = len(times)
    if n == 0:
        return {"N": 0, "mean": None, "sigma": None, "median": None, "phys": "N/A"}

    mean = float(times.mean())
    sigma = float(times.std(ddof=1)) if n > 1 else 0.0
    median = float(np.median(times))
    phys = f"({mean:.3f} ± {sigma:.3f}) {unit}"
    return {
        "N": n,
        "mean": mean,
        "sigma": sigma,
        "median": median,
        "phys": phys,
    }

# ------------------------ TOON + IoU ------------------------ #
# Парсер TOON — как у коллеги (регексы, фоллбэки, очистка ``` / тегов <answer>).
# Qwen3.5 + SFT с тегами: ответ вида <answer>\n  obj[N]...\n  rel[M]...\n</answer>
# и/или префикс thinking до </think> — нормализуем перед разбором.

BBox = List[int]  # [x1, y1, x2, y2]

TOON_OBJ_HEADER = re.compile(r"obj\s*\[\s*(\d+)\s*\]\s*\{.*?\}\s*:\s*", re.IGNORECASE)
TOON_REL_HEADER = re.compile(r"rel\s*\[\s*(\d+)\s*\]\s*\{.*?\}\s*:\s*", re.IGNORECASE)

_THINK_END = "</think>"
_THINK_STRIP_OBJ_HDR = re.compile(r"^\s*obj\[\d+\]\{id,name", re.MULTILINE)
_THINK_STRIP_REL_HDR = re.compile(r"^\s*rel\[\d+\]\{subj,pred,obj\}", re.MULTILINE)


def _strip_thinking_qwen35(text: str) -> str:
    """Убирает цепочку thinking до TOON (как infer_sg_qwen_swift_batch / real)."""
    s = text.strip()
    while _THINK_END in s:
        idx = s.find(_THINK_END)
        prefix = s[:idx]
        suffix = s[idx + len(_THINK_END) :].lstrip()
        if _THINK_STRIP_OBJ_HDR.search(prefix) or _THINK_STRIP_REL_HDR.search(prefix):
            s = (prefix + "\n" + suffix).strip() if suffix else prefix.strip()
            break
        if not suffix:
            return prefix.strip() if prefix.strip() else ""
        s = suffix
    return s


def _unwrap_answer_tags(text: str) -> str:
    """Извлекает TOON из последнего содержательного <answer>...</answer>, иначе снимает теги."""
    if not text:
        return text
    blocks = list(re.finditer(r"<answer\s*>([\s\S]*?)</answer\s*>", text, re.IGNORECASE))
    if blocks:
        for m in reversed(blocks):
            inner = m.group(1).strip()
            if inner and re.search(r"obj\s*\[\s*\d+\s*\]", inner, re.IGNORECASE):
                return inner
        inner_last = blocks[-1].group(1).strip()
        if inner_last:
            return inner_last
    return re.sub(r"<\/?answer\s*>", " ", text, flags=re.IGNORECASE)


def _normalize_toon_text(text: str) -> str:
    """Thinking + <answer> (GT/предсказание после SFT с тегами)."""
    t = _strip_thinking_qwen35(text)
    t = _unwrap_answer_tags(t)
    return t.strip()


def _refine_node_edge(x: str) -> str:
    return x.replace("_", " ").replace("-", " ").strip().lower()


def _parse_toon(text: str) -> Tuple[List[Dict], List[Dict]]:
    """Парсер TOON как у коллеги. Возвращает (objects, rels).
    objects: list of {toon_id, name, bbox: [x1,y1,x2,y2] (float)}
    rels: list of {sub_id, predicate, obj_id}
    При ошибке парсинга или отсутствии объектов бросает ValueError.
    """
    if text is None:
        raise ValueError("empty prediction")

    text = _normalize_toon_text(text)
    if not text:
        raise ValueError("empty prediction")

    cleaned = text.replace("```", " ")
    cleaned = re.sub(r"<\/?answer\s*>", " ", cleaned, flags=re.IGNORECASE)

    lines = [ln.rstrip() for ln in cleaned.splitlines() if ln.strip()]

    obj_start = None
    rel_start = None
    for i, ln in enumerate(lines):
        if obj_start is None and TOON_OBJ_HEADER.search(ln):
            obj_start = i
        if rel_start is None and TOON_REL_HEADER.search(ln):
            rel_start = i

    if obj_start is None:
        for i, ln in enumerate(lines):
            if ln.count(",") >= 5 and re.match(r"\s*\d+\s*,", ln):
                obj_start = i
                break

    if obj_start is None:
        raise ValueError("cannot locate obj block")

    if rel_start is None:
        for i in range(obj_start + 1, len(lines)):
            ln = lines[i]
            if ln.count(",") >= 2 and re.match(r"\s*\d+\s*,", ln):
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) >= 3 and parts[0].isdigit() and parts[2].isdigit():
                    rel_start = i
                    break

    obj_lines = []
    rel_lines = []
    for i, ln in enumerate(lines[obj_start:]):
        abs_i = obj_start + i
        if abs_i == obj_start:
            if ":" in ln:
                tail = ln.split(":", 1)[1].strip()
                if tail:
                    obj_lines.append(tail)
            continue
        if rel_start is not None and abs_i >= rel_start:
            break
        obj_lines.append(ln)

    if rel_start is not None:
        for i, ln in enumerate(lines[rel_start:]):
            if i == 0:
                if ":" in ln:
                    tail = ln.split(":", 1)[1].strip()
                    if tail:
                        rel_lines.append(tail)
                continue
            rel_lines.append(ln)

    objects = []
    for ln in obj_lines:
        if ln.lower().startswith("rel["):
            break
        # Повтор заголовка obj[N]{...}: (часто после prefill «obj[») — даёт 6 «полей» в csv, но row[0] не int
        if TOON_OBJ_HEADER.search(ln) and not re.match(r"^\s*\d+\s*,", ln):
            continue
        if ln.count(",") < 5:
            continue
        try:
            row = next(csv.reader([ln], skipinitialspace=True))
        except Exception:
            continue
        if len(row) < 6:
            continue
        if not row[0].strip().isdigit():
            continue
        try:
            toon_id = int(row[0].strip())
            name = _refine_node_edge(row[1])
            coords = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
        except (ValueError, TypeError):
            continue
        objects.append({"toon_id": toon_id, "name": name, "bbox": coords})

    rels = []
    for ln in rel_lines:
        if ln.count(",") < 2:
            continue
        try:
            row = next(csv.reader([ln], skipinitialspace=True))
        except Exception:
            continue
        if len(row) < 3:
            continue
        if not row[0].strip().isdigit() or not row[2].strip().isdigit():
            continue
        sub_id = int(row[0].strip())
        predicate = _refine_node_edge(row[1])
        obj_id = int(row[2].strip())
        rels.append({"sub_id": sub_id, "predicate": predicate, "obj_id": obj_id})

    if not objects:
        raise ValueError("no objects parsed")

    return objects, rels


def parse_toon_scene(toon: str) -> Dict:
    """Парсит TOON-текст в формат {"objects": [...], "relationships": [...]}.
    Использует парсер коллеги (_parse_toon); при любой ошибке возвращает пустую сцену.
    """
    if toon is None:
        return {"objects": [], "relationships": []}
    try:
        objects_raw, rels_raw = _parse_toon(toon)
        objects = []
        for o in objects_raw:
            x1, y1, x2, y2 = o["bbox"]
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            objects.append({
                "id": o["toon_id"],
                "name": o["name"],
                "bbox": [x1, y1, x2, y2],
            })
        relationships = [
            {"s_id": r["sub_id"], "pred": r["predicate"], "o_id": r["obj_id"]}
            for r in rels_raw
        ]
        return {"objects": objects, "relationships": relationships}
    except Exception:
        return {"objects": [], "relationships": []}


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    aw = max(0, ax2 - ax1)
    ah = max(0, ay2 - ay1)
    bw = max(0, bx2 - bx1)
    bh = max(0, by2 - by1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def hungarian_iou_match(
    gt_objs: List[Dict],
    pred_objs: List[Dict],
    iou_thr: float,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Матчинг объектов по IoU через венгерский алгоритм.
    Возвращает список (gt_idx, pred_idx, iou) и матрицу IoU (n_gt x n_pred).
    В матч попадают только пары с IoU >= iou_thr.
    """
    n_gt = len(gt_objs)
    n_pred = len(pred_objs)
    if n_gt == 0 or n_pred == 0:
        return [], np.zeros((max(n_gt, 1), max(n_pred, 1)))

    iou_mat = np.zeros((n_gt, n_pred))
    for gi in range(n_gt):
        for pj in range(n_pred):
            iou_mat[gi, pj] = bbox_iou(gt_objs[gi]["bbox"], pred_objs[pj]["bbox"])

    # Венгерский алгоритм минимизирует стоимость; стоимость = -IoU
    cost = -iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for gi, pj in zip(row_ind, col_ind):
        iou = float(iou_mat[gi, pj])
        if iou >= iou_thr:
            matches.append((gi, pj, iou))
    return matches, iou_mat

# ------------------------ QWEN: ТОЛЬКО СИНОНИМЫ (1/0) ------------------------ #

SYSTEM_SYNONYM = """You are a judge for scene graph evaluation. 
You must answer with a single digit: 1 if the two given terms are synonymous in the given context, 0 otherwise. 
No explanation, only 1 or 0."""

SYSTEM_TRIPLET = """You are a judge for scene graph evaluation. 
You must answer with a single digit: 1 if the two given relationship triplets (subject, predicate, object) 
are semantically equivalent in the scene context, 0 otherwise. No explanation, only 1 or 0."""

def _build_synonym_prompt(term_gt: str, term_pred: str, context: str) -> str:
    """Промпт без контекста сцены (fallback)."""
    if context == "object":
        ctx = "object/category labels in a scene graph (e.g. person, car, tree)."
    else:
        ctx = "relationship predicate in a scene graph (e.g. on, wearing, next-to)."
    return f"Are \"{term_gt}\" and \"{term_pred}\" synonymous in the context of {ctx} Answer with only 1 or 0."


def _build_synonym_prompt_with_full_scene(
    term_gt: str,
    term_pred: str,
    gt_summary: str,
    pred_summary: str,
) -> str:
    """Промпт для судьи объектов с полными разметками GT и Pred (единый подход с судьёй отношений)."""
    return (
        "You are given the full ground-truth and predicted scene graphs for one image. "
        "Based on this full context, decide if the two object/category labels below refer to the same entity.\n\n"
        f"Full GT scene graph: {gt_summary}\n\n"
        f"Full predicted scene graph: {pred_summary}\n\n"
        f"Do the two labels {term_gt!r} (from GT) and {term_pred!r} (from prediction) refer to the same entity in this scene? "
        "Answer only 1 or 0."
    )


def _parse_synonym_answer(text: str) -> int:
    """Из ответа Qwen извлекаем 1 или 0."""
    text = (text or "").strip()
    if re.search(r"\b1\b", text) and not re.search(r"\b0\b", text):
        return 1
    if re.search(r"\b0\b", text):
        return 0
    if re.search(r"\b1\b", text):
        return 1
    return 0


class QwenSynonymJudge:
    """Вызов Qwen только для решения: два термина синонимы (1) или нет (0)."""

    def __init__(self, model_name: str = QWEN_MODEL_NAME, max_model_len: int = 4096):
        # Локальный каталог (с config.json) — без запросов к HF API; hub id всегда может дернуть api.huggingface.co.
        _local = bool(model_name) and os.path.isdir(os.path.expanduser(model_name))
        _offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
        print("[judge] Loading Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=_local or _offline,
        )
        print("[judge] Initializing Qwen (vLLM) for synonym checks...")
        gpu_mem_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.40"))
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
        )
        print("[judge] Qwen synonym judge ready.")

    def batch_synonym_check(
        self,
        pairs: List[Tuple[str, str]],
        context: str = "object",
        max_new_tokens: int = 8,
        scene_contexts: Optional[List[Tuple[str, str]]] = None,
    ) -> List[int]:
        """
        Для каждой пары (term_gt, term_pred) возвращает 1 (синонимы) или 0.
        context: "object" или "relation" (используется только при отсутствии scene_contexts).
        scene_contexts: если задан (длина = len(pairs)), каждый элемент (gt_summary, pred_summary) —
          полная разметка GT и предсказания; судья решает по контексту сцены (единый подход с судьёй отношений).
        """
        if not pairs:
            return []
        if scene_contexts is not None and len(scene_contexts) != len(pairs):
            scene_contexts = None
        prompts = []
        for i, (term_gt, term_pred) in enumerate(pairs):
            if scene_contexts and i < len(scene_contexts) and scene_contexts[i] is not None:
                gt_sum, pred_sum = scene_contexts[i]
                user = _build_synonym_prompt_with_full_scene(term_gt, term_pred, gt_sum, pred_sum)
            else:
                user = _build_synonym_prompt(term_gt, term_pred, context)
            messages = [
                {"role": "system", "content": SYSTEM_SYNONYM},
                {"role": "user", "content": user},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        return [_parse_synonym_answer(o.outputs[0].text) for o in outputs]

    def batch_relation_triplet_check(
        self,
        pairs: List[Tuple[Tuple[str, str, str], Tuple[str, str, str]]],
        max_new_tokens: int = 16,
        scene_contexts: Optional[List[Tuple[str, str]]] = None,
    ) -> List[int]:
        """
        Для каждой пары триплетов (gt_triplet, pred_triplet) возвращает 1 (эквивалентны) или 0.
        scene_contexts: если задан (длина = len(pairs)), каждый элемент (gt_summary, pred_summary) —
          полная разметка GT и предсказания для контекста; судья решает на основе общего понимания сцены.
        """
        if not pairs:
            return []
        if scene_contexts is not None and len(scene_contexts) != len(pairs):
            scene_contexts = None
        prompts = []
        for i, ((s1, p1, o1), (s2, p2, o2)) in enumerate(pairs):
            if scene_contexts and i < len(scene_contexts) and scene_contexts[i] is not None:
                gt_sum, pred_sum = scene_contexts[i]
                user = (
                    "You are given the full ground-truth and predicted scene graphs for one image. "
                    "Based on this full context, decide if the specific relation pair below are semantically equivalent.\n\n"
                    f"Full GT scene graph: {gt_sum}\n\n"
                    f"Full predicted scene graph: {pred_sum}\n\n"
                    "Now consider this specific relation pair (same entities, matched by bounding box):\n"
                    f"GT relation: subject={s1!r}, predicate={p1!r}, object={o1!r}.\n"
                    f"Pred relation: subject={s2!r}, predicate={p2!r}, object={o2!r}.\n"
                    "Are these two relations semantically equivalent in the context of the scene above? Answer only 1 or 0."
                )
            else:
                user = (
                    "Subject and object in both triplets refer to the same entities (already matched). "
                    "Judge only whether the predicates are semantically equivalent in the scene graph context.\n"
                    f"Triplet 1: subject={s1!r}, predicate={p1!r}, object={o1!r}.\n"
                    f"Triplet 2: subject={s2!r}, predicate={p2!r}, object={o2!r}.\n"
                    "Answer only 1 or 0."
                )
            messages = [
                {"role": "system", "content": SYSTEM_TRIPLET},
                {"role": "user", "content": user},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        return [_parse_synonym_answer(o.outputs[0].text) for o in outputs]

# ------------------------ ФОРМАТ СЦЕНЫ ДЛЯ КОНТЕКСТА СУДЬИ ------------------------ #

def format_scene_summary(scene: Dict) -> str:
    """Краткое текстовое описание сцены (объекты + отношения) для контекста судьи."""
    objects = scene.get("objects", [])
    rels = scene.get("relationships", [])
    id_to_name = {o["id"]: (o.get("name") or "") for o in objects}
    obj_str = "; ".join(f"{o['id']}:{id_to_name.get(o['id'], '')}" for o in objects)
    rel_str = "; ".join(
        f"({id_to_name.get(r['s_id'], '')}, {r.get('pred', '')}, {id_to_name.get(r['o_id'], '')})"
        for r in rels
    )
    return f"Objects: {obj_str}. Relations: {rel_str}"


# ------------------------ НОРМАЛИЗАЦИЯ ТРИПЛЕТОВ ДЛЯ СУДЬИ ОТНОШЕНИЙ ------------------------ #

def _is_object_synonym(name_gt: str, name_pred: str, obj_result_map: Dict[Tuple[str, str], int]) -> bool:
    """Считаем синонимами при точном совпадении или при ответе Qwen 1 для пары (gt, pred)."""
    if name_gt == name_pred:
        return True
    return obj_result_map.get((name_gt, name_pred), 0) == 1


def normalize_relation_triplet_pair(
    t_gt: Tuple[str, str, str],
    t_pred: Tuple[str, str, str],
    obj_result_map: Dict[Tuple[str, str], int],
) -> Tuple[Tuple[str, str, str], Tuple[str, str, str], Dict]:
    """
    Нормализует пару триплетов для судьи отношений: подставляет имена объектов из GT,
    если они синонимы с pred, чтобы судья сравнивал только предикаты.
    Returns:
        (norm_gt, norm_pred): триплеты для подачи в batch_relation_triplet_check.
        info: {"obj_subj": (s_gt, s_p, is_syn), "obj_obj": (o_gt, o_p, is_syn)}.
    """
    s_gt, gp, o_gt = t_gt
    s_p, pp, o_p = t_pred
    is_syn_subj = _is_object_synonym(s_gt, s_p, obj_result_map)
    is_syn_obj = _is_object_synonym(o_gt, o_p, obj_result_map)
    norm_gt = (s_gt, gp, o_gt)
    norm_pred = (
        s_gt if is_syn_subj else s_p,
        pp,
        o_gt if is_syn_obj else o_p,
    )
    info = {
        "obj_subj": (s_gt, s_p, is_syn_subj),
        "obj_obj": (o_gt, o_p, is_syn_obj),
    }
    return norm_gt, norm_pred, info


# ------------------------ МЕТРИКИ ПО ФОРМУЛАМ ------------------------ #

def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2/(1/P + 1/R)."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 / (1.0 / p + 1.0 / r) if (p + r) > 0 else 0.0
    return (p, r, f1)


def compute_sample_metrics_data(
    gt_scene: Dict,
    pred_scene: Dict,
    iou_thr: float,
) -> Dict:
    """
    По одному сэмплу: матчинг венгерский, сбор пар для споров (объекты и предикаты).
    Возвращает структуру для последующего вызова Qwen и подсчёта TP/FP/FN по формулам.

    Порог iou_thr (обычно 0.5) — бинарный фильтр: IoU >= thr → матч, иначе нет.
    bbox_mean_iou_matched — среднее IoU пар, прошедших порог.
    """
    gt_objs = gt_scene.get("objects", [])
    pred_objs = pred_scene.get("objects", [])
    gt_rels = gt_scene.get("relationships", [])
    pred_rels = pred_scene.get("relationships", [])

    matches, iou_matrix = hungarian_iou_match(gt_objs, pred_objs, iou_thr)
    n_gt = len(gt_objs)
    n_pred = len(pred_objs)
    iou_matched = [m[2] for m in matches]
    mean_iou = float(np.mean(iou_matched)) if iou_matched else 0.0
    object_matches_with_iou = [(m[0], m[1], float(m[2])) for m in matches]

    pred_id_to_gt_id: Dict[int, int] = {}
    for gi, pj, _ in matches:
        pred_id_to_gt_id[pred_objs[pj]["id"]] = gt_objs[gi]["id"]

    # Объекты: для каждого матча либо exact, либо пара для Qwen
    object_exact_or_dispute: List[Optional[Tuple[str, str]]] = []
    for gi, pj, _ in matches:
        gn = (gt_objs[gi]["name"] or "").strip()
        pn = (pred_objs[pj]["name"] or "").strip()
        if gn == pn:
            object_exact_or_dispute.append(None)  # exact
        else:
            object_exact_or_dispute.append((gn, pn))

    # Кандидаты отношений: (gt_ri, pred_rj) где (pred_s, pred_o) -> (gt_s, gt_o)
    relation_candidates: List[Tuple[int, int, str, str]] = []  # gt_ri, pred_rj, gt_pred, pred_pred
    for ri, r_gt in enumerate(gt_rels):
        gs, gp, go = r_gt["s_id"], r_gt["pred"], r_gt["o_id"]
        for rj, r_p in enumerate(pred_rels):
            ps, pp, po = r_p["s_id"], r_p["pred"], r_p["o_id"]
            if pred_id_to_gt_id.get(ps) == gs and pred_id_to_gt_id.get(po) == go:
                relation_candidates.append((ri, rj, gp or "", pp or ""))

    n_gt_rel = len(gt_rels)
    n_pred_rel = len(pred_rels)
    object_matches = [(m[0], m[1]) for m in matches]  # (gt_idx, pred_idx) для построения TP/FP/FN
    iou_matrix_list = iou_matrix.tolist()  # n_gt x n_pred для детальной статистики

    return {
        "bbox_num_objects_gt": n_gt,
        "bbox_num_objects_pred": n_pred,
        "bbox_num_objects_matched": len(matches),
        "bbox_mean_iou_matched": mean_iou,
        "object_exact_or_dispute": object_exact_or_dispute,
        "object_matches": object_matches,
        "object_matches_with_iou": object_matches_with_iou,
        "iou_matrix": iou_matrix_list,
        "relation_candidates": relation_candidates,
        "n_gt_rel": n_gt_rel,
        "n_pred_rel": n_pred_rel,
    }


def apply_synonym_results_and_compute_metrics(
    data: Dict,
    object_synonym_results: List[int],
    relation_synonym_results: List[int],
) -> Dict:
    """
    После вызова Qwen: подставляем 0/1 для споров и считаем TP/FP/FN, затем P/R/F1.
    object_synonym_results: 1 для синонимов по объектам (порядок как в data["object_exact_or_dispute"] где не None).
    relation_synonym_results: 1 для синонимов по предикатам (порядок по data["relation_candidates"] где не exact).
    """
    obj_exact_or_dispute = data["object_exact_or_dispute"]
    rel_candidates = data["relation_candidates"]
    n_gt = data["bbox_num_objects_gt"]
    n_pred = data["bbox_num_objects_pred"]
    n_gt_rel = data["n_gt_rel"]
    n_pred_rel = data["n_pred_rel"]

    obj_idx = 0
    name_ok: List[bool] = []
    for x in obj_exact_or_dispute:
        if x is None:
            name_ok.append(True)
        else:
            name_ok.append(object_synonym_results[obj_idx] == 1)
            obj_idx += 1
    assert obj_idx == len(object_synonym_results)

    tp_obj = sum(name_ok)
    fp_obj = n_pred - tp_obj
    fn_obj = n_gt - tp_obj
    p_obj, r_obj, f1_obj = _precision_recall_f1(tp_obj, fp_obj, fn_obj)

    # Предикаты: для каждого кандидата exact или результат Qwen
    rel_idx = 0
    pred_ok: List[bool] = []
    for (_, _, gp, pp) in rel_candidates:
        if gp == pp:
            pred_ok.append(True)
        else:
            pred_ok.append(relation_synonym_results[rel_idx] == 1)
            rel_idx += 1
    assert rel_idx == len(relation_synonym_results)

    # Матрица: строки = GT relations, столбцы = pred relations. Ребро (i,j) если кандидат (i,j) и pred_ok.
    # relation_candidates уже список (gt_ri, pred_rj, ...). Строим матрицу и венгерский матчинг.
    n_gt_r = n_gt_rel
    n_pred_r = n_pred_rel
    cost = np.ones((n_gt_r, n_pred_r))  # 0 = edge, 1 = no edge; minimize cost = max matches
    for k, (ri, rj, _, _) in enumerate(rel_candidates):
        if pred_ok[k]:
            cost[ri, rj] = 0
    row_ind, col_ind = linear_sum_assignment(cost)
    tp_rel = sum(1 for i, j in zip(row_ind, col_ind) if cost[i, j] == 0)
    fp_rel = n_pred_r - tp_rel
    fn_rel = n_gt_r - tp_rel
    p_rel, r_rel, f1_rel = _precision_recall_f1(tp_rel, fp_rel, fn_rel)

    sgg_score = (f1_obj + f1_rel) / 2.0  # среднее арифметическое F1 объектов и F1 отношений
    return {
        "bbox_num_objects_gt": data["bbox_num_objects_gt"],
        "bbox_num_objects_pred": data["bbox_num_objects_pred"],
        "bbox_num_objects_matched": data["bbox_num_objects_matched"],
        "bbox_num_objects_unmatched_gt": n_gt - data["bbox_num_objects_matched"],
        "bbox_num_objects_unmatched_pred": n_pred - data["bbox_num_objects_matched"],
        "bbox_mean_iou_matched": data["bbox_mean_iou_matched"],
        "precision_objects": p_obj,
        "recall_objects": r_obj,
        "f1_objects": f1_obj,
        "precision_relations": p_rel,
        "recall_relations": r_rel,
        "f1_relations": f1_rel,
        "sgg_score": sgg_score,
    }


def compute_sample_metrics_detailed(
    gt_scene: Dict,
    pred_scene: Dict,
    data: Dict,
    object_synonym_results: List[int],
    relation_synonym_results: List[int],
) -> Tuple[Dict, Dict, Dict]:
    """
    Считает метрики строгие (без синонимов) и с Qwen, возвращает детальные списки для отчёта.
    Returns:
        metrics_strict: P/R/F1 и sgg_score при строгом сравнении (все споры = 0).
        metrics_qwen: P/R/F1 и sgg_score с учётом ответов Qwen.
        details: dict с ключами:
          - gt_objects, pred_objects: списки {id, name, bbox}
          - gt_triplets, pred_triplets: списки (subj_name, pred, obj_name)
          - tp_objects_strict, fp_objects_strict, fn_objects_strict (списки объектов)
          - tp_objects_qwen, fp_objects_qwen, fn_objects_qwen
          - tp_relations_strict, fp_relations_strict, fn_relations_strict (списки триплетов)
          - tp_relations_qwen, fp_relations_qwen, fn_relations_qwen
          - object_disputes_with_answers: [(gt_name, pred_name, 0|1), ...]
          - relation_disputes_with_answers: список dict с ключами gt, pred, answer, obj_subj, obj_obj, normalized_gt, normalized_pred
            (для вывода: спорная ситуация, решения Qwen по объектам, триплеты для судьи)
    """
    gt_objs = gt_scene.get("objects", [])
    pred_objs = pred_scene.get("objects", [])
    gt_rels = gt_scene.get("relationships", [])
    pred_rels = pred_scene.get("relationships", [])
    obj_exact = data["object_exact_or_dispute"]
    obj_matches = data["object_matches"]
    rel_candidates = data["relation_candidates"]
    n_gt, n_pred = data["bbox_num_objects_gt"], data["bbox_num_objects_pred"]
    n_gt_r, n_pred_r = data["n_gt_rel"], data["n_pred_rel"]

    id_to_name_gt = {o["id"]: (o.get("name") or "") for o in gt_objs}
    id_to_name_pred = {o["id"]: (o.get("name") or "") for o in pred_objs}

    def _name_ok(syn_results: List[int]) -> List[bool]:
        idx = 0
        out = []
        for x in obj_exact:
            if x is None:
                out.append(True)
            else:
                out.append(syn_results[idx] == 1)
                idx += 1
        return out

    def _pred_ok(syn_results: List[int]) -> List[bool]:
        idx = 0
        out = []
        for (_, _, gp, pp) in rel_candidates:
            if gp == pp:
                out.append(True)
            else:
                out.append(syn_results[idx] == 1)
                idx += 1
        return out

    obj_strict = [0] * sum(1 for x in obj_exact if x is not None)
    rel_strict = [0] * sum(1 for (_, _, gp, pp) in rel_candidates if gp != pp)
    name_ok_strict = _name_ok(obj_strict)
    name_ok_qwen = _name_ok(object_synonym_results)
    pred_ok_strict = _pred_ok(rel_strict)
    pred_ok_qwen = _pred_ok(relation_synonym_results)

    def _tp_fp_fn_objects(name_ok_list: List[bool]):
        tp_set = set()
        for k, ok in enumerate(name_ok_list):
            if ok and k < len(obj_matches):
                gi, pj = obj_matches[k]
                tp_set.add((gi, pj))
        tp_list = [{"gt": gt_objs[gi], "pred": pred_objs[pj]} for (gi, pj) in tp_set]
        matched_pred = {pj for (_, pj) in tp_set}
        matched_gt = {gi for (gi, _) in tp_set}
        fp_list = [pred_objs[pj] for pj in range(n_pred) if pj not in matched_pred]
        fn_list = [gt_objs[gi] for gi in range(n_gt) if gi not in matched_gt]
        return tp_list, fp_list, fn_list

    def _rel_matching(pred_ok_list: List[bool]):
        cost = np.ones((n_gt_r, n_pred_r))
        for k, (ri, rj, _, _) in enumerate(rel_candidates):
            if pred_ok_list[k]:
                cost[ri, rj] = 0
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = [(row_ind[i], col_ind[i]) for i in range(len(row_ind)) if cost[row_ind[i], col_ind[i]] == 0]
        return matched

    def _tp_fp_fn_relations(matched_pairs: List[Tuple[int, int]]):
        matched_gt_r = {ri for (ri, _) in matched_pairs}
        matched_pred_r = {rj for (_, rj) in matched_pairs}
        tp_list = []
        for ri, rj in matched_pairs:
            r_gt = gt_rels[ri]
            r_p = pred_rels[rj]
            s_gt = id_to_name_gt.get(r_gt["s_id"], "")
            o_gt = id_to_name_gt.get(r_gt["o_id"], "")
            s_p = id_to_name_pred.get(r_p["s_id"], "")
            o_p = id_to_name_pred.get(r_p["o_id"], "")
            tp_list.append({"gt": (s_gt, r_gt["pred"], o_gt), "pred": (s_p, r_p["pred"], o_p)})
        fp_list = [(id_to_name_pred.get(pred_rels[rj]["s_id"], ""), pred_rels[rj]["pred"], id_to_name_pred.get(pred_rels[rj]["o_id"], "")) for rj in range(n_pred_r) if rj not in matched_pred_r]
        fn_list = [(id_to_name_gt.get(gt_rels[ri]["s_id"], ""), gt_rels[ri]["pred"], id_to_name_gt.get(gt_rels[ri]["o_id"], "")) for ri in range(n_gt_r) if ri not in matched_gt_r]
        return tp_list, fp_list, fn_list

    tp_o_s, fp_o_s, fn_o_s = _tp_fp_fn_objects(name_ok_strict)
    tp_o_q, fp_o_q, fn_o_q = _tp_fp_fn_objects(name_ok_qwen)
    matched_rel_s = _rel_matching(pred_ok_strict)
    matched_rel_q = _rel_matching(pred_ok_qwen)
    tp_r_s, fp_r_s, fn_r_s = _tp_fp_fn_relations(matched_rel_s)
    tp_r_q, fp_r_q, fn_r_q = _tp_fp_fn_relations(matched_rel_q)

    metrics_strict = apply_synonym_results_and_compute_metrics(data, obj_strict, rel_strict)
    metrics_qwen = apply_synonym_results_and_compute_metrics(data, object_synonym_results, relation_synonym_results)

    obj_dispute_answers = []
    o_idx = 0
    for x in obj_exact:
        if x is not None:
            obj_dispute_answers.append((x[0], x[1], object_synonym_results[o_idx]))
            o_idx += 1
    obj_result_map = {}
    o_idx = 0
    for x in obj_exact:
        if x is not None:
            obj_result_map[tuple(x)] = object_synonym_results[o_idx]
            o_idx += 1
    rel_dispute_answers = []
    r_idx = 0
    for (ri, rj, gp, pp) in rel_candidates:
        if gp != pp:
            r_gt = gt_rels[ri]
            r_p = pred_rels[rj]
            t_gt = (id_to_name_gt.get(r_gt["s_id"], ""), gp, id_to_name_gt.get(r_gt["o_id"], ""))
            t_pred = (id_to_name_pred.get(r_p["s_id"], ""), pp, id_to_name_pred.get(r_p["o_id"], ""))
            norm_gt, norm_pred, norm_info = normalize_relation_triplet_pair(t_gt, t_pred, obj_result_map)
            rel_dispute_answers.append({
                "gt": t_gt,
                "pred": t_pred,
                "answer": relation_synonym_results[r_idx],
                "obj_subj": norm_info["obj_subj"],
                "obj_obj": norm_info["obj_obj"],
                "normalized_gt": norm_gt,
                "normalized_pred": norm_pred,
            })
            r_idx += 1

    gt_triplets = [(id_to_name_gt.get(r["s_id"], ""), r["pred"], id_to_name_gt.get(r["o_id"], "")) for r in gt_rels]
    pred_triplets = [(id_to_name_pred.get(r["s_id"], ""), r["pred"], id_to_name_pred.get(r["o_id"], "")) for r in pred_rels]

    object_matches_with_iou = data.get("object_matches_with_iou", [])
    iou_matched_pairs = []
    for k, (gi, pj, iou) in enumerate(object_matches_with_iou):
        iou_matched_pairs.append({
            "gt": gt_objs[gi],
            "pred": pred_objs[pj],
            "iou": round(iou, 4),
            "in_tp_strict": name_ok_strict[k] if k < len(name_ok_strict) else False,
            "in_tp_qwen": name_ok_qwen[k] if k < len(name_ok_qwen) else False,
        })
    iou_matrix = data.get("iou_matrix", [])

    details = {
        "gt_objects": [{"id": o["id"], "name": o.get("name", ""), "bbox": o["bbox"]} for o in gt_objs],
        "pred_objects": [{"id": o["id"], "name": o.get("name", ""), "bbox": o["bbox"]} for o in pred_objs],
        "gt_triplets": gt_triplets,
        "pred_triplets": pred_triplets,
        "iou_matrix": iou_matrix,
        "iou_matched_pairs": iou_matched_pairs,
        "tp_objects_strict": tp_o_s,
        "fp_objects_strict": fp_o_s,
        "fn_objects_strict": fn_o_s,
        "tp_objects_qwen": tp_o_q,
        "fp_objects_qwen": fp_o_q,
        "fn_objects_qwen": fn_o_q,
        "tp_relations_strict": tp_r_s,
        "fp_relations_strict": fp_r_s,
        "fn_relations_strict": fn_r_s,
        "tp_relations_qwen": tp_r_q,
        "fp_relations_qwen": fp_r_q,
        "fn_relations_qwen": fn_r_q,
        "object_disputes_with_answers": obj_dispute_answers,
        "relation_disputes_with_answers": rel_dispute_answers,
    }
    return metrics_strict, metrics_qwen, details


# ------------------------ ВСПОМОГАТЕЛЬНОЕ ------------------------ #

def extract_gt_pred_toon(sample: Dict) -> Tuple[str, str]:
    """Supports both legacy and new compact schemas.

    Priority:
    1) content/predict
    2) conversations[gpt/predict]
    3) messages assistant as predict fallback
    """
    gt_toon = sample.get("content")
    pred_toon = sample.get("predict")

    if gt_toon is None or pred_toon is None:
        for c in sample.get("conversations", []):
            role = c.get("from")
            if gt_toon is None and role == "gpt":
                gt_toon = c.get("value")
            elif pred_toon is None and role == "predict":
                pred_toon = c.get("value")

    if pred_toon is None:
        for m in sample.get("messages", []):
            if m.get("role") == "assistant":
                pred_toon = m.get("content")
                break

    return gt_toon, pred_toon


def round3(x):
    if x is None:
        return None
    return float(f"{x:.3f}")

def is_valid_toon(toon: str) -> bool:
    """Считаем toon валидным, если он распарсился и есть хотя бы один объект."""
    scene = parse_toon_scene(toon)
    return len(scene.get("objects", [])) > 0

# ------------------------ ГЛОБАЛЬНАЯ СТАТИСТИКА ------------------------ #

def compute_global_stats(jsonl_path: str):
    gen_times = []
    metrics_list = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            t = sample.get("gen_time_sec", None)
            if isinstance(t, (int, float)) and t is not None and t > 0:
                gen_times.append(float(t))
            m = sample.get("metrics", None)
            if isinstance(m, dict):
                metrics_list.append(m)

    gen_times = np.array(gen_times, dtype=float)
    time_stats = stats_with_phys(gen_times, unit="s")

    if not metrics_list:
        return

    n_failed = sum(1 for m in metrics_list if m.get("toon_valid") is False)
    n_total = len(metrics_list)
    failure_rate_pct = 100.0 * n_failed / n_total if n_total else 0.0

    iou_thr = metrics_list[0].get("iou_thr", 0.5)
    iou_vals = [float(m["bbox_mean_iou_matched"]) for m in metrics_list
                if m.get("bbox_mean_iou_matched") is not None and m.get("bbox_num_objects_matched", 0) > 0]
    n_iou_samples = len(iou_vals)
    mean_iou_global = float(np.mean(iou_vals)) if iou_vals else float("nan")

    _table_keys = [
        ("precision_objects",   "strict_precision_objects",   "qwen_precision_objects"),
        ("recall_objects",      "strict_recall_objects",      "qwen_recall_objects"),
        ("f1_objects",          "strict_f1_objects",          "qwen_f1_objects"),
        ("precision_relations", "strict_precision_relations", "qwen_precision_relations"),
        ("recall_relations",    "strict_recall_relations",    "qwen_recall_relations"),
        ("f1_relations",        "strict_f1_relations",        "qwen_f1_relations"),
        ("sgg_score",           "strict_sgg_score",           "qwen_sgg_score"),
    ]

    def _mean(key: str) -> float:
        vals = [float(m[key]) for m in metrics_list if m.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n========== МЕТРИКИ (IoU порог = {iou_thr}) ==========\n")
    print(f"  N сэмплов: {n_total}")
    print(f"  failure_rate: {failure_rate_pct:.2f}%  ({n_failed} / {n_total})")
    print(f"  bbox_mean_iou (matched > 0): {mean_iou_global:.3f}  ({n_iou_samples} / {n_total} сэмплов)")
    if len(gen_times) > 0:
        print(f"  gen_time: {time_stats['phys']}")
    print()

    w_name = max(len(label) for label, _, _ in _table_keys) + 1
    w_col = 12
    print(f"  {'metric':<{w_name}} {'строго':>{w_col}}{'Qwen':>{w_col}}")
    print("  " + "-" * (w_name + 2 * w_col))
    for label, strict_key, qwen_key in _table_keys:
        v_strict = _mean(strict_key)
        v_qwen = _mean(qwen_key)
        s_strict = f"{v_strict:.3f}" if not np.isnan(v_strict) else "N/A"
        s_qwen = f"{v_qwen:.3f}" if not np.isnan(v_qwen) else "N/A"
        print(f"  {label:<{w_name}} {s_strict:>{w_col}}{s_qwen:>{w_col}}")
    print()

# ------------------------ MAIN ------------------------ #



def main():
    parser = argparse.ArgumentParser(
        description="Compute PSG-SGG metrics from one predictions jsonl and write one compact metrics JSON.",
    )
    parser.add_argument("--pred-jsonl", required=True, type=str, help="Path to predictions jsonl.")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for metrics JSON.")
    parser.add_argument("--output-name", type=str, default="", help="Output filename, e.g. gemini-3-flash-preview-psg-metrics.json")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--batch-size-qwen", type=int, default=32)
    parser.add_argument("--max-new-tokens-qwen", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.40, help="vLLM GPU memory utilization for Qwen judge")
    parser.add_argument("--cuda-visible-devices", type=str, default="", help="Optional CUDA_VISIBLE_DEVICES override for Qwen judge")
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default="",
        help="Optional synonym-judge model path; default env QWEN_MODEL_PATH or built-in default.",
    )
    parser.add_argument(
        "--per-sample-jsonl",
        type=str,
        default="",
        help="Optional path: write one JSONL line per input row, each line = original sample + 'metrics' dict (for video overlays / debugging).",
    )
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)

    pred_jsonl = Path(args.pred_jsonl)
    out_dir = Path(args.output_dir)
    if not pred_jsonl.exists():
        print(f"[ERROR] Input file not found: {pred_jsonl}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.output_name.strip() if args.output_name else f"{pred_jsonl.stem}-metrics.json"
    out_json = out_dir / out_name
    iou_thr = float(args.iou_thr)

    samples: List[Dict] = []
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[init] loaded samples: {len(samples)} from {pred_jsonl}")

    per_sample_data = []
    all_obj_disputes = []
    all_obj_dispute_scene_contexts = []
    all_rel_dispute_triplets = []
    gen_times = []

    for sample in tqdm(samples, desc="parse + collect disputes"):
        t = sample.get("gen_time_sec", None)
        if isinstance(t, (int, float)) and t and t > 0:
            gen_times.append(float(t))

        gt_toon, pred_toon = extract_gt_pred_toon(sample)
        if not gt_toon or not pred_toon:
            per_sample_data.append(None)
            continue
        pred_valid = bool(pred_toon) and is_valid_toon(pred_toon)
        if not pred_valid:
            per_sample_data.append(None)
            continue

        gt_scene = parse_toon_scene(gt_toon)
        pred_scene = parse_toon_scene(pred_toon)
        data = compute_sample_metrics_data(gt_scene, pred_scene, iou_thr)

        gt_summary = format_scene_summary(gt_scene)
        pred_summary = format_scene_summary(pred_scene)
        id_to_name_gt = {o["id"]: (o.get("name") or "") for o in gt_scene.get("objects", [])}
        id_to_name_pred = {o["id"]: (o.get("name") or "") for o in pred_scene.get("objects", [])}
        gt_rels = gt_scene.get("relationships", [])
        pred_rels = pred_scene.get("relationships", [])

        obj_offset = len(all_obj_disputes)
        for x in data["object_exact_or_dispute"]:
            if x is not None:
                all_obj_disputes.append(x)
                all_obj_dispute_scene_contexts.append((gt_summary, pred_summary))

        rel_offset = len(all_rel_dispute_triplets)
        for (ri, rj, gp, pp) in data["relation_candidates"]:
            if gp != pp:
                r_gt = gt_rels[ri]
                r_p = pred_rels[rj]
                t_gt = (id_to_name_gt.get(r_gt["s_id"], ""), gp, id_to_name_gt.get(r_gt["o_id"], ""))
                t_pred = (id_to_name_pred.get(r_p["s_id"], ""), pp, id_to_name_pred.get(r_p["o_id"], ""))
                all_rel_dispute_triplets.append((t_gt, t_pred, gt_summary, pred_summary))

        data["_obj_dispute_offset"] = obj_offset
        data["_obj_dispute_len"] = len(all_obj_disputes) - obj_offset
        data["_rel_dispute_offset"] = rel_offset
        data["_rel_dispute_len"] = len(all_rel_dispute_triplets) - rel_offset
        per_sample_data.append(data)

    judge = None
    obj_syn_results: List[int] = []
    rel_syn_results: List[int] = []
    if all_obj_disputes or all_rel_dispute_triplets:
        qwen_model_path = args.qwen_model_path or os.getenv("QWEN_MODEL_PATH", QWEN_MODEL_NAME)
        print(f"[init] Loading Qwen synonym judge: {qwen_model_path}")
        judge = QwenSynonymJudge(model_name=qwen_model_path)
        batch_q = args.batch_size_qwen
        for start in tqdm(range(0, len(all_obj_disputes), batch_q), desc="qwen object disputes"):
            chunk = all_obj_disputes[start : start + batch_q]
            ctx_chunk = all_obj_dispute_scene_contexts[start : start + batch_q]
            obj_syn_results.extend(
                judge.batch_synonym_check(chunk, context="object", max_new_tokens=8, scene_contexts=ctx_chunk)
            )

        obj_result_map = {tuple(pair): r for pair, r in zip(all_obj_disputes, obj_syn_results)}
        normalized_rel_pairs = [
            normalize_relation_triplet_pair(x[0], x[1], obj_result_map)[:2]
            for x in all_rel_dispute_triplets
        ]
        rel_scene_contexts = [(x[2], x[3]) for x in all_rel_dispute_triplets]
        for start in tqdm(range(0, len(normalized_rel_pairs), batch_q), desc="qwen relation disputes"):
            chunk = normalized_rel_pairs[start : start + batch_q]
            ctx_chunk = rel_scene_contexts[start : start + batch_q]
            rel_syn_results.extend(
                judge.batch_relation_triplet_check(
                    chunk, max_new_tokens=args.max_new_tokens_qwen, scene_contexts=ctx_chunk
                )
            )

    metric_rows = []
    for data in per_sample_data:
        if data is None:
            metric_rows.append(
                {
                    "bbox_mean_iou_matched": 0.0,
                    "strict_precision_objects": 0.0,
                    "strict_recall_objects": 0.0,
                    "strict_f1_objects": 0.0,
                    "strict_precision_relations": 0.0,
                    "strict_recall_relations": 0.0,
                    "strict_f1_relations": 0.0,
                    "strict_sgg_score": 0.0,
                    "qwen_precision_objects": 0.0,
                    "qwen_recall_objects": 0.0,
                    "qwen_f1_objects": 0.0,
                    "qwen_precision_relations": 0.0,
                    "qwen_recall_relations": 0.0,
                    "qwen_f1_relations": 0.0,
                    "qwen_sgg_score": 0.0,
                    "toon_valid": False,
                }
            )
            continue

        o_start = data["_obj_dispute_offset"]
        o_len = data["_obj_dispute_len"]
        r_start = data["_rel_dispute_offset"]
        r_len = data["_rel_dispute_len"]
        obj_res = obj_syn_results[o_start : o_start + o_len]
        rel_res = rel_syn_results[r_start : r_start + r_len]
        zeros_obj = [0] * o_len
        zeros_rel = [0] * r_len
        m_strict = apply_synonym_results_and_compute_metrics(data, zeros_obj, zeros_rel)
        m_qwen = apply_synonym_results_and_compute_metrics(data, obj_res, rel_res)
        metric_rows.append(
            {
                "bbox_mean_iou_matched": float(m_qwen["bbox_mean_iou_matched"]),
                "strict_precision_objects": float(m_strict["precision_objects"]),
                "strict_recall_objects": float(m_strict["recall_objects"]),
                "strict_f1_objects": float(m_strict["f1_objects"]),
                "strict_precision_relations": float(m_strict["precision_relations"]),
                "strict_recall_relations": float(m_strict["recall_relations"]),
                "strict_f1_relations": float(m_strict["f1_relations"]),
                "strict_sgg_score": float(m_strict["sgg_score"]),
                "qwen_precision_objects": float(m_qwen["precision_objects"]),
                "qwen_recall_objects": float(m_qwen["recall_objects"]),
                "qwen_f1_objects": float(m_qwen["f1_objects"]),
                "qwen_precision_relations": float(m_qwen["precision_relations"]),
                "qwen_recall_relations": float(m_qwen["recall_relations"]),
                "qwen_f1_relations": float(m_qwen["f1_relations"]),
                "qwen_sgg_score": float(m_qwen["sgg_score"]),
                "toon_valid": True,
            }
        )

    def _mean(key: str) -> float:
        vals = [float(m[key]) for m in metric_rows if m.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    per_sample_path = (args.per_sample_jsonl or "").strip()
    if per_sample_path:
        psp = Path(per_sample_path)
        psp.parent.mkdir(parents=True, exist_ok=True)

        def _metrics_for_ui(mrow: dict) -> dict:
            return {
                "iou_thr": iou_thr,
                "bbox_mean_iou_matched": mrow.get("bbox_mean_iou_matched"),
                "qwen_precision_objects": mrow.get("qwen_precision_objects"),
                "qwen_recall_objects": mrow.get("qwen_recall_objects"),
                "qwen_f1_objects": mrow.get("qwen_f1_objects"),
                "qwen_precision_relations": mrow.get("qwen_precision_relations"),
                "qwen_recall_relations": mrow.get("qwen_recall_relations"),
                "qwen_f1_relations": mrow.get("qwen_f1_relations"),
                "qwen_sgg_score": mrow.get("qwen_sgg_score"),
                "strict_precision_objects": mrow.get("strict_precision_objects"),
                "strict_recall_objects": mrow.get("strict_recall_objects"),
                "strict_f1_objects": mrow.get("strict_f1_objects"),
                "strict_precision_relations": mrow.get("strict_precision_relations"),
                "strict_recall_relations": mrow.get("strict_recall_relations"),
                "strict_f1_relations": mrow.get("strict_f1_relations"),
                "strict_sgg_score": mrow.get("strict_sgg_score"),
                "toon_valid": mrow.get("toon_valid"),
            }

        with open(psp, "w", encoding="utf-8") as f:
            for sample, mrow in zip(samples, metric_rows):
                out = dict(sample)
                out["metrics"] = _metrics_for_ui(mrow)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[done] per-sample metrics jsonl -> {psp}")

    n_total = len(metric_rows)
    n_valid = sum(1 for m in metric_rows if m.get("toon_valid"))
    n_invalid = n_total - n_valid
    time_stats = stats_with_phys(np.array(gen_times, dtype=float), unit="s")

    summary = {
        "predictions_file": str(pred_jsonl),
        "num_samples": n_total,
        "num_valid_toon": n_valid,
        "num_invalid_toon": n_invalid,
        "invalid_rate_pct": (100.0 * n_invalid / n_total) if n_total else 0.0,
        "iou_thr": iou_thr,
        "time_sec": time_stats,
        "Obj_AP@50_strict": _mean("strict_precision_objects"),
        "Obj_AP@50_Qwen": _mean("qwen_precision_objects"),
        "Obj_Recall_strict": _mean("strict_recall_objects"),
        "Obj_Recall_Qwen": _mean("qwen_recall_objects"),
        "Obj_F1_strict": _mean("strict_f1_objects"),
        "Obj_F1_qwen": _mean("qwen_f1_objects"),
        "Rel_AP@50_strict": _mean("strict_precision_relations"),
        "Rel_AP@50_Qwen": _mean("qwen_precision_relations"),
        "Rel_Recall_strict": _mean("strict_recall_relations"),
        "Rel_Recall_Qwen": _mean("qwen_recall_relations"),
        "Rel_F1_strict": _mean("strict_f1_relations"),
        "Rel_F1_qwen": _mean("qwen_f1_relations"),
        "SGG_Score_strict": _mean("strict_sgg_score"),
        "SGG_Score_qwen": _mean("qwen_sgg_score"),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Metrics Summary ===")
    print(f"{'Metric':<24} {'Value':>10}")
    print("-" * 36)
    for k in [
        "Obj_AP@50_strict", "Obj_AP@50_Qwen", "Obj_Recall_strict", "Obj_Recall_Qwen", "Obj_F1_strict", "Obj_F1_qwen",
        "Rel_AP@50_strict", "Rel_AP@50_Qwen", "Rel_Recall_strict", "Rel_Recall_Qwen",         "Rel_F1_strict", "Rel_F1_qwen",
        "SGG_Score_strict",
        "SGG_Score_qwen",
    ]:
        print(f"{k:<24} {summary[k]:>10.4f}")
    print(f"{'Failure_Rate_%':<24} {summary['invalid_rate_pct']:>10.2f}")
    print(f"[done] metrics json -> {out_json}")


if __name__ == "__main__":
    main()
