#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полная статистика key-frame selection: таблицы до/после по train и test,
по кадрам, классам объектов и типам отношений. Пересечение train–test.

Запуск из корня репозитория (подпись в таблицах — --label):

  Key-frame:
    ... --train-focus .../pvsg_sgg_prev_train_focus_graph.jsonl --test-focus .../pvsg_sgg_prev_test_focus_graph.jsonl

  MaxInfo (таблички с заголовком «MaxInfo»):
    ... --label MaxInfo --train-focus .../pvsg_sgg_train_maxinfo_prev.jsonl --test-focus .../pvsg_sgg_test_maxinfo_prev.jsonl

  OUR (filtered_our):
    ... --label OUR --train-focus .../pvsg_sgg_prev_filtered_our_train.jsonl --test-focus .../pvsg_sgg_prev_filtered_our_test.jsonl

  Опция --csv выводит таблицы в CSV (разделитель ;).

  Три команды (из корня репозитория) для каждого метода:

  # 1) Key-frame selection
  python3 key-frame-selection/scripts/focus_graph_full_stats.py --train-original data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_train.jsonl --train-focus data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_train_focus_graph.jsonl --test-original data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_test.jsonl --test-focus data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_test_focus_graph.jsonl

  # 2) MaxInfo
  python3 key-frame-selection/scripts/focus_graph_full_stats.py --label MaxInfo --train-original data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_train.jsonl --train-focus data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_train_maxinfo_prev.jsonl --test-original data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_test.jsonl --test-focus data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_test_maxinfo_prev.jsonl

  # 3) OUR (filtered_our)
  python3 key-frame-selection/scripts/focus_graph_full_stats.py --label OUR --train-original data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_train.jsonl --train-focus data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_filtered_our_train.jsonl --test-original data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_test.jsonl --test-focus data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_filtered_our_test.jsonl
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

VIDEO_ID_PATTERN = re.compile(r"frames/([^/]+)/\d+\.(?:png|jpg|jpeg)", re.IGNORECASE)


def get_video_id(rec: dict):
    vid = rec.get("video_id")
    if vid:
        return vid
    m = VIDEO_ID_PATTERN.search(rec.get("image", ""))
    return m.group(1) if m else None


def load_jsonl(path: str) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def group_records_by_video(records: list) -> dict:
    video_to_records = defaultdict(list)
    video_order = []
    for rec in records:
        vid = get_video_id(rec)
        if vid is None:
            continue
        if vid not in video_to_records:
            video_order.append(vid)
        video_to_records[vid].append(rec)
    return {vid: video_to_records[vid] for vid in video_order}


def parse_scene_graph_from_record(rec: dict):
    """Уникальные имена объектов и тройки (subj, pred, obj) из gpt value."""
    objects = set()
    relations = set()
    for conv in rec.get("conversations", []):
        if conv.get("from") != "gpt":
            continue
        text = conv.get("value", "")
        in_obj = False
        in_rel = False
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if line.startswith("obj[") and "{" in line:
                in_obj = True
                in_rel = False
                continue
            if line.startswith("rel[") and "{" in line:
                in_rel = True
                in_obj = False
                continue
            if in_obj and line and line[0].isdigit():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    objects.add(parts[1])
            if in_rel and line and line[0].isdigit():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    relations.add((parts[0], parts[1], parts[2]))
    return objects, relations


def preds_from_triples(rels: set) -> set:
    return {p for (_s, p, _o) in rels}


def aggregate_scene_graph_sets(records: list):
    all_objs = set()
    all_rels = set()
    for rec in records:
        objs, rels = parse_scene_graph_from_record(rec)
        all_objs |= objs
        all_rels |= rels
    return all_objs, all_rels


def object_predicate_frequencies(records: list):
    """В скольких кадрах (записях) встречается каждая категория объектов и каждый предикат."""
    obj_count = defaultdict(int)
    pred_count = defaultdict(int)
    for rec in records:
        objs, rels = parse_scene_graph_from_record(rec)
        for o in objs:
            obj_count[o] += 1
        for (_s, p, _o) in rels:
            pred_count[p] += 1
    return dict(obj_count), dict(pred_count)


def compute_split_stats(name: str, original_path: str, focus_path: str) -> dict | None:
    if not Path(original_path).exists() or not Path(focus_path).exists():
        return None
    orig = load_jsonl(original_path)
    focus = load_jsonl(focus_path)
    orig_groups = group_records_by_video(orig)
    focus_groups = group_records_by_video(focus)

    frames_before = len(orig)
    frames_after = len(focus)
    pct_frames = (100.0 * frames_after / frames_before) if frames_before else 0.0

    objs_before_set, rels_before_set = aggregate_scene_graph_sets(orig)
    objs_after_set, rels_after_set = aggregate_scene_graph_sets(focus)
    pred_before_set = preds_from_triples(rels_before_set)
    pred_after_set = preds_from_triples(rels_after_set)
    obj_freq_before, pred_freq_before = object_predicate_frequencies(orig)

    objs_before = len(objs_before_set)
    objs_after = len(objs_after_set)
    pred_before = len(pred_before_set)
    pred_after = len(pred_after_set)
    pct_objs = (100.0 * objs_after / objs_before) if objs_before else 0.0
    pct_preds = (100.0 * pred_after / pred_before) if pred_before else 0.0

    # Per-video %
    all_videos = sorted(orig_groups.keys())
    pcts = []
    for vid in all_videos:
        n_orig = len(orig_groups[vid])
        n_focus = len(focus_groups.get(vid, []))
        if n_orig:
            pcts.append(100.0 * n_focus / n_orig)
    mean_pct = statistics.mean(pcts) if pcts else 0.0
    median_pct = statistics.median(pcts) if pcts else 0.0
    min_pct = min(pcts) if pcts else 0.0
    max_pct = max(pcts) if pcts else 0.0
    stdev_pct = statistics.stdev(pcts) if len(pcts) >= 2 else 0.0

    return {
        "name": name,
        "frames_before": frames_before,
        "frames_after": frames_after,
        "pct_frames": pct_frames,
        "num_videos": len(all_videos),
        "min_pct_video": min_pct,
        "max_pct_video": max_pct,
        "mean_pct_video": mean_pct,
        "median_pct_video": median_pct,
        "stdev_pct_video": stdev_pct,
        "objs_before": objs_before,
        "objs_after": objs_after,
        "pct_objs": pct_objs,
        "pred_before": pred_before,
        "pred_after": pred_after,
        "pct_preds": pct_preds,
        "objs_before_set": objs_before_set,
        "objs_after_set": objs_after_set,
        "pred_before_set": pred_before_set,
        "pred_after_set": pred_after_set,
        "obj_freq_before": obj_freq_before,
        "pred_freq_before": pred_freq_before,
    }


def fmt_num(x):
    if isinstance(x, float):
        return f"{x:.1f}"
    return str(x)


def col(s, w, right=False):
    """Строка в колонке ширины w: обрезка или дополнение пробелами."""
    s = str(s)
    if len(s) > w:
        return s[:w]
    return s.rjust(w) if right else s.ljust(w)


def main():
    parser = argparse.ArgumentParser(
        description="Полная статистика key-frame selection: train/test, кадры, объекты, отношения."
    )
    parser.add_argument("--train-original", required=True, help="Оригинальный train jsonl")
    parser.add_argument("--train-focus", required=True, help="Train _focus_graph.jsonl")
    parser.add_argument("--test-original", required=True, help="Оригинальный test jsonl")
    parser.add_argument("--test-focus", required=True, help="Test _focus_graph.jsonl")
    parser.add_argument("--label", default="Key-frame selection", help="Подпись метода в заголовке таблиц (например: MaxInfo, OUR)")
    parser.add_argument("--csv", action="store_true", help="Вывести таблицы в CSV (разделитель ;)")
    args = parser.parse_args()

    train = compute_split_stats("train", args.train_original, args.train_focus)
    test = compute_split_stats("test", args.test_original, args.test_focus)
    if train is None:
        print("Ошибка: не найден train (original или focus).", file=__import__("sys").stderr)
        return 1
    if test is None:
        print("Ошибка: не найден test (original или focus).", file=__import__("sys").stderr)
        return 1

    total_before = train["frames_before"] + test["frames_before"]
    total_after = train["frames_after"] + test["frames_after"]
    total_pct = (100.0 * total_after / total_before) if total_before else 0.0
    obj_overlap_before = len(train["objs_before_set"] & test["objs_before_set"])
    obj_overlap_after = len(train["objs_after_set"] & test["objs_after_set"])
    pred_overlap_before = len(train["pred_before_set"] & test["pred_before_set"])
    pred_overlap_after = len(train["pred_after_set"] & test["pred_after_set"])

    if args.csv:
        print("Подвыборка;Кадров_до;Кадров_после;%_кадров;Видео;Объектов_до;Объектов_после;%_объектов;Предикатов_до;Предикатов_после;%_предикатов")
        for s in [train, test]:
            print(f"{s['name']};{s['frames_before']};{s['frames_after']};{fmt_num(s['pct_frames'])};{s['num_videos']};{s['objs_before']};{s['objs_after']};{fmt_num(s['pct_objs'])};{s['pred_before']};{s['pred_after']};{fmt_num(s['pct_preds'])}")
        print(f"ВСЕГО;{total_before};{total_after};{fmt_num(total_pct)};{train['num_videos'] + test['num_videos']};—;—;—;—;—;—")
        print("")
        print("Пересечение_train_test;Общих_объектов_до;Общих_объектов_после;Общих_предикатов_до;Общих_предикатов_после")
        print(f";{obj_overlap_before};{obj_overlap_after};{pred_overlap_before};{pred_overlap_after}")
        return 0

    # Фиксированные ширины колонок (числа выравниваем вправо)
    W = {
        "split": 10,
        "frames_b": 12,
        "frames_a": 12,
        "pct_f": 9,
        "videos": 6,
        "obj_b": 11,
        "obj_a": 11,
        "pct_o": 9,
        "pred_b": 12,
        "pred_a": 12,
        "pct_p": 9,
    }
    sep = "│"
    top = "─" * 10 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 9 + "┼" + "─" * 6 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┼" + "─" * 9 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 9
    hdr = "─" * 10 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 9 + "┼" + "─" * 6 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┼" + "─" * 9 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 9
    total_width = len(top) + 2

    print()
    print("╔" + "═" * (total_width - 2) + "╗")
    title = f"  {args.label}: сводная статистика (до → после отбора)"
    print("║" + title.ljust(total_width - 2) + "║")
    print("╠" + top + "╣")
    print("║" + col("Подвыборка", W["split"]) + sep +
          col("Кадров до", W["frames_b"], True) + sep +
          col("Кадров посл.", W["frames_a"], True) + sep +
          col("% кадров", W["pct_f"], True) + sep +
          col("Видео", W["videos"], True) + sep +
          col("Объект. до", W["obj_b"], True) + sep +
          col("Объект. посл.", W["obj_a"], True) + sep +
          col("% объект.", W["pct_o"], True) + sep +
          col("Предикатов до", W["pred_b"], True) + sep +
          col("Предикатов посл.", W["pred_a"], True) + sep +
          col("% предик.", W["pct_p"], True) + "║")
    print("╠" + hdr + "╣")
    for s in [train, test]:
        print("║" + col(s["name"], W["split"]) + sep +
              col(s["frames_before"], W["frames_b"], True) + sep +
              col(s["frames_after"], W["frames_a"], True) + sep +
              col(fmt_num(s["pct_frames"]), W["pct_f"], True) + sep +
              col(s["num_videos"], W["videos"], True) + sep +
              col(s["objs_before"], W["obj_b"], True) + sep +
              col(s["objs_after"], W["obj_a"], True) + sep +
              col(fmt_num(s["pct_objs"]), W["pct_o"], True) + sep +
              col(s["pred_before"], W["pred_b"], True) + sep +
              col(s["pred_after"], W["pred_a"], True) + sep +
              col(fmt_num(s["pct_preds"]), W["pct_p"], True) + "║")
    print("╠" + hdr + "╣")
    print("║" + col("ВСЕГО", W["split"]) + sep +
          col(total_before, W["frames_b"], True) + sep +
          col(total_after, W["frames_a"], True) + sep +
          col(fmt_num(total_pct), W["pct_f"], True) + sep +
          col(train["num_videos"] + test["num_videos"], W["videos"], True) + sep +
          col("—", W["obj_b"], True) + sep + col("—", W["obj_a"], True) + sep + col("—", W["pct_o"], True) + sep +
          col("—", W["pred_b"], True) + sep + col("—", W["pred_a"], True) + sep + col("—", W["pct_p"], True) + "║")
    print("╚" + "═" * (total_width - 2) + "╝")

    # Таблица 2: разнообразие по видео (% по каждому видео)
    w2, w2n = 10, 9
    sep2 = "│"
    line2 = "─" * w2 + "┼" + "─" * w2n + "┼" + "─" * w2n + "┼" + "─" * w2n + "┼" + "─" * w2n + "┼" + "─" * w2n
    tw2 = len(line2) + 2
    print()
    print("┌" + "─" * (tw2 - 2) + "┐")
    print("│" + " Разнообразие по видео (% отобранных кадров по каждому видео)".ljust(tw2 - 2) + "│")
    print("├" + line2 + "┤")
    print("│" + col("Подвыборка", w2) + sep2 +
          col("Мин %", w2n, True) + sep2 + col("Макс %", w2n, True) + sep2 +
          col("Среднее %", w2n, True) + sep2 + col("Медиана %", w2n, True) + sep2 +
          col("Стд. откл.%", w2n, True) + "│")
    print("├" + line2 + "┤")
    for s in [train, test]:
        print("│" + col(s["name"], w2) + sep2 +
              col(fmt_num(s["min_pct_video"]), w2n, True) + sep2 + col(fmt_num(s["max_pct_video"]), w2n, True) + sep2 +
              col(fmt_num(s["mean_pct_video"]), w2n, True) + sep2 + col(fmt_num(s["median_pct_video"]), w2n, True) + sep2 +
              col(fmt_num(s["stdev_pct_video"]), w2n, True) + "│")
    print("└" + "─" * (tw2 - 2) + "┘")

    # Таблица 3: пересечение train – test
    w3a, w3b = 40, 14
    sep3 = "│"
    line3 = "─" * w3a + "┼" + "─" * w3b + "┼" + "─" * w3b
    tw3 = len(line3) + 2
    print()
    print("┌" + "─" * (tw3 - 2) + "┐")
    print("│" + " Пересечение train – test".ljust(tw3 - 2) + "│")
    print("├" + line3 + "┤")
    print("│" + col("Метрика", w3a) + sep3 + col("До отбора", w3b, True) + sep3 + col("После отбора", w3b, True) + "│")
    print("├" + line3 + "┤")
    print("│" + col("Общих категорий объектов", w3a) + sep3 + col(obj_overlap_before, w3b, True) + sep3 + col(obj_overlap_after, w3b, True) + "│")
    print("│" + col("Общих типов отношений (предикатов)", w3a) + sep3 + col(pred_overlap_before, w3b, True) + sep3 + col(pred_overlap_after, w3b, True) + "│")
    print("└" + "─" * (tw3 - 2) + "┘")

    # Потерянные классы/предикаты (с частотой в полном датасете — кол-во кадров и % от всех кадров)
    print()
    print("  Потерянные после отбора (частота в полном датасете: кадров, %):")
    for s in [train, test]:
        lost_objs = s["objs_before_set"] - s["objs_after_set"]
        lost_preds = s["pred_before_set"] - s["pred_after_set"]
        total_frames = s["frames_before"]
        if lost_objs or lost_preds:
            if lost_objs:
                parts = []
                for obj in sorted(lost_objs):
                    freq = s["obj_freq_before"].get(obj, 0)
                    pct = (100.0 * freq / total_frames) if total_frames else 0.0
                    parts.append(f"{obj} ({freq}, {pct:.4f}%)")
                print(f"    [{s['name']}] объекты: {', '.join(parts)}")
            else:
                print(f"    [{s['name']}] объекты: —")
            if lost_preds:
                parts = []
                for pred in sorted(lost_preds):
                    freq = s["pred_freq_before"].get(pred, 0)
                    pct = (100.0 * freq / total_frames) if total_frames else 0.0
                    parts.append(f"{pred} ({freq}, {pct:.4f}%)")
                print(f"    [{s['name']}] предикаты: {', '.join(parts)}")
            else:
                print(f"    [{s['name']}] предикаты: —")
        else:
            print(f"    [{s['name']}] потери нет.")
    print()
    return 0


if __name__ == "__main__":
    exit(main())
