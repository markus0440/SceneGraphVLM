#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для подсчёта процента отобранных кадров (key-frame selection) по каждому видео:
сравнивает оригинальный jsonl и *_focus_graph.jsonl, выводит статистику по видео и общую.
Запуск из корня репозитория:
  python key-frame-selection/scripts/pct_per_video.py \
    --original data_playground/pvsg_sgg_prev/original_data/pvsg_sgg_prev_test.jsonl \
    --focus data_playground/pvsg_sgg/pvsg_sgg_prev_test_focus_graph.jsonl
"""

import argparse
import json
import re
import sys
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
    """video_id -> list of records (order preserved by first appearance)."""
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


def main():
    parser = argparse.ArgumentParser(description="Процент отобранных кадров по каждому видео (original vs focus_graph).")
    parser.add_argument("--original", required=True, help="Оригинальный jsonl (до key-frame selection)")
    parser.add_argument("--focus", required=True, help="Файл *_focus_graph.jsonl (после отбора)")
    parser.add_argument("--label", default="Key-frame selection", help="Подпись метода в заголовке (например: MaxInfo, OUR)")
    parser.add_argument("--verbose", action="store_true", help="Вывести таблицу по каждому видео")
    args = parser.parse_args()

    original_records = load_jsonl(args.original)
    focus_records = load_jsonl(args.focus)

    orig_groups = group_records_by_video(original_records)
    focus_groups = group_records_by_video(focus_records)

    all_videos = sorted(orig_groups.keys())
    if set(focus_groups) - set(orig_groups):
        print("Warning: в focus есть видео, которых нет в original — они пропущены.", file=sys.stderr)

    stats = []
    for vid in all_videos:
        n_orig = len(orig_groups[vid])
        n_focus = len(focus_groups.get(vid, []))
        pct = (100.0 * n_focus / n_orig) if n_orig else 0.0
        stats.append({"video_id": vid, "original": n_orig, "selected": n_focus, "pct": pct})

    total_orig = sum(s["original"] for s in stats)
    total_sel = sum(s["selected"] for s in stats)
    total_pct = (100.0 * total_sel / total_orig) if total_orig else 0.0

    pcts = [s["pct"] for s in stats]
    mean_pct = statistics.mean(pcts) if pcts else 0.0
    median_pct = statistics.median(pcts) if pcts else 0.0
    min_pct = min(pcts) if pcts else 0.0
    max_pct = max(pcts) if pcts else 0.0

    print("=" * 70)
    print(f"{args.label}: доля отобранных кадров по видео (original → после отбора)")
    print("=" * 70)
    print(f"  Оригинал:     {total_orig} кадров")
    print(f"  После отбора: {total_sel} кадров")
    print(f"  Общий процент отбора: {total_pct:.1f}%")
    print()
    print("  По видео (разнообразие):")
    print(f"    Число видео:     {len(stats)}")
    print(f"    Мин % по видео:  {min_pct:.1f}%")
    print(f"    Макс % по видео: {max_pct:.1f}%")
    print(f"    Среднее %:       {mean_pct:.1f}%")
    print(f"    Медиана %:       {median_pct:.1f}%")
    if len(pcts) >= 2:
        print(f"    Стд. откл. %:    {statistics.stdev(pcts):.1f}%")
    print("=" * 70)

    if args.verbose:
        print("\nПо каждому видео (video_id | original | selected | %):")
        for s in stats:
            print(f"  {s['video_id']}  {s['original']:5d}  {s['selected']:5d}  {s['pct']:5.1f}%")


if __name__ == "__main__":
    main()
