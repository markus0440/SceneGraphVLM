#!/usr/bin/env python3
"""
Build final SceneGraphVLM annotation variants from base SFT JSONL files.

The dataset-specific converters create base SFT `train.jsonl` / `test.jsonl`
files. This script rewrites them into the final runtime format:

- clean by default: bad boxes / dangling relations are repaired where possible,
  zero-relation samples are dropped;
- image paths are normalized to `/workspace/datasets/frames/...` by default;
- `eval`, `noprevgraph`, and GRPO variants are generated consistently.

Use `--emit-unclean` to also save non-clean variants under `unclean/`.
Use `--no-clean` if you intentionally want the main outputs to remain non-clean.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ag_scenegraph_common import ag_relation_count, clean_ag_solution_text


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPT_DIR.parents[1]
DATA_ROOT_DEFAULT = "datasets/data_playground"

IMG_W = 640
IMG_H = 480

AG_EVAL_SIZE = 148
PSG_EVAL_SIZE = 100
PVSG_EVAL_SIZE = 78

PVSG_VARIANTS = (
    "pvsg_all_data_gt_prompt",
    "pvsg_base_annot_gt_prompt",
    "pvsg_maxinfo_gt_prompt",
    "pvsg_psfr_gt_prompt",
)

TEMPORAL_START_MARKERS = (
    "\nYou are also given the previous frame's ground-truth scene graph in TOON format.",
    "\nYou are also given the previous frame scene graph in TOON format.",
    "\nYou are also given the previous frame ground-truth scene graph in TOON format.",
    "You are also given the previous frame's ground-truth scene graph in TOON format.",
    "You are also given the previous frame scene graph in TOON format.",
    "You are also given the previous frame ground-truth scene graph in TOON format.",
)
TEMPORAL_END_MARKERS = (
    "\n\n\nNow, generate the complete scene graph for the provided image.",
    "\n\nNow, generate the complete scene graph for the provided image.",
    "\nNow, generate the complete scene graph for the provided image.",
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
REL_HEADER_RE = re.compile(r"^\s*rel\[(\d+)\]\{subj,pred,obj\}:\s*$", re.I)
REL_PAIR_HEADER_RE = re.compile(
    r"^\s*rel_pairs\[(\d+)\]\{subj,attention,spatial,contacting,obj\}:\s*$",
    re.I,
)

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
HYPHENS_RE = re.compile(r"-+")
OBJ_SET_RE = re.compile(r"(predefined object set:\s*)(\[[^\]]*\])", re.S | re.I)
REL_SET_RE = re.compile(r"(predefined relationship set:\s*)(\[[^\]]*\])", re.S | re.I)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root_parts: Tuple[str, ...]
    eval_size: int
    temporal: bool
    canonicalize_grpo_prompt: bool = False


def resolve_under_repo(repo_root: Path, path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: bad JSON: {e}") from e


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def clone_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(record, ensure_ascii=False))


def clone_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [clone_record(record) for record in records]


def get_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    return messages


def get_first_message(messages: Sequence[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == role:
            return msg
    return None


def get_last_message(messages: Sequence[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == role:
            return msg
    return None


def get_answer_text(record: Dict[str, Any]) -> str:
    if isinstance(record.get("solution"), str):
        return record["solution"]
    assistant = get_last_message(get_messages(record), "assistant")
    if assistant is None or not isinstance(assistant.get("content"), str):
        raise ValueError("record has neither solution nor assistant answer")
    return assistant["content"]


def set_answer_text(record: Dict[str, Any], text: str) -> None:
    messages = record.get("messages")
    if isinstance(messages, list):
        assistant = get_last_message(messages, "assistant")
        if assistant is not None:
            assistant["content"] = text
            return
    record["solution"] = text


def select_eval(records: List[Dict[str, Any]], size: int, strategy: str, seed: int) -> List[Dict[str, Any]]:
    if size <= 0 or len(records) <= size:
        return clone_records(records)
    if strategy == "first":
        return clone_records(records[:size])
    if strategy == "random":
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(records)), size))
        return clone_records(records[i] for i in indices)
    raise ValueError(f"Unknown eval strategy: {strategy}")


def strip_prevgraph_block(text: str) -> Tuple[str, bool]:
    start = -1
    for marker in TEMPORAL_START_MARKERS:
        start = text.find(marker)
        if start != -1:
            break
    if start == -1:
        return text, False

    end = -1
    for marker in TEMPORAL_END_MARKERS:
        idx = text.find(marker, start)
        if idx != -1:
            end = idx
            break
    if end == -1:
        raise ValueError("Found previous-frame block but could not find final Now marker")
    return text[:start].rstrip() + "\n\n" + text[end:].lstrip(), True


def without_prevgraph(records: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for record in records:
        out = clone_record(record)
        for msg in out.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"], _ = strip_prevgraph_block(content)
                break
        yield out


def canon_label(value: str) -> str:
    out = str(value).strip().lower().replace("_", "-")
    out = NON_ALNUM_RE.sub("-", out)
    out = HYPHENS_RE.sub("-", out).strip("-")
    return out


def valid_bbox(parts: Sequence[str], width: int = IMG_W, height: int = IMG_H) -> bool:
    try:
        x1, y1, x2, y2 = [int(x) for x in parts]
    except Exception:
        return False
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def clean_rel_solution_text(solution: str, canonicalize_labels: bool = False) -> Tuple[str, int]:
    match = ANSWER_RE.search(solution)
    body = match.group(1) if match else solution
    lines = [line.strip() for line in body.splitlines() if line.strip()]

    obj_idx = rel_idx = None
    for i, line in enumerate(lines):
        if obj_idx is None and line.lower().startswith("obj["):
            obj_idx = i
        if REL_HEADER_RE.match(line):
            rel_idx = i
            break
    if obj_idx is None or rel_idx is None:
        return solution.strip() + "\n", 0

    objects: List[Tuple[int, str, Tuple[str, str, str, str]]] = []
    old_to_new: Dict[int, int] = {}
    for line in lines[obj_idx + 1 : rel_idx]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6 or not valid_bbox(parts[2:]):
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        label = canon_label(parts[1]) if canonicalize_labels else parts[1].strip()
        if not label or old_id in old_to_new:
            continue
        old_to_new[old_id] = len(objects) + 1
        objects.append((old_id, label, (parts[2], parts[3], parts[4], parts[5])))

    relations: List[Tuple[int, str, int]] = []
    seen = set()
    for line in lines[rel_idx + 1 :]:
        if line.startswith("</answer>"):
            break
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            old_subj = int(parts[0])
            old_obj = int(parts[2])
        except ValueError:
            continue
        if old_subj not in old_to_new or old_obj not in old_to_new:
            continue
        subj = old_to_new[old_subj]
        obj = old_to_new[old_obj]
        pred = canon_label(parts[1]) if canonicalize_labels else parts[1].strip()
        if not pred or subj == obj:
            continue
        key = (subj, pred, obj)
        if key in seen:
            continue
        seen.add(key)
        relations.append(key)

    out = ["<answer>", f"obj[{len(objects)}]{{id,name,x1,y1,x2,y2}}:"]
    for new_id, (_old_id, label, bbox) in enumerate(objects, start=1):
        out.append(f"{new_id},{label},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
    out.append(f"rel[{len(relations)}]{{subj,pred,obj}}:")
    for subj, pred, obj in relations:
        out.append(f"{subj},{pred},{obj}")
    out.append("</answer>")
    return "\n".join(out) + "\n", len(relations)


def relation_count(record: Dict[str, Any], dataset: str) -> int:
    text = get_answer_text(record)
    if dataset == "ag":
        try:
            return ag_relation_count(text)
        except Exception:
            return 0
    _cleaned, rel_count = clean_rel_solution_text(text, canonicalize_labels=False)
    return rel_count


def canonicalize_json_label_list(arr_text: str) -> str:
    arr = json.loads(arr_text)
    out: List[str] = []
    seen = set()
    for item in arr:
        label = canon_label(str(item))
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return json.dumps(out, ensure_ascii=False)


def canonicalize_psg_prompt(content: str) -> str:
    def repl_obj(match: re.Match[str]) -> str:
        return match.group(1) + canonicalize_json_label_list(match.group(2))

    def repl_rel(match: re.Match[str]) -> str:
        return match.group(1) + canonicalize_json_label_list(match.group(2))

    return REL_SET_RE.sub(repl_rel, OBJ_SET_RE.sub(repl_obj, content))


def rewrite_image_path(image_path: str, repo_root: Path, mode: str, workspace_frames_root: str) -> str:
    raw = str(image_path).replace("\\", "/")
    if mode == "keep":
        return raw

    marker = "/datasets/frames/"
    rel_tail: Optional[str] = None
    if marker in raw:
        rel_tail = raw.split(marker, 1)[1]
    elif raw.startswith("datasets/frames/"):
        rel_tail = raw[len("datasets/frames/") :]

    if mode == "workspace":
        if rel_tail is not None:
            return f"{workspace_frames_root.rstrip('/')}/{rel_tail.lstrip('/')}"
        return raw

    if mode == "local":
        if Path(raw).is_absolute():
            return str(Path(raw).resolve())
        return str((repo_root / raw).resolve())

    raise ValueError(f"Unknown image path mode: {mode}")


def rewrite_record_image_paths(
    record: Dict[str, Any],
    repo_root: Path,
    image_path_mode: str,
    workspace_frames_root: str,
) -> Dict[str, Any]:
    out = clone_record(record)
    images = out.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("record has no images list")
    out["images"] = [
        rewrite_image_path(str(image), repo_root, image_path_mode, workspace_frames_root)
        for image in images
    ]
    return out


def clean_record(record: Dict[str, Any], dataset: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    out = clone_record(record)
    answer = get_answer_text(out)
    if dataset == "ag":
        try:
            cleaned, _stats = clean_ag_solution_text(answer)
            rel_count = ag_relation_count(cleaned)
        except Exception:
            return None, True
    else:
        cleaned, rel_count = clean_rel_solution_text(answer, canonicalize_labels=False)
    if rel_count <= 0:
        return None, True
    set_answer_text(out, cleaned)
    return out, False


def prepare_sft_records(
    records: Iterable[Dict[str, Any]],
    dataset: str,
    repo_root: Path,
    image_path_mode: str,
    workspace_frames_root: str,
    clean: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    stats = {"total": 0, "kept": 0, "removed_zero_or_invalid": 0}
    for record in records:
        stats["total"] += 1
        current = clone_record(record)
        if clean:
            cleaned, removed = clean_record(current, dataset)
            if removed or cleaned is None:
                stats["removed_zero_or_invalid"] += 1
                continue
            current = cleaned
        current = rewrite_record_image_paths(current, repo_root, image_path_mode, workspace_frames_root)
        out.append(current)
        stats["kept"] += 1
    return out, stats


def infer_split(image_path: str, fallback: str) -> str:
    path = image_path.replace("\\", "/")
    if "/test_images/" in path or path.startswith("test_images/"):
        return "test"
    if "/train_images/" in path or path.startswith("train_images/"):
        return "train"
    return fallback


def infer_video_and_t(dataset: str, image_path: str, idx: int) -> Tuple[Optional[str], int]:
    path = Path(image_path)
    try:
        t = int(path.stem)
    except ValueError:
        t = idx

    if dataset == "ag":
        return path.parent.name, t
    if dataset == "pvsg":
        parts = list(path.parts)
        if "frames" in parts:
            i = len(parts) - 1 - parts[::-1].index("frames")
            if i + 1 < len(parts):
                return parts[i + 1], t
        return path.parent.name, t
    return None, t


def convert_to_grpo(
    records: Iterable[Dict[str, Any]],
    dataset: str,
    canonicalize_psg: bool,
    fallback_split: str,
) -> Iterator[Dict[str, Any]]:
    for idx, record in enumerate(records):
        messages = get_messages(record)
        user = get_first_message(messages, "user")
        if user is None or not isinstance(user.get("content"), str):
            raise ValueError("record has no user message with text content")

        user_content = user["content"]
        solution = get_answer_text(record).strip() + "\n"
        if canonicalize_psg:
            user_content = canonicalize_psg_prompt(user_content)
            solution, _ = clean_rel_solution_text(solution, canonicalize_labels=True)

        images = record.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("record has no images list")

        out: Dict[str, Any] = {
            "id": idx,
            "images": images,
            "width": int(record.get("width", IMG_W)),
            "height": int(record.get("height", IMG_H)),
            "messages": [{"role": "user", "content": user_content}],
            "solution": solution,
        }

        if dataset in {"ag", "pvsg"}:
            video_id, t = infer_video_and_t(dataset, str(images[0]), idx)
            out["video_id"] = record.get("video_id", video_id)
            out["t"] = record.get("t", t)
        elif dataset == "psg":
            out["image_id"] = record.get("image_id", Path(str(images[0])).stem)
            out["split"] = record.get("split", infer_split(str(images[0]), fallback_split))
        yield out


def write_sft_family(
    dataset_dir: Path,
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    spec: DatasetSpec,
    eval_strategy: str,
    seed: int,
    overwrite: bool,
    subdir: str = "",
) -> Dict[str, int]:
    root = dataset_dir / subdir if subdir else dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    targets = {
        "train": train_records,
        "test": test_records,
        "eval": select_eval(test_records, spec.eval_size, eval_strategy, seed),
    }
    for split, records in targets.items():
        path = root / f"{split}.jsonl"
        if path.exists() and not overwrite:
            print(f"  keep existing {path}")
            counts[split] = sum(1 for _ in iter_jsonl(path))
        else:
            counts[split] = write_jsonl(path, records)
            print(f"  wrote {path} ({counts[split]})")

    if spec.temporal:
        for split, records in list(targets.items()):
            path = root / f"{split}_noprevgraph.jsonl"
            if path.exists() and not overwrite:
                print(f"  keep existing {path}")
                counts[f"{split}_noprevgraph"] = sum(1 for _ in iter_jsonl(path))
            else:
                n = write_jsonl(path, without_prevgraph(records))
                counts[f"{split}_noprevgraph"] = n
                print(f"  wrote {path} ({n})")
    return counts


def write_grpo_family(
    dataset_dir: Path,
    spec: DatasetSpec,
    overwrite: bool,
    subdir: str = "",
) -> Dict[str, int]:
    sft_root = dataset_dir / subdir if subdir else dataset_dir
    grpo_root = sft_root / "grpo"
    grpo_root.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    split_names = ["train", "test", "eval"]
    if spec.temporal:
        split_names += ["train_noprevgraph", "test_noprevgraph", "eval_noprevgraph"]

    for split_name in split_names:
        src = sft_root / f"{split_name}.jsonl"
        dst = grpo_root / f"{split_name}.jsonl"
        if dst.exists() and not overwrite:
            print(f"  keep existing {dst}")
            counts[split_name] = sum(1 for _ in iter_jsonl(dst))
            continue

        fallback_split = "test" if split_name.startswith(("test", "eval")) else "train"
        records = list(iter_jsonl(src))
        converted = convert_to_grpo(
            records,
            dataset=spec.name,
            canonicalize_psg=spec.canonicalize_grpo_prompt,
            fallback_split=fallback_split,
        )
        counts[split_name] = write_jsonl(dst, converted)
        print(f"  wrote {dst} ({counts[split_name]})")
    return counts


def load_base_splits(dataset_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = dataset_dir / "train.jsonl"
    test = dataset_dir / "test.jsonl"
    if not train.is_file() or not test.is_file():
        raise FileNotFoundError(
            f"Missing base SFT train/test in {dataset_dir}. "
            "Run the dataset-specific sft_to_jsonl_* converter first."
        )
    return list(iter_jsonl(train)), list(iter_jsonl(test))


def build_dataset(
    spec: DatasetSpec,
    data_root: Path,
    repo_root: Path,
    eval_strategy: str,
    seed: int,
    image_path_mode: str,
    workspace_frames_root: str,
    clean: bool,
    emit_unclean: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    dataset_dir = data_root.joinpath(*spec.root_parts)
    print(f"\n[{spec.name}] {dataset_dir}")

    if dry_run:
        print(f"  requires: {dataset_dir / 'train.jsonl'}")
        print(f"  requires: {dataset_dir / 'test.jsonl'}")
        print(f"  final image path mode: {image_path_mode}")
        print(f"  clean main outputs: {clean}")
        print(f"  emit unclean outputs: {emit_unclean}")
        return

    raw_train, raw_test = load_base_splits(dataset_dir)
    report: Dict[str, Any] = {
        "dataset": spec.name,
        "root": str(dataset_dir),
        "clean_main_outputs": clean,
        "image_path_mode": image_path_mode,
        "splits": {},
    }

    if emit_unclean:
        unclean_train, unclean_train_stats = prepare_sft_records(
            raw_train, spec.name, repo_root, image_path_mode, workspace_frames_root, clean=False
        )
        unclean_test, unclean_test_stats = prepare_sft_records(
            raw_test, spec.name, repo_root, image_path_mode, workspace_frames_root, clean=False
        )
        report["unclean"] = {
            "train": unclean_train_stats,
            "test": unclean_test_stats,
            "sft_counts": write_sft_family(
                dataset_dir, unclean_train, unclean_test, spec, eval_strategy, seed, overwrite, subdir="unclean"
            ),
        }
        report["unclean"]["grpo_counts"] = write_grpo_family(dataset_dir, spec, overwrite, subdir="unclean")

    train_records, train_stats = prepare_sft_records(
        raw_train, spec.name, repo_root, image_path_mode, workspace_frames_root, clean=clean
    )
    test_records, test_stats = prepare_sft_records(
        raw_test, spec.name, repo_root, image_path_mode, workspace_frames_root, clean=clean
    )
    report["splits"]["train"] = train_stats
    report["splits"]["test"] = test_stats
    report["sft_counts"] = write_sft_family(
        dataset_dir, train_records, test_records, spec, eval_strategy, seed, overwrite
    )
    report["grpo_counts"] = write_grpo_family(dataset_dir, spec, overwrite)

    report_path = dataset_dir / "annotation_variants_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  wrote {report_path}")


def specs_from_args(args: argparse.Namespace) -> List[DatasetSpec]:
    requested = set(args.only)
    specs: List[DatasetSpec] = []

    if "all" in requested or "ag" in requested:
        specs.append(DatasetSpec("ag", ("AG_json",), args.ag_eval_size, temporal=True))

    if "all" in requested or "psg" in requested:
        specs.append(
            DatasetSpec(
                "psg",
                ("PSG_json",),
                args.psg_eval_size,
                temporal=False,
                canonicalize_grpo_prompt=not args.no_psg_canonical_grpo,
            )
        )

    if "all" in requested or "pvsg" in requested:
        variants = args.pvsg_variant or list(PVSG_VARIANTS)
        for variant in variants:
            specs.append(DatasetSpec("pvsg", ("PVSG_json", variant), args.pvsg_eval_size, temporal=True))

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final clean SFT/GRPO annotation variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT), help="SceneGraphVLM root")
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT, help="Root containing AG_json, PSG_json, PVSG_json")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("all", "ag", "psg", "pvsg"),
        default=["all"],
        help="Datasets to process",
    )
    parser.add_argument(
        "--pvsg-variant",
        action="append",
        choices=PVSG_VARIANTS,
        default=[],
        help="PVSG variant to process; repeat for a subset",
    )
    parser.add_argument("--ag-eval-size", type=int, default=AG_EVAL_SIZE)
    parser.add_argument("--psg-eval-size", type=int, default=PSG_EVAL_SIZE)
    parser.add_argument("--pvsg-eval-size", type=int, default=PVSG_EVAL_SIZE)
    parser.add_argument("--eval-strategy", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-path-mode",
        choices=("workspace", "local", "keep"),
        default="workspace",
        help="How to write image paths in all final outputs",
    )
    parser.add_argument("--workspace-frames-root", default="/workspace/datasets/frames")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean/drop zero-relation main outputs")
    parser.add_argument("--emit-unclean", action="store_true", help="Also write non-clean variants under unclean/")
    parser.add_argument("--no-psg-canonical-grpo", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing derived files")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = resolve_under_repo(repo_root, args.data_root)
    specs = specs_from_args(args)
    if not specs:
        print("Nothing to do.", file=sys.stderr)
        sys.exit(1)

    print(f"repo_root: {repo_root}")
    print(f"data_root: {data_root}")
    print(f"image path mode: {args.image_path_mode}")
    print(f"clean main outputs: {not args.no_clean}")
    print(f"emit unclean outputs: {args.emit_unclean}")

    for spec in specs:
        build_dataset(
            spec=spec,
            data_root=data_root,
            repo_root=repo_root,
            eval_strategy=args.eval_strategy,
            seed=args.seed,
            image_path_mode=args.image_path_mode,
            workspace_frames_root=args.workspace_frames_root,
            clean=not args.no_clean,
            emit_unclean=args.emit_unclean,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
