#!/usr/bin/env python3
"""
End-to-end annotation preparation entrypoint for SceneGraphVLM.

This script intentionally depends only on files and tools inside the
SceneGraphVLM repository plus user-downloaded raw datasets described in the
dataset READMEs.

Pipeline:
  1) prepare raw annotations/media into intermediate TOON JSON + 640x480 frames
  2) export base SFT train/test JSONL files
  3) derive eval, noprevgraph, and GRPO annotation variants

Use --dry-run first to inspect the commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPT_DIR.parents[1]


def run_cmd(cmd: Sequence[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def py(script: str, *args: str) -> List[str]:
    return [sys.executable, script, *args]


def wants(args: argparse.Namespace, name: str) -> bool:
    return "all" in args.datasets or name in args.datasets


def prepare_psg(args: argparse.Namespace, repo_root: Path) -> None:
    if args.skip_prepare:
        print("\n[psg] skip prepare stage")
    else:
        cmd = py("datasets/annotations/PSG_annot/tools/prepare_original_psg_sft.py")
        if args.psg_from_hf:
            cmd.append("--from-hf")
        if args.skip_existing_toon:
            cmd.append("--skip-sft")
        run_cmd(cmd, repo_root, args.dry_run)

    if args.skip_sft:
        print("\n[psg] skip SFT JSONL export")
    else:
        run_cmd(py("datasets/annotations/PSG_annot/tools/sft_to_jsonl_psg.py"), repo_root, args.dry_run)

    if not args.skip_variants:
        run_cmd(
            py("datasets/tools/build_annotation_variants.py", "--only", "psg", *variant_args(args)),
            repo_root,
            args.dry_run,
        )


def prepare_ag(args: argparse.Namespace, repo_root: Path) -> None:
    ag_root = repo_root / "datasets/annotations/AG_annot"

    if args.ag_dump_frames and not args.skip_prepare:
        dump_cmd = py("tools/dump_frames.py")
        if args.ag_dump_all_frames:
            dump_cmd.append("--all_frames")
        run_cmd(dump_cmd, ag_root, args.dry_run)

    if args.skip_prepare:
        print("\n[ag] skip prepare stage")
    else:
        run_cmd(
            py(
                "datasets/annotations/AG_annot/tools/prepare_original_ag_sft.py",
                "--num_workers",
                str(args.num_workers),
            ),
            repo_root,
            args.dry_run,
        )

    if args.skip_sft:
        print("\n[ag] skip SFT JSONL export")
    else:
        run_cmd(py("datasets/annotations/AG_annot/tools/sft_to_jsonl_ag.py"), repo_root, args.dry_run)

    if not args.skip_variants:
        run_cmd(
            py("datasets/tools/build_annotation_variants.py", "--only", "ag", *variant_args(args)),
            repo_root,
            args.dry_run,
        )


def prepare_pvsg(args: argparse.Namespace, repo_root: Path) -> None:
    if args.skip_prepare:
        print("\n[pvsg] skip prepare/filter stage")
    else:
        run_cmd(
            py(
                "datasets/annotations/PVSG_annot/tools/prepare_original_pvsg_sft.py",
                "--num_workers",
                str(args.num_workers),
            ),
            repo_root,
            args.dry_run,
        )

        if "all" in args.pvsg_filters or "base" in args.pvsg_filters:
            run_cmd(py("utils/BaseAnnot/prepare_filtered_pvsg_sft.py"), repo_root, args.dry_run)

        if "all" in args.pvsg_filters or "maxinfo" in args.pvsg_filters:
            maxinfo_cmd = py("utils/MaxInfo/pvsg_maxinfo_filter.py")
            if args.maxinfo_fp16:
                maxinfo_cmd.append("--fp16")
            run_cmd(maxinfo_cmd, repo_root, args.dry_run)

        if "all" in args.pvsg_filters or "psfr" in args.pvsg_filters:
            run_cmd(py("utils/PSFR/pvsg_psfr_filter.py"), repo_root, args.dry_run)

    if args.skip_sft:
        print("\n[pvsg] skip SFT JSONL export")
    else:
        pvsg_exports = [
            (
                "datasets/annotations/PVSG_annot/data_sft_original",
                "datasets/data_playground/PVSG_json/pvsg_all_data_gt_prompt",
            ),
            (
                "datasets/annotations/PVSG_annot/data_sft_base_annot",
                "datasets/data_playground/PVSG_json/pvsg_base_annot_gt_prompt",
            ),
            (
                "datasets/annotations/PVSG_annot/data_sft_maxinfo",
                "datasets/data_playground/PVSG_json/pvsg_maxinfo_gt_prompt",
            ),
            (
                "datasets/annotations/PVSG_annot/data_sft_psfr",
                "datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt",
            ),
        ]
        for input_dir, output_dir in pvsg_exports:
            run_cmd(
                py(
                    "datasets/annotations/PVSG_annot/tools/sft_to_jsonl_pvsg.py",
                    "--input_dir",
                    input_dir,
                    "--output_dir",
                    output_dir,
                ),
                repo_root,
                args.dry_run,
            )

    if not args.skip_variants:
        run_cmd(
            py("datasets/tools/build_annotation_variants.py", "--only", "pvsg", *variant_args(args)),
            repo_root,
            args.dry_run,
        )


def variant_args(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    if args.overwrite:
        out.append("--overwrite")
    if args.eval_strategy:
        out.extend(["--eval-strategy", args.eval_strategy])
    out.extend(["--image-path-mode", args.image_path_mode])
    if args.no_clean:
        out.append("--no-clean")
    if args.emit_unclean:
        out.append("--emit-unclean")
    if args.no_psg_canonical_grpo:
        out.append("--no-psg-canonical-grpo")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SceneGraphVLM annotation files from raw datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT), help="SceneGraphVLM root")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("all", "ag", "psg", "pvsg"),
        default=["all"],
        help="Datasets to prepare",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Skip raw/intermediate TOON stage")
    parser.add_argument("--skip-sft", action="store_true", help="Skip base SFT train/test JSONL export")
    parser.add_argument("--skip-variants", action="store_true", help="Skip eval/noprevgraph/GRPO derivation")
    parser.add_argument("--skip-existing-toon", action="store_true", help="Pass --skip-sft to PSG prepare script")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite derived variant files")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")

    parser.add_argument("--num-workers", type=int, default=32, help="Workers for AG/PVSG prepare scripts")
    parser.add_argument("--psg-from-hf", action="store_true", help="Download PSG annotations from Hugging Face if needed")
    parser.add_argument("--ag-dump-frames", action="store_true", help="Run AG Charades frame dumping before AG prepare")
    parser.add_argument("--ag-dump-all-frames", action="store_true", help="Dump all AG frames instead of annotated frames only")
    parser.add_argument(
        "--pvsg-filters",
        nargs="+",
        choices=("all", "base", "maxinfo", "psfr"),
        default=["all"],
        help="PVSG filters to run during prepare stage",
    )
    parser.add_argument("--maxinfo-fp16", action="store_true", help="Pass --fp16 to PVSG MaxInfo filter")

    parser.add_argument("--eval-strategy", choices=("first", "random"), default="first")
    parser.add_argument("--image-path-mode", choices=("workspace", "local", "keep"), default="workspace")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean/drop zero-relation main outputs")
    parser.add_argument("--emit-unclean", action="store_true", help="Also save non-clean variants under unclean/")
    parser.add_argument("--no-psg-canonical-grpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repo root not found: {repo_root}")

    print(f"repo_root: {repo_root}")
    print("This pipeline uses only SceneGraphVLM scripts and user-provided raw datasets.")

    if wants(args, "psg"):
        prepare_psg(args, repo_root)
    if wants(args, "ag"):
        prepare_ag(args, repo_root)
    if wants(args, "pvsg"):
        prepare_pvsg(args, repo_root)


if __name__ == "__main__":
    main()
