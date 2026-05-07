#!/usr/bin/env python3
"""
Sanitize this repository for double-blind review / Anonymous GitHub.

Run from the repository root:
    python sanitize_double_blind.py

Recommended workflow:
    git checkout -b anonymous-review
    python sanitize_double_blind.py --dry-run
    python sanitize_double_blind.py
    git diff
    git status

The script:
- replaces known local usernames, home paths, and non-anonymous GitHub links;
- clears Jupyter notebook outputs and execution counts;
- removes common local-only artifacts (.DS_Store, __pycache__, .ipynb_checkpoints);
- prints a post-check with any remaining suspicious terms.

It intentionally does NOT rename the project/method name SceneGraphVLM by default.
Use --rename-project only if the method/repo name itself can identify the authors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Files/directories we should never edit in-place.
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

# Text-like extensions. Unknown files are tested by UTF-8 decoding.
TEXT_EXTS = {
    ".bat",
    ".bib",
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".dockerfile",
    ".env",
    ".gitignore",
    ".html",
    ".ini",
    ".ipynb",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# Ordered: specific replacements first, broad replacements later.
REPLACEMENTS: list[tuple[str, str]] = [
    # Exact local paths found in this repo.
    (r"/data/homes/makarov_vd/workspace/SceneGraphVLM", r"/path/to/SceneGraphVLM"),
    (r"/data/homes/makarov_vd", r"/path/to/home"),
    (r"/Users/dark-creator/solomon/self/key-frame-selection/tmp/data/indoor", r"tmp/data/indoor"),
    (r"/Users/dark-creator/solomon/self/key-frame-selection/tmp/output/indoor-key-frames", r"tmp/output/indoor-key-frames"),
    (r"/Users/dark-creator/solomon/self/key-frame-selection/tmp/output/indoor", r"tmp/output/indoor"),
    (r"/Users/dark-creator/solomon/self/key-frame-selection/tmp/output/visualization", r"tmp/output/visualization"),
    (r"/Users/dark-creator/solomon/self/key-frame-selection", r"/path/to/key-frame-selection"),
    (r"/Users/dark-creator", r"/path/to/home"),

    # Usernames / handles.
    (r"\bmakarov[_-]?vd\b", r"anonymous_user"),
    (r"\bdark-creator\b", r"anonymous_user"),
    (r"\bstrangecreator/key-frame-selection\b", r"anonymous/key-frame-selection"),
    (r"github\.com/anonymous/key-frame-selection(?:\.git)?", r"github.com/anonymous/key-frame-selection"),
    (r"github\.com/strangecreator/key-frame-selection(?:\.git)?", r"github.com/anonymous/key-frame-selection"),
    (r"\bstrangecreator\b", r"anonymous"),

    # Wording that can suggest a personal connection to an identifiable repo/person.
    (
        r"In this repository, the \*\*per-frame selection engine\*\* comes from a colleague[’']s standalone project:",
        r"In this repository, the **per-frame selection engine** is adapted from an external standalone implementation:",
    ),
    (
        r"the \*\*per-frame selection engine\*\* comes from a colleague[’']s standalone project",
        r"the **per-frame selection engine** is adapted from an external standalone implementation",
    ),

    # Generic email fallback. Keep after the specific replacements.
    (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", r"anonymous@example.com"),
]

PROJECT_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bSceneGraphVLM\b", r"AnonymousVLM"),
]

SUSPICIOUS_PATTERNS: list[tuple[str, str]] = [
    ("local Linux home", r"/data/homes/"),
    ("local macOS home", r"/Users/"),
    ("makarov username", r"makarov[_-]?vd"),
    ("dark-creator username", r"dark-creator"),
    ("strangecreator handle", r"strangecreator"),
    ("personal wording", r"colleague[’']s standalone project"),
    ("email address", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
]

LOCAL_ARTIFACT_NAMES = {".DS_Store"}
LOCAL_ARTIFACT_DIRS = {".ipynb_checkpoints", "__pycache__"}

CANDIDATE_SUBSTRINGS = (
    "/data/homes",
    "/Users/",
    "makarov",
    "dark-creator",
    "strangecreator",
    "github.com/",
    "colleague",
    "@",
)


def might_need_replacements(text: str, rename_project: bool = False) -> bool:
    lowered = text.lower()
    if rename_project and "scenegraphvlm" in lowered:
        return True
    return any(token.lower() in lowered for token in CANDIDATE_SUBSTRINGS)


def run_git_ls_files(root: Path) -> list[Path] | None:
    """Return tracked files if root is a git repo; otherwise None."""
    try:
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    files = [root / line for line in res.stdout.splitlines() if line.strip()]
    return files if files else None


def walk_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            yield base / name


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTS or path.name in TEXT_EXTS:
        return True
    try:
        data = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def apply_replacements(text: str, rename_project: bool = False) -> str:
    if not might_need_replacements(text, rename_project=rename_project):
        return text
    patterns = REPLACEMENTS + (PROJECT_REPLACEMENTS if rename_project else [])
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def clean_notebook(path: Path, rename_project: bool = False) -> tuple[bool, str]:
    old = path.read_text(encoding="utf-8")
    try:
        nb = json.loads(old)
    except json.JSONDecodeError:
        # Fallback: treat as plain text if notebook is malformed.
        new = apply_replacements(old, rename_project=rename_project)
        return new != old, new

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

            # Make demo notebooks less machine-specific when possible.
            source = cell.get("source")
            if isinstance(source, list):
                src = "".join(source)
                src2 = src.replace(
                    'ROOT = Path("/data/homes/makarov_vd/workspace/SceneGraphVLM")',
                    'ROOT = Path.cwd()',
                )
                src2 = re.sub(
                    r'CHECKPOINT_DIR = ROOT / "sft/Qwen3\.5/work_dirs/[^"]+"',
                    'CHECKPOINT_DIR = ROOT / "sft/Qwen3.5/work_dirs/<checkpoint>"',
                    src2,
                )
                if src2 != src:
                    cell["source"] = src2.splitlines(keepends=True)
                    changed = True

    new = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    new = apply_replacements(new, rename_project=rename_project)
    # Validate after regex substitutions.
    json.loads(new)
    return changed or (new != old), new


def clean_text_file(path: Path, rename_project: bool = False) -> tuple[bool, str | None]:
    try:
        old = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, None

    if path.suffix.lower() == ".ipynb":
        return clean_notebook(path, rename_project=rename_project)

    new = apply_replacements(old, rename_project=rename_project)
    return new != old, new


def remove_local_artifacts(root: Path, dry_run: bool) -> list[Path]:
    removed: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Do not descend into .git or other skipped dirs except artifact dirs we want to remove.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS or d in LOCAL_ARTIFACT_DIRS]
        base = Path(dirpath)

        for dirname in list(dirnames):
            if dirname in LOCAL_ARTIFACT_DIRS:
                target = base / dirname
                removed.append(target)
                if not dry_run:
                    shutil.rmtree(target, ignore_errors=True)
                dirnames.remove(dirname)

        for filename in filenames:
            if filename in LOCAL_ARTIFACT_NAMES:
                target = base / filename
                removed.append(target)
                if not dry_run:
                    target.unlink(missing_ok=True)
    return removed


def collect_files(root: Path) -> list[Path]:
    tracked = run_git_ls_files(root)
    if tracked is not None:
        return [p for p in tracked if p.is_file()]
    return [p for p in walk_files(root) if p.is_file()]


def scan_suspicious(root: Path, files: Iterable[Path]) -> list[tuple[str, Path, int, str]]:
    hits: list[tuple[str, Path, int, str]] = []
    compiled = [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in SUSPICIOUS_PATTERNS]

    for path in files:
        if not path.exists() or not path.is_file() or not is_probably_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not might_need_replacements(text, rename_project=False):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, rx in compiled:
                if rx.search(line):
                    snippet = line.strip()
                    if len(snippet) > 220:
                        snippet = snippet[:217] + "..."
                    hits.append((label, path.relative_to(root), lineno, snippet))
                    break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanitize repo for double-blind review.")
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=".",
        help="Path to repository root. Default: current directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    parser.add_argument(
        "--rename-project",
        action="store_true",
        help="Also replace SceneGraphVLM -> AnonymousVLM. Use only if the method name itself deanonymizes you.",
    )
    parser.add_argument(
        "--no-clean-artifacts",
        action="store_true",
        help="Do not remove .DS_Store, __pycache__, or .ipynb_checkpoints.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: repo root does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    files = collect_files(root)
    changed: list[Path] = []
    skipped_binary = 0

    for path in files:
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if not is_probably_text(path):
            skipped_binary += 1
            continue
        try:
            did_change, new_text = clean_text_file(path, rename_project=args.rename_project)
        except Exception as exc:
            print(f"WARN: failed to process {rel}: {exc}", file=sys.stderr)
            continue
        if did_change and new_text is not None:
            changed.append(rel)
            if not args.dry_run:
                path.write_text(new_text, encoding="utf-8", newline="")

    removed: list[Path] = []
    if not args.no_clean_artifacts:
        removed = remove_local_artifacts(root, dry_run=args.dry_run)

    # Recollect after modifications/deletions for post-check.
    final_files = collect_files(root)
    hits = [] if args.dry_run else scan_suspicious(root, final_files)

    prefix = "DRY-RUN: " if args.dry_run else ""
    print(f"{prefix}changed text/notebook files: {len(changed)}")
    for rel in changed[:200]:
        print(f"  M {rel}")
    if len(changed) > 200:
        print(f"  ... and {len(changed) - 200} more")

    print(f"{prefix}removed local artifacts: {len(removed)}")
    for path in removed[:100]:
        try:
            print(f"  D {path.relative_to(root)}")
        except ValueError:
            print(f"  D {path}")
    if len(removed) > 100:
        print(f"  ... and {len(removed) - 100} more")

    print(f"skipped binary/non-UTF8 files: {skipped_binary}")

    if args.dry_run:
        print("\nDry run only: no files were changed, so the post-check was skipped.")
        print("Run without --dry-run, then inspect `git diff` before pushing.")
        return 0

    if hits:
        print("\nWARNING: remaining suspicious terms found:")
        for label, rel, lineno, snippet in hits[:80]:
            print(f"  [{label}] {rel}:{lineno}: {snippet}")
        if len(hits) > 80:
            print(f"  ... and {len(hits) - 80} more")
        print("\nReview these before pushing.")
        return 1

    print("\nPost-check passed: no configured suspicious terms remain.")
    print("Next: run `git diff`, inspect the notebook/config changes, then push the anonymous-review branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
