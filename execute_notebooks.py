#!/usr/bin/env python3
"""
execute_notebooks.py — Run all companion notebooks and save outputs.

Usage (from companion-code/ directory):
    python3 execute_notebooks.py            # run all chapters
    python3 execute_notebooks.py ch01 ch04  # run specific chapters only

Each notebook is executed in its own chapter directory so relative paths
(e.g. pathlib.Path("ch0N_scripts.py")) resolve correctly.

Outputs are saved in-place — the .ipynb file is overwritten with outputs
populated. GitHub renders these outputs directly; Colab users see them too.
"""

from __future__ import annotations

import subprocess
import sys
import pathlib
import time

COMPANION_CODE_DIR = pathlib.Path(__file__).parent
TIMEOUT = 600  # seconds per notebook

CHAPTERS = [
    "ch01-what-breaks-after-you-ship",
    "ch02-detecting-hallucinations",
    "ch03-containing-hallucinations-cicd",
    "ch04-prompt-injection-defense",
    "ch05-rag-retrieval-security",
    "ch06-red-teaming",
    "ch07-autonomous-agents-scope-containment-monitoring",
    "ch08-pii-memorization-right-to-erasure",
    "ch09-bias-harmful-output-content-safety",
    "ch10-eu-ai-act-nist-engineering-artifacts",
    "ch11-hardening-stack-pr-gate",
]

CHAPTER_IDS = {slug: f"ch{slug[2:4]}" for slug in CHAPTERS}


def run_notebook(chapter_slug: str) -> tuple[bool, str]:
    """Execute a single notebook in-place. Returns (success, message)."""
    chapter_id = CHAPTER_IDS[chapter_slug]
    nb_path = COMPANION_CODE_DIR / chapter_slug / f"{chapter_id}_notebook.ipynb"

    if not nb_path.exists():
        return False, f"Notebook not found: {nb_path}"

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={TIMEOUT}",
        "--ExecutePreprocessor.kernel_name=python3",
        # Don't abort on cell errors — capture them as output
        "--ExecutePreprocessor.allow_errors=True",
        str(nb_path),
    ]

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(COMPANION_CODE_DIR / chapter_slug),
        timeout=TIMEOUT + 30,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        return True, f"OK ({elapsed:.0f}s)"
    else:
        # nbconvert stderr often contains the actual error
        err = (result.stderr or result.stdout or "")[-300:]
        return False, f"FAILED ({elapsed:.0f}s): {err.strip()}"


def main() -> None:
    # Allow filtering to specific chapters by passing ch01, ch02, etc.
    requested = [a.lower().strip("ch") for a in sys.argv[1:]]
    chapters_to_run = [
        slug for slug in CHAPTERS
        if not requested or slug[2:4] in requested or slug in requested
    ]

    print(f"execute_notebooks.py — executing {len(chapters_to_run)} notebook(s)\n")

    results: list[tuple[str, bool, str]] = []
    for slug in chapters_to_run:
        chapter_id = CHAPTER_IDS[slug]
        print(f"  [{chapter_id}] running...", end=" ", flush=True)
        ok, msg = run_notebook(slug)
        status = "✓" if ok else "✗"
        print(f"{status} {msg}")
        results.append((chapter_id, ok, msg))

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed, {failed} failed")

    if failed:
        print("\nFailed chapters:")
        for chapter_id, ok, msg in results:
            if not ok:
                print(f"  {chapter_id}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
