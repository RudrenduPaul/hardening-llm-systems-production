#!/usr/bin/env python3
"""
notebook_prep.py — Add Colab badge + setup cell to all companion notebooks.

Run from companion-code/ directory:
    python3 notebook_prep.py

What it does per notebook:
  1. Inserts Colab badge into the first markdown cell (after the # heading)
  2. Inserts a Colab-conditional setup code cell (git clone + pip install)
  3. Leaves all other content unchanged
"""

from __future__ import annotations

import json
import pathlib
import re
import uuid

COMPANION_CODE_DIR = pathlib.Path(__file__).parent
GITHUB_REPO_OWNER = "RudrenduPaul"
GITHUB_REPO_NAME = "hardening-llm-systems-production"
GITHUB_REPO = f"{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}"

# Per-chapter: (folder slug, chNN prefix, pip packages for Colab install)
CHAPTERS: list[tuple[str, str, str]] = [
    (
        "ch01-what-breaks-after-you-ship",
        "ch01",
        "",  # stdlib only
    ),
    (
        "ch02-detecting-hallucinations",
        "ch02",
        "deepeval==0.21.7 ragas==0.1.21 matplotlib",
    ),
    (
        "ch03-containing-hallucinations-cicd",
        "ch03",
        "deepeval==0.21.7 matplotlib",
    ),
    (
        "ch04-prompt-injection-defense",
        "ch04",
        "pydantic>=2.0",
    ),
    (
        "ch05-rag-retrieval-security",
        "ch05",
        "numpy matplotlib pydantic>=2.0",
    ),
    (
        "ch06-red-teaming",
        "ch06",
        "matplotlib",
    ),
    (
        "ch07-autonomous-agents-scope-containment",
        "ch07",
        "langgraph==0.2.0 langchain-openai==0.2.0",
    ),
    (
        "ch08-agent-telemetry-incident-response",
        "ch08",
        "sentence-transformers==2.6.0 opentelemetry-sdk==1.21.0 langfuse==2.28.0",
    ),
    (
        "ch09-data-leakage-bias-pii",
        "ch09",
        "scipy matplotlib textblob presidio-analyzer==2.2.354 presidio-anonymizer==2.2.354",
    ),
    (
        "ch10-eu-ai-act-nist-compliance",
        "ch10",
        "pyyaml pydantic>=2.0",
    ),
    (
        "ch11-hardening-stack-pr-gate",
        "ch11",
        "pydantic>=2.0 langchain-core",
    ),
]


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _colab_badge_line(chapter_slug: str, chapter_id: str) -> str:
    nb_path = (
        f"companion-code/{chapter_slug}/{chapter_id}_notebook.ipynb"
    )
    colab_url = (
        f"https://colab.research.google.com/github/{GITHUB_REPO}"
        f"/blob/main/{nb_path}"
    )
    return (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_url})\n"
    )


def _colab_setup_cell(chapter_slug: str, packages: str) -> dict:
    """Return a code cell with Colab-conditional setup (clone + pip install)."""
    pip_line = (
        f'    !pip install -q {packages}\n' if packages else ""
    )
    source: list[str] = [
        "# ── Colab setup ────────────────────────────────────────────────────────────\n",
        "# This cell only runs when executed in Google Colab.\n",
        "# Local Jupyter users: skip — all code is stdlib or pip-installable.\n",
        "import sys, os\n",
        "\n",
        "if 'google.colab' in sys.modules:\n",
        f"    !git clone -q https://github.com/{GITHUB_REPO}.git\n",
        f"    os.chdir('{GITHUB_REPO_NAME}/companion-code/{chapter_slug}')\n",
    ]
    if pip_line:
        source.append(pip_line)
    source.append(
        "    print('Colab setup complete — repo cloned, packages installed.')\n"
    )

    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _new_id(),
        "metadata": {"tags": ["colab-setup"]},
        "outputs": [],
        "source": source,
    }


def _inject_badge_into_markdown(source: list[str] | str, badge_line: str) -> list[str]:
    """Insert the Colab badge on the line after the first # heading."""
    if isinstance(source, str):
        lines = source.splitlines(keepends=True)
    else:
        lines = list(source)

    # Already has a badge? Skip.
    full_text = "".join(lines)
    if "colab-badge.svg" in full_text:
        return lines

    result: list[str] = []
    badge_added = False
    for line in lines:
        result.append(line)
        if not badge_added and re.match(r"^#[^#]", line.strip()):
            # Ensure the heading line ends with \n
            if not result[-1].endswith("\n"):
                result[-1] += "\n"
            result.append("\n")
            result.append(badge_line)
            badge_added = True

    return result


def process_notebook(chapter_slug: str, chapter_id: str, packages: str) -> None:
    nb_path = COMPANION_CODE_DIR / chapter_slug / f"{chapter_id}_notebook.ipynb"

    if not nb_path.exists():
        print(f"  SKIP {nb_path.name} — file not found")
        return

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells: list[dict] = nb["cells"]
    badge_line = _colab_badge_line(chapter_slug, chapter_id)

    # 1. Inject badge into the first markdown cell
    for cell in cells:
        if cell["cell_type"] == "markdown":
            cell["source"] = _inject_badge_into_markdown(cell["source"], badge_line)
            break

    # 2. Insert Colab setup cell after the first markdown cell
    already_has_colab = any(
        "colab-setup" in cell.get("metadata", {}).get("tags", [])
        for cell in cells
    )
    if not already_has_colab:
        insert_pos = 1  # default: after the very first cell
        for i, cell in enumerate(cells):
            if cell["cell_type"] == "markdown":
                insert_pos = i + 1
                break
        cells.insert(insert_pos, _colab_setup_cell(chapter_slug, packages))

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  ✓ {chapter_id}_notebook.ipynb — badge + Colab cell added")


def main() -> None:
    print("notebook_prep.py — preparing notebooks for Colab + execution\n")
    for chapter_slug, chapter_id, packages in CHAPTERS:
        process_notebook(chapter_slug, chapter_id, packages)
    print("\nDone. Run execute_notebooks.py (or nbconvert) to populate outputs.")


if __name__ == "__main__":
    main()
