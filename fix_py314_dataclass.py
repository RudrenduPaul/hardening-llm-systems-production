#!/usr/bin/env python3
"""
fix_py314_dataclass.py — Fix Python 3.14 dataclass/sys.modules compatibility.

Python 3.14 changed dataclasses._is_type() to look up cls.__module__ via
sys.modules. When a module is loaded with importlib.util.spec_from_file_location
but NOT registered in sys.modules before exec_module(), dataclasses inside
that module fail with:
    AttributeError: 'NoneType' object has no attribute '__dict__'

Fix: insert `sys.modules[name] = mod_var` immediately after `module_from_spec`.

Affected notebooks: ch01, ch02, ch03 (confirmed via grep).
"""

from __future__ import annotations

import json
import pathlib
import re

COMPANION_CODE_DIR = pathlib.Path(__file__).parent

# Map notebook path → (module_var_name, module_str_name)
AFFECTED: list[tuple[pathlib.Path, str, str]] = [
    (
        COMPANION_CODE_DIR / "ch01-what-breaks-after-you-ship" / "ch01_notebook.ipynb",
        "ch01",
        "ch01_scripts",
    ),
    (
        COMPANION_CODE_DIR / "ch02-detecting-hallucinations" / "ch02_notebook.ipynb",
        "ch02",
        "ch02_scripts",
    ),
    (
        COMPANION_CODE_DIR / "ch03-containing-hallucinations-cicd" / "ch03_notebook.ipynb",
        "ch03",
        "ch03_scripts",
    ),
]


def fix_cell_source(source: list[str] | str, mod_var: str, mod_str: str) -> list[str]:
    """
    Find the exec_module line and insert a sys.modules registration just before it.
    Returns modified source lines (or original if no match found).
    """
    if isinstance(source, str):
        lines = source.splitlines(keepends=True)
    else:
        lines = list(source)

    sys_modules_line = f"sys.modules['{mod_str}'] = {mod_var}  # required for Python 3.14\n"
    exec_module_pattern = re.compile(
        rf"\s*spec\.loader\.exec_module\({re.escape(mod_var)}\)"
    )
    already_fixed = any("sys.modules[" in line and mod_str in line for line in lines)
    if already_fixed:
        return lines

    result: list[str] = []
    for line in lines:
        if exec_module_pattern.match(line):
            indent = len(line) - len(line.lstrip())
            result.append(" " * indent + sys_modules_line)
        result.append(line)

    return result


def fix_notebook(nb_path: pathlib.Path, mod_var: str, mod_str: str) -> None:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if not isinstance(src, list):
            src = [src]
        full = "".join(src)
        if "spec_from_file_location" in full and "exec_module" in full:
            new_src = fix_cell_source(src, mod_var, mod_str)
            if new_src != src:
                cell["source"] = new_src
                # Clear stale error outputs so re-run starts fresh
                cell["outputs"] = []
                cell["execution_count"] = None
                changed = True

    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  ✓ Fixed {nb_path.name}")
    else:
        print(f"  — Already fixed or not found: {nb_path.name}")


def main() -> None:
    print("fix_py314_dataclass.py\n")
    for nb_path, mod_var, mod_str in AFFECTED:
        fix_notebook(nb_path, mod_var, mod_str)
    print("\nDone. Re-run execute_notebooks.py to regenerate outputs.")


if __name__ == "__main__":
    main()
