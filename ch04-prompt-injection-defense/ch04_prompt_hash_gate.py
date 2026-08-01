"""ch04_prompt_hash_gate.py — Listing 4.7: Prompt-hash gate for DSPy/AdalFlow prompts."""

import hashlib
import sys
from pathlib import Path


COMPILED_PROMPT_PATH = Path("prompts/system_prompt.json")
APPROVED_HASH_PATH = Path(".prompt-hashes/system_prompt.sha256")


def compute_sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of the file at path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not COMPILED_PROMPT_PATH.exists():
        print(f"ERROR: compiled prompt not found at {COMPILED_PROMPT_PATH}", file=sys.stderr)
        return 1

    current_hash = compute_sha256(COMPILED_PROMPT_PATH)

    if not APPROVED_HASH_PATH.exists():
        # First run: write the approved hash and instruct the team to commit it.
        APPROVED_HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
        APPROVED_HASH_PATH.write_text(current_hash + "\n")
        print(f"INFO: No approved hash found. Wrote initial hash to {APPROVED_HASH_PATH}.")
        print("Commit this file and re-run the gate to establish the baseline.")
        return 1

    approved_hash = APPROVED_HASH_PATH.read_text().strip()

    if current_hash == approved_hash:
        print(f"OK: compiled prompt hash matches approved baseline ({current_hash[:12]}...)")
        return 0

    print(
        f"FAIL: compiled prompt has changed.\n"
        f"  Approved: {approved_hash[:12]}...\n"
        f"  Current:  {current_hash[:12]}...\n"
        f"  File:     {COMPILED_PROMPT_PATH}\n"
        "Request a security review of the compiled prompt and update "
        f"{APPROVED_HASH_PATH} with the new hash once approved.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
