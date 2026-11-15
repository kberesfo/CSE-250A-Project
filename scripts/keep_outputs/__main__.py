"""Mark notebooks so nbstripout keeps their outputs on commit.

nbstripout skips any notebook whose metadata has ``"keep_output": true``.
Phase and study notebooks are committed with their outputs so they render on
GitHub; eda/ and dev/ notebooks are left stripped.

    uv run python scripts/keep_outputs.py notebooks/phase1/eval.ipynb ...

A notebook that already has the flag is not rewritten, so its modification
time (which ``make`` relies on) is left alone.
"""

import sys
from pathlib import Path

import nbformat


def keep_outputs(path: Path) -> bool:
    """Set ``keep_output`` on one notebook. Returns True if the file changed."""
    nb = nbformat.read(path, as_version=4)
    if nb.metadata.get("keep_output") is True:
        return False
    nb.metadata["keep_output"] = True
    nbformat.write(nb, path)
    return True


def main(paths: list[str]) -> int:
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        print(f"not found: {', '.join(missing)}", file=sys.stderr)
        return 1
    for p in paths:
        if keep_outputs(Path(p)):
            print(f"keep_output set: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
