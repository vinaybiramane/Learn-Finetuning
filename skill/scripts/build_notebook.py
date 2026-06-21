"""
build_notebook.py — construct a valid .ipynb from cells, and VERIFY it runs.

Writing notebook JSON by hand is error-prone (escaping, cell schema). This helper
builds it correctly and — most importantly — executes every code cell so you
never hand the learner a notebook that errors.

USAGE (import it from a tiny one-shot generator script, then delete that script):

    from build_notebook import md, code, write_nb, verify_nb

    write_nb("demos/01_what_is_a_weight_matrix.ipynb", [
        md(r'''
# 01 · What is a weight matrix, *really*?
Plain-language intro. LaTeX is fine: $y = W x$.
'''),
        code('''
import numpy as np
print(np.dot([3, 1], [2, 8]))   # 14
'''),
        md(r"## Recap\n- a layer is `W @ x` ...\n\nBAM!"),
    ])

    verify_nb("demos/01_what_is_a_weight_matrix.ipynb")   # runs every code cell

Notes
-----
* Markdown cells: pass the text. Use RAW strings (r'''...''') when the markdown
  contains LaTeX, so backslashes like \\frac survive.
* Code cells: short, top-level statements so each cell shows its own output.
* verify_nb executes cells with stdout captured to a UTF-8 buffer, so Windows
  cp1252 consoles don't choke on math glyphs (Δ, α, ·) inside print()s — that is
  a console-encoding artifact, NOT a notebook problem.
"""
import io
import json
import sys
from contextlib import redirect_stdout


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def write_nb(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    n_md = sum(c["cell_type"] == "markdown" for c in cells)
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"wrote {path}  ({n_md} md + {n_code} code cells)")
    return path


def verify_nb(path):
    """Execute every code cell in order, sharing one namespace (as Jupyter does).
    Returns True if all cells run; prints the first failure otherwise."""
    nb = json.load(open(path, encoding="utf-8"))
    ns, buf = {}, io.StringIO()
    with redirect_stdout(buf):
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            try:
                exec("".join(c["source"]), ns)
            except Exception as e:  # noqa: BLE001
                print(f"{path}: code cell #{i} FAILED -> {e!r}", file=sys.stderr)
                return False
    print(f"{path}: all code cells ran OK")
    return True


if __name__ == "__main__":
    # Verify any notebooks passed as arguments: python build_notebook.py a.ipynb b.ipynb
    ok = all(verify_nb(p) for p in sys.argv[1:])
    sys.exit(0 if ok else 1)
