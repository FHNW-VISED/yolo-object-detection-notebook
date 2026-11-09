"""Smoke run: execute the whole built notebook headless with NB_SMOKE=1 (tiny data, CPU).

On failure, report only the section, cell index and the tail of the traceback, so the output
an agent has to read stays short.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "yolo-object-detection-notebook.ipynb"
EXECUTED = ROOT / "build" / "smoke-executed.ipynb"
CELL_TIMEOUT_S = int(os.environ.get("NB_CELL_TIMEOUT", "300"))
TOTAL_BUDGET_S = int(os.environ.get("NB_SMOKE_BUDGET", "900"))
DUMP_RESULTS = (
    "import json, pathlib\n"
    "if 'results' in globals():\n"
    "    pathlib.Path('build').mkdir(exist_ok=True)\n"
    "    pathlib.Path('build/smoke-results.json').write_text(json.dumps(results, default=str))\n"
)


def run_smoke(nb: nbformat.NotebookNode, executed: Path) -> tuple[str | None, int | None, float]:
    """Execute `nb` headless with NB_SMOKE=1 and save it to `executed`.

    Returns (failure message or None, index of the failing cell or None, seconds). The message names the
    section, the cell index and the tail of the traceback only, so the output an agent reads stays short.
    """
    os.environ["NB_SMOKE"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    client = NotebookClient(
        nb, timeout=CELL_TIMEOUT_S, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    start = time.monotonic()
    failure, failing = None, None
    try:
        client.execute()
    except CellExecutionError as err:
        failing = next(
            (
                i
                for i, c in enumerate(nb.cells)
                if c.cell_type == "code"
                and any(o.get("output_type") == "error" for o in c.get("outputs", []))
            ),
            None,
        )
        where = (
            f"{nb.cells[failing].metadata.get('section', '?')} (notebook cell {failing})"
            if failing is not None
            else "?"
        )
        plain = re.sub(r"\x1b\[[0-9;]*m", "", str(err))  # no colour codes: agents read this
        failure = f"smoke run failed in {where}:\n" + "\n".join(plain.strip().splitlines()[-25:])
    finally:
        executed.parent.mkdir(exist_ok=True)
        nbformat.write(nb, executed)
    return failure, failing, time.monotonic() - start


def test_notebook_runs_in_smoke_mode():
    assert NOTEBOOK.exists(), "run `just build` first"
    nb = nbformat.read(NOTEBOOK, as_version=4)
    # Test-only cell (never in the student notebook): dump results for tests/test_content.py.
    nb.cells.append(nbformat.v4.new_code_cell(DUMP_RESULTS))
    failure, _, elapsed = run_smoke(nb, EXECUTED)
    if failure:  # outside the except block, so pytest prints only this clean message
        pytest.fail(failure, pytrace=False)
    assert elapsed < TOTAL_BUDGET_S, f"smoke run took {elapsed:.0f}s > budget {TOTAL_BUDGET_S}s (N-015)"
