"""Mechanical checks for the rules in RULES.md that a script can decide.

Prints one line per problem (compact on purpose: agents read this output) and exits 1 on any FAIL.
Rule IDs refer to RULES.md. Word budgets (cut pass, 2026-10-22) and mentions of a question before it is
asked (N-052) only print WARNING lines: they never change the exit code.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import jupytext

from build import CALLOUT_KINDS, MARKER_HINT, parse_marker  # same marker grammar as the build

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
SPEC = ROOT / "SPEC.md"

Q_RE = re.compile(r"\*\*Q(\d+)\.?\*\*")
Q_MENTION_RE = re.compile(r"\bQ(\d+)\b")
LEVELS = ("🟢", "🟡", "🔴")
HELPER_MIN_LINES = 8
# N-035: the only allowed first lines of a code cell (collapsed, a question's cell, or a cell to edit).
CODE_START_RE = re.compile(r"^# (@title\b|Q\d+\b|✏️ YOUR TURN|✍️ YOUR CODE)")
BOXPLOT_RE = re.compile(r"\bboxplot\(")

# Visible words allowed per section: markdown and callout text, without go-deeper boxes, <details> blocks
# and 🔴 Optional parts. A target, not a gate: an over-budget section prints a WARNING.
WORD_BUDGETS = {0: 400, 1: 400, 2: 550, 3: 550, 4: 500, 5: 800, 6: 450, 7: 650}  # SPEC L-001
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
DETAILS_RE = re.compile(r"<details\b.*?</details>", re.S | re.I)
HTML_RE = re.compile(r"<!--.*?-->|<[^>]+>", re.S)
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
WORD_RE = re.compile(r"\S*\w\S*")

problems: list[str] = []
warnings: list[str] = []


def fail(where: str, msg: str) -> None:
    problems.append(f"FAIL {where}: {msg}")


def is_helper_cell(src: str) -> bool:
    """A code cell that only imports / defines functions or classes (no top-level work)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    body = [n for n in tree.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)]
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    return (
        bool(body)
        and all(isinstance(n, defs) for n in body)
        and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in body)
    )


def forward_question_refs(line: str, next_q: int) -> list[int]:
    """N-052 (4): the question numbers a markdown line mentions before they are asked. `next_q` is the first
    question not asked yet; the line's own **Qn.** markers count as asked."""
    for m in Q_RE.finditer(line):
        next_q = max(next_q, int(m.group(1)) + 1)
    return [int(m.group(1)) for m in Q_MENTION_RE.finditer(Q_RE.sub(" ", line)) if int(m.group(1)) >= next_q]


def callout_kind(src: str) -> str | None:
    """The kind of a markdown cell written as a callout marker cell, else None."""
    parsed = parse_marker(src.lstrip("\n").partition("\n")[0])
    return parsed[0] if parsed else None


def check_callout(where: str, src: str) -> None:
    """N-034: coloured boxes are callout marker cells, never markdown tables with bgcolor."""
    if re.search(r"<table\b", src, re.I) and re.search(r"\bbgcolor\b", src, re.I):
        fail(where, "coloured box as a markdown table: write a '<!-- callout: KIND | TITLE -->' cell (N-034)")
    if MARKER_HINT not in src:
        return
    kind = callout_kind(src)
    if kind is None or src.count(MARKER_HINT) > 1:
        fail(where, "malformed callout marker: line 1 must be '<!-- callout: KIND | TITLE -->' (N-034)")
    elif kind not in CALLOUT_KINDS:
        fail(where, f"unknown callout kind '{kind}', known: {', '.join(CALLOUT_KINDS)} (N-034)")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(LINK_RE.sub(r"\1", HTML_RE.sub(" ", text))))


def visible_words(cells) -> int:
    """Words a student reads without opening anything: markdown cells and callout boxes, minus go-deeper
    boxes, <details> blocks and every 🔴 Optional part (its heading up to the next heading of the same or a
    higher level)."""
    total, optional_level = 0, 0
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        src, kind = cell.source, callout_kind(cell.source)
        if kind == "go-deeper":
            continue
        if kind is not None:
            first, _, body = src.lstrip("\n").partition("\n")
            src = f"{(parse_marker(first) or (kind, None))[1] or ''}\n{body}"
        lines, fence = [], False
        for line in DETAILS_RE.sub(" ", src).splitlines():
            if line.lstrip().startswith("```"):
                fence = not fence
            m = None if fence or kind is not None else HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                if optional_level and level <= optional_level:
                    optional_level = 0
                if "🔴" in m.group(2) and "optional" in m.group(2).lower():
                    optional_level = level
            if not optional_level:
                lines.append(line)
        total += count_words("\n".join(lines))
    return total


def check_word_budget(num: int, cells) -> None:
    """A WARNING (never a FAIL) when a section shows more words than its budget."""
    words, budget = visible_words(cells), WORD_BUDGETS.get(num)
    if budget is not None and words > budget:
        warnings.append(
            f"WARNING words section {num}: {words} visible words, budget {budget} (+{words - budget})"
        )


def yourcode_stubs(src: str) -> list[ast.FunctionDef]:
    """The top-level functions of a '# ✍️ YOUR CODE' cell (the stubs the learner fills)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


def spec_rows() -> dict[str, str]:
    """sections/NN_x.py -> status, from SPEC.md's sections table."""
    if not SPEC.exists():
        fail("SPEC.md", "missing")
        return {}
    rows = {}
    for line in SPEC.read_text().splitlines():
        m = re.match(r"^\|\s*\d+\s*\|\s*(sections/\d\d_[\w-]+\.py)\s*\|.*\|\s*([\w-]+)\s*\|\s*$", line)
        if m:
            rows[m.group(1)] = m.group(2).lower()
    return rows


# Notebooks made with Susanne (SPEC "**With:** Susanne Suter"): her overlay and her own edit (2026-10-28) win.
# She removed the objective lines (SN-003) and chose a box plot herself: N-003's objective and N-040 are off.
WITH_SUSANNE = "**With:** Susanne Suter" in (ROOT / "SPEC.md").read_text()


def main() -> int:
    files = sorted(SECTIONS.glob("[0-9][0-9]_*.py"))
    on_disk = {f"sections/{p.name}" for p in files}
    rows = spec_rows()
    listed = set(rows)
    for missing in sorted(listed - on_disk):
        if rows[missing] != "todo":  # unbuilt sections may be missing
            fail(missing, f"status '{rows[missing]}' in SPEC.md but the file does not exist (N-016)")
    for extra in sorted(on_disk - listed):
        fail(extra, "file exists but is not listed in SPEC.md sections table (N-016)")

    expected_q = 1
    smoke_defined = False
    for path in files:
        rel = f"sections/{path.name}"
        nb = jupytext.read(path, fmt="py:percent")
        cells = nb.cells
        check_word_budget(int(path.name[:2]), cells)
        if not cells or cells[0].cell_type != "markdown" or not cells[0].source.lstrip().startswith("#"):
            fail(rel, "first cell must be a markdown heading (N-003)")
        elif (
            not WITH_SUSANNE
            and "**Objective:**" not in cells[0].source
            and not (
                len(cells) > 1
                and cells[1].cell_type == "markdown"
                and callout_kind(cells[1].source) == "objective"
            )
        ):  # the Objective may also be a 🎯 objective callout right under the heading (N-044)
            fail(rel, "first markdown cell needs an '**Objective:**' line or an objective callout (N-003)")

        for i, cell in enumerate(cells):
            where = f"{rel} cell {i}"
            src = cell.source
            if cell.cell_type == "code":
                if re.search(r"^\s*SMOKE\s*=", src, re.M):
                    smoke_defined = True
                if src.lstrip().startswith("!pip"):
                    fail(where, "use %pip, not !pip (N-021)")
                if not CODE_START_RE.match(src.lstrip()):
                    fail(where, "run-only cell must be collapsed with '# @title <what it does>' (N-035)")
                if BOXPLOT_RE.search(src) and not WITH_SUSANNE:
                    fail(where, "box plot: draw distributions as violin plots (N-040)")
                n_lines = len(src.strip().splitlines())
                if (
                    n_lines >= HELPER_MIN_LINES
                    and is_helper_cell(src)
                    and not src.lstrip().startswith("# @title")
                ):
                    fail(
                        where,
                        f"helper-only cell ({n_lines} lines) needs '# @title ...' to collapse (N-008)",
                    )
                if src.lstrip().startswith("# ✍️ YOUR CODE"):
                    for fn in yourcode_stubs(src):
                        doc = ast.get_docstring(fn) or ""
                        if not re.search(r"\be\.g\.", doc):
                            fail(where, f"✍️ {fn.name} docstring needs an example return 'e.g. …' (N-050)")
                continue
            check_callout(where, src)
            if re.search(r"\bstyle\s*=", src):
                fail(where, "markdown uses style= — Colab strips it; use a callout cell (N-025, N-034)")
            for line in src.splitlines():
                for q in forward_question_refs(line, expected_q):
                    warnings.append(f"WARNING {where}: mentions Q{q} before it is asked (N-052)")
                for m in Q_RE.finditer(line):
                    q = int(m.group(1))
                    if q != expected_q:
                        fail(where, f"question Q{q} out of sequence, expected Q{expected_q} (N-006)")
                    expected_q = q + 1
                    if not any(lv in line for lv in LEVELS):
                        fail(where, f"Q{q} has no level marker 🟢/🟡/🔴 on its line (N-005)")
                    if "```python" in src:
                        nxt = next((c for c in cells[i + 1 :] if c.cell_type == "code"), None)
                        if nxt is None or not re.match(rf"# (@title )?Q{q}\b", nxt.source.lstrip()):
                            fail(
                                where,
                                f"Q{q} shows code: next code cell must start with '# Q{q}' "
                                f"or '# @title Q{q} …' (N-007)",
                            )

    if files and not smoke_defined:
        fail("sections", "no cell defines SMOKE = ... (N-014)")

    for p in problems:
        print(p)
    for w in warnings:
        print(w)
    print(f"lint: {len(files)} sections, {expected_q - 1} questions, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
