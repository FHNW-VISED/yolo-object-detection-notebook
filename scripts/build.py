"""Assemble sections/*.py (Jupytext percent format) into the single student notebook.

Agents and humans edit only the small section files; the .ipynb files are build artifacts.
Usage: uv run python scripts/build.py            -> writes both Editions (outputs stripped)

Editions (PRD 2026-10-28, Q4): one ordered list of cells, two notebooks. A cell without `edition` metadata is
in
both; `# %% edition="teaching"` or `# %% edition="own"` keeps it in one. Never edit an Edition on its own.

Callouts (SPEC L-001, RULES N-034): a markdown cell whose first line is
`<!-- callout: KIND | TITLE -->` becomes a collapsed code cell that redraws a coloured HTML box, with the
same HTML saved as its output so the box shows in Colab before anything runs.

Her boxes (2026-10-28, C-12) are code cells that call the notebook's own `callout(kind, title, body)`.
When such a cell is a `# @title` line plus one `callout(...)` call with literal arguments (no `fig=`), the
build runs the notebook's `callout` (taken from the sections source) and saves its HTML as the cell's
output, for the same reason. A cell that only imports and shows one literal `HTML(...)` box (Learning
Objectives, How to Use) gets that HTML saved as its output too.
"""

from __future__ import annotations

import ast
import base64
import io
import re
import sys
from pathlib import Path

import jupytext
import markdown
import nbformat

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
NAME = "yolo-object-detection-notebook"
OUT = ROOT / f"{NAME}.ipynb"
# Edition id -> notebook file at the repo root. The Teaching Edition keeps the repo's original file name.
EDITIONS: dict[str, str] = {
    "teaching": f"{NAME}.ipynb",
}

# ---- callouts (SPEC L-001) ---------------------------------------------------------------------
# kind id -> (emoji, default title, background, text and left-bar colour)
CALLOUT_KINDS: dict[str, tuple[str, str, str, str]] = {
    "theory": ("📘", "Theory", "#E3F2FD", "#0D47A1"),
    "your-turn": ("✏️", "Your turn", "#FFF4D6", "#7A4A00"),
    "questions": ("❓", "Questions", "#E0F7F4", "#00564D"),
    "across-tasks": ("🔀", "Across tasks", "#EDE7F6", "#311B92"),
    "pitfall": ("⚠️", "Pitfall", "#FDECEA", "#8B1A10"),
    "recipe": ("🧾", "Recipe", "#E8F5E9", "#1B5E20"),
    "objective": ("🎯", "Objective", "#B9E3EA", "#08505C"),
    "takeaway": ("✅", "Takeaway", "#E2F0C4", "#3A5A0B"),
    "go-deeper": ("🔍", "Go deeper", "#F3F4F6", "#4B5563"),
    "answer": ("💡", "Worked answer", "#F6EFE4", "#5C4027"),
    "banner": ("", "", "#0B6E79", "#FFFFFF"),
}
BANNER_BAR = "#06464D"  # darker teal left bar on the white-on-teal banner
# A marker line is exactly `<!-- callout: KIND -->` or `<!-- callout: KIND | TITLE -->`.
MARKER_RE = re.compile(r"^<!--\s*callout:\s*(?P<kind>[a-z][a-z-]*)\s*(?:\|\s*(?P<title>.*?)\s*)?-->\s*$")
MARKER_HINT = "<!-- callout"  # anything containing this must be a valid marker on the cell's first line

FONT = "'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif"
MONO = "'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace"
MD_EXTENSIONS = ["fenced_code", "tables", "sane_lists", "md_in_html"]
# Inline styles for tags inside the box (outputs allow inline CSS; markdown cells do not).
TAG_STYLES = {
    "p": "margin:6px 0 0 0;",
    "ul": "margin:6px 0 0 0;padding-left:22px;",
    "ol": "margin:6px 0 0 0;padding-left:22px;",
    "li": "margin:2px 0;",
    "a": "color:inherit;text-decoration:underline;",
    "code": f"font-family:{MONO};font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;"
    "border-radius:3px;",
    "pre": f"font-family:{MONO};font-size:0.92em;background:rgba(0,0,0,0.07);padding:8px 10px;"
    "border-radius:4px;margin:6px 0 0 0;white-space:pre-wrap;",
    "blockquote": "margin:6px 0 0 0;padding-left:10px;border-left:3px solid currentColor;",
    "details": "margin:6px 0 0 0;",
    "summary": "cursor:pointer;font-weight:bold;",
    "table": "border-collapse:collapse;margin:6px 0 0 0;",
    "th": "border:1px solid rgba(0,0,0,0.2);padding:3px 8px;white-space:normal;text-align:left;",
    "td": "border:1px solid rgba(0,0,0,0.2);padding:3px 8px;white-space:normal;",
    "h1": "font-size:1.2em;margin:8px 0 0 0;",
    "h2": "font-size:1.15em;margin:8px 0 0 0;",
    "h3": "font-size:1.1em;margin:8px 0 0 0;",
    "h4": "font-size:1em;margin:8px 0 0 0;",
}
TAG_RE = re.compile(r"<(" + "|".join(TAG_STYLES) + r")(\s[^>]*)?>")


def parse_marker(first_line: str) -> tuple[str, str | None] | None:
    """(kind, title or None) if the line is a callout marker, else None. Does not check the kind."""
    m = MARKER_RE.match(first_line.strip())
    if not m:
        return None
    return m.group("kind"), (m.group("title") or None)


def _style_tags(body: str) -> str:
    def add(m: re.Match) -> str:
        tag, attrs = m.group(1), m.group(2) or ""
        if tag == "a" and not re.search(r"\btarget\s*=", attrs):
            attrs += ' target="_blank" rel="noopener"'  # links in an output frame open in a new tab
        if re.search(r"\bstyle\s*=", attrs):
            return f"<{tag}{attrs}>"
        return f'<{tag}{attrs} style="{TAG_STYLES[tag]}">'

    body = body.replace("<pre><code", '<pre><code style="background:none;padding:0;"')
    return TAG_RE.sub(add, body)


def _md_to_html(text: str) -> str:
    # md_in_html renders markdown inside <details> only with markdown="1"; add it so authors need not.
    text = re.sub(r"<details(?![^>]*\bmarkdown\s*=)", '<details markdown="1"', text)
    return markdown.markdown(text, extensions=MD_EXTENSIONS)


def callout_html(kind: str, title: str | None, body_md: str) -> str:
    """The self-contained HTML box for one callout (inline CSS only)."""
    emoji, default_title, bg, fg = CALLOUT_KINDS[kind]
    title = title if title is not None else default_title
    title_html = re.sub(r"^<p>(.*)</p>$", r"\1", _md_to_html(title).strip(), flags=re.S) if title else ""
    title_html = _style_tags(title_html)
    body = _style_tags(_md_to_html(body_md).strip())
    if kind == "answer":  # N-024: a worked answer is a toggle, closed until the learner opens it
        summary = _style_tags(f"<summary>{emoji} {title_html}</summary>")
        body = f'<details style="margin:0;">{summary}\n{body}\n</details>'
        title_html = ""
    bar = BANNER_BAR if kind == "banner" else fg
    box = (
        f"background:{bg};color:{fg};border-left:6px solid {bar};border-radius:6px;padding:12px 14px;"
        f"line-height:1.5;font-family:{FONT};font-size:14px;width:100%;box-sizing:border-box;"
        "white-space:normal;overflow-wrap:break-word;margin:2px 0;"
    )
    if kind == "banner":
        head = f'<div style="font-size:22px;font-weight:bold;line-height:1.3;">{title_html}</div>'
    elif title_html:
        head = f'<div style="font-weight:bold;">{(emoji + " ") if emoji else ""}{title_html}</div>'
    else:
        head = ""
    return f'<div class="callout callout-{kind}" style="{box}">\n{head}\n{body}\n</div>'


def callout_cell(kind: str, title: str | None, body_md: str) -> nbformat.NotebookNode:
    html = callout_html(kind, title, body_md)
    safe = html.replace('"""', "&quot;&quot;&quot;")  # cannot close the raw string below
    cell = nbformat.v4.new_code_cell(
        f'# @title\nfrom IPython.display import HTML; HTML(r"""{safe}""")',
        outputs=[
            nbformat.v4.new_output(
                "display_data",
                data={"text/html": safe, "text/plain": f"<callout: {kind}>"},
                metadata={},
            )
        ],
    )
    cell.execution_count = None
    cell.metadata["cellView"] = "form"
    cell.metadata["callout"] = kind
    return cell


def convert_callout(cell: nbformat.NotebookNode, where: str) -> nbformat.NotebookNode:
    """Return the callout code cell for a marked markdown cell, or the cell unchanged."""
    if cell.cell_type != "markdown" or MARKER_HINT not in cell.source:
        return cell
    first, _, body = cell.source.lstrip("\n").partition("\n")
    parsed = parse_marker(first)
    if parsed is None or cell.source.count(MARKER_HINT) > 1:
        sys.exit(f"{where}: malformed callout marker, expected '<!-- callout: KIND | TITLE -->' as line 1")
    kind, title = parsed
    if kind not in CALLOUT_KINDS:
        sys.exit(f"{where}: unknown callout kind '{kind}' (known: {', '.join(CALLOUT_KINDS)})")
    return callout_cell(kind, title, body)


# ---- her callout() cells: saved output (N-034) ---------------------------------------------------


class _Html:
    """Stands in for IPython.display.HTML while the build runs the notebook's callout()."""

    def __init__(self, data: str):
        self.data = data


def notebook_callout(sections: Path = SECTIONS):
    """The `callout` function defined in the sections (top level of a section file), or None."""
    for path in section_files(sections):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name == "callout":
                ns = {"HTML": _Html, "BytesIO": io.BytesIO, "base64": base64}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
                return ns["callout"]
    return None


def prerender_callout(cell: nbformat.NotebookNode, callout) -> None:
    """Save the box as the output of a cell that is only `# @title …` + one literal `callout(...)` call."""
    if callout is None or cell.cell_type != "code" or "callout(" not in cell.source:
        return
    try:
        tree = ast.parse(cell.source)
    except SyntaxError:
        return
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return
    call = tree.body[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "callout"):
        return
    if call.keywords:  # fig=… is drawn at run time
        return
    try:
        args = [ast.literal_eval(a) for a in call.args]
    except ValueError:  # an f-string or a variable: only the kernel can draw it
        return
    html = callout(*args).data
    cell.outputs = [
        nbformat.v4.new_output(
            "display_data", data={"text/html": html, "text/plain": f"<callout: {args[0]}>"}, metadata={}
        )
    ]


def prerender_html(cell: nbformat.NotebookNode) -> None:
    """Save the box as the output of a cell that only imports and then shows one HTML(...) of a string literal
    (Learning Objectives, How to Use), so it shows in Colab before the cell runs (N-034), like a callout."""
    if cell.cell_type != "code" or "HTML(" not in cell.source:
        return
    try:
        *head, last = ast.parse(cell.source).body
    except (SyntaxError, ValueError):
        return
    if not all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in head):
        return
    call = last.value if isinstance(last, ast.Expr) else None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "HTML"
            and len(call.args) == 1 and not call.keywords):
        return
    try:
        html = ast.literal_eval(call.args[0])
    except ValueError:  # an f-string or a variable: only the kernel can draw it
        return
    if isinstance(html, str):
        cell.outputs = [
            nbformat.v4.new_output(
                "display_data", data={"text/html": html, "text/plain": "<IPython.core.display.HTML object>"},
                metadata={},
            )
        ]


# ---- assembly ----------------------------------------------------------------------------------


def section_files(sections: Path = SECTIONS) -> list[Path]:
    files = sorted(p for p in sections.glob("[0-9][0-9]_*.py"))
    if not files:
        sys.exit("no sections/NN_*.py files found")
    return files


def read_cells(sections: Path = SECTIONS) -> list[nbformat.NotebookNode]:
    """Every cell of every section, in order, with its section stamped and its callout converted."""
    cells = []
    callout = notebook_callout(sections)
    for path in section_files(sections):
        nb = jupytext.read(path, fmt="py:percent")
        for k, cell in enumerate(nb.cells):
            where = f"sections/{path.name} cell {k}"
            edition = cell.metadata.get("edition")
            if edition is not None and edition not in EDITIONS:
                sys.exit(f"{where}: unknown edition '{edition}' (known: {', '.join(EDITIONS)}; none = both)")
            if cell.cell_type == "code":  # inserting a cell after a code cell can leave a trailing blank line
                cell.source = cell.source.rstrip()
            cell = convert_callout(cell, where)
            prerender_callout(cell, callout)
            prerender_html(cell)
            if edition is not None:
                cell.metadata["edition"] = edition
            cell.id = f"{path.stem[:50]}-{k}"  # deterministic ids: same sections -> byte-identical .ipynb
            cell.metadata.pop("lines_to_next_cell", None)
            cell.metadata["section"] = path.stem
            # Colab collapses a code cell whose first line is "# @title ..." into a form.
            if cell.cell_type == "code" and cell.source.lstrip().startswith("# @title"):
                cell.metadata["cellView"] = "form"
            cells.append(cell)
    return cells


def own_wording(sections: Path) -> list[tuple[str, str, str]]:
    """(section file, old, new) rows of sections/own_wording.tsv: words of shared cells that the Own-Data
    Edition says differently ("animal" → "object"). Kept as swaps, not copied cells, so her text stays one
    source."""
    path = sections / "own_wording.tsv"
    if not path.exists():
        return []
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:] if line.strip()]
    if any(len(r) != 3 for r in rows):
        sys.exit("sections/own_wording.tsv: every row needs three tab-separated columns: section, old, new")
    return [(r[0], r[1], r[2]) for r in rows]


def apply_own_wording(cells: list[nbformat.NotebookNode], sections: Path) -> list[nbformat.NotebookNode]:
    """Copies of the Own-Data cells with the swaps applied; a swap must match once in its section."""
    swaps = own_wording(sections)
    out = [c if c.metadata.get("edition") == "teaching" else nbformat.from_dict(c) for c in cells]
    for section, old, new in swaps:
        stem = section.removesuffix(".py")
        hits = [
            c
            for c in out
            if c.metadata.get("section") == stem
            and c.metadata.get("edition") != "teaching"
            and old in c.source
        ]
        count = sum(c.source.count(old) for c in hits)
        if count != 1:
            sys.exit(
                f"sections/own_wording.tsv: '{old}' occurs {count} times in the Own-Data cells of {section} "
                "(must be once). Her text changed? Update or delete the row."
            )
        hits[0].source = hits[0].source.replace(old, new)
    return out


SKIP_MARKER_RE = re.compile(r"^# needs: (boxes|masks)[ \t]*\n?", re.M)


def strip_skip_markers(cells: list[nbformat.NotebookNode]) -> list[nbformat.NotebookNode]:
    """Copies without the `# needs: boxes|masks` lines: they drive the Own-Data skip hook and mean nothing in
    the Teaching Edition, where every chapter runs."""
    out = []
    for c in cells:
        if c.cell_type == "code" and SKIP_MARKER_RE.search(c.source):
            c = nbformat.from_dict(c)
            c.source = SKIP_MARKER_RE.sub("", c.source)
        out.append(c)
    return out


def write_edition(cells: list[nbformat.NotebookNode], edition: str, out: Path) -> Path:
    keep = [c for c in cells if c.metadata.get("edition", edition) == edition]
    nb = nbformat.v4.new_notebook(cells=keep)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


def build_all(sections: Path = SECTIONS, out_dir: Path = ROOT) -> dict[str, Path]:
    cells = read_cells(sections)
    per_edition = {"teaching": strip_skip_markers(cells)}
    return {ed: write_edition(per_edition[ed], ed, out_dir / name) for ed, name in EDITIONS.items()}


def build() -> Path:
    """Both Editions; returns the Teaching Edition's path (what older callers expect)."""
    return build_all()["teaching"]


if __name__ == "__main__":
    outs = build_all()
    print(f"built {', '.join(p.name for p in outs.values())}: {len(section_files())} sections")
