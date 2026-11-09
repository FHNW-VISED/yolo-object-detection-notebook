"""Render a notebook the way Colab shows it: form cells collapsed to their title, outputs kept. → HTML."""
import re
import subprocess
import sys
from pathlib import Path

import nbformat

src, out = Path(sys.argv[1]), Path(sys.argv[2])
nb = nbformat.read(src, as_version=4)
for c in nb.cells:
    if c.cell_type != "code":
        continue
    m = re.search(r"^\s*#\s*@title\s*(.*?)\s*(\{.*\})?\s*$", c.source, re.M)
    if m:  # Colab shows a form cell as its title only
        c.source = f"# ▸ {m[1] or '(form cell)'}"
    c.metadata["cell_index"] = nb.cells.index(c)
for i, c in enumerate(nb.cells):
    c.metadata["cell_index"] = i
    merged = []
    for o in c.get("outputs", []):  # the runner stores every stream message apart: merge them as Colab does
        if o.get("output_type") == "stream" and merged and merged[-1].get("output_type") == "stream" \
                and merged[-1]["name"] == o["name"]:
            merged[-1]["text"] += o["text"]
        else:
            merged.append(o)
    for o in merged:
        if o.get("output_type") == "stream":  # Colab overwrites a line on \r: keep what remains visible
            o["text"] = "\n".join(next((part for part in reversed(line.split("\r")) if part.strip()), "")
                                  for line in o["text"].split("\n"))
    if c.cell_type == "code":
        c.outputs = merged
tmp = out.with_suffix(".colab.ipynb")
nbformat.write(nb, tmp)
subprocess.run(["jupyter", "nbconvert", "--to", "html", "--template", "lab", "--no-prompt", str(tmp),
                "--output-dir", str(out.parent), "--output", out.stem], check=True, capture_output=True)
print("rendered", out)
