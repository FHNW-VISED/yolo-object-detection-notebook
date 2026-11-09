"""Run copy for the hand-back screenshots only: sliders drawn at fixed values (static HTML has no widgets),
Mac GPU instead of CUDA, optional experiments on. The notebook itself is not changed."""
import sys
import nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
PATCH = '''import contextlib
class _StaticOutput(contextlib.nullcontext):
    def _repr_html_(self):
        return ''
widgets.Output = lambda *a, **k: _StaticOutput()
def _static_interact(f, **kw):
    return f(**{k: w.value for k, w in kw.items()})
widgets.interact = _static_interact
widgets.HBox = lambda *a, **k: None
DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE
print(f"(hand-back run: device {DEVICE})")'''
out = []
for c in nb.cells:
    if c.cell_type == "code" and "# ✏️ YOUR TURN: run the optional experiments?" in c.source:
        c.source = c.source.replace("RUN_EXPERIMENTS = False", "RUN_EXPERIMENTS = True")
    if c.cell_type == "code" and "def show_threshold(" in c.source:
        c.source += "\nshow_threshold(0.7, 0)"
    if c.cell_type == "code" and "def show_nms(" in c.source:
        c.source += "\nshow_nms(0.95)\nshow_nms(0.3)"
    out.append(c)
    if c.cell_type == "code" and c.source.lstrip().startswith("# @title Setup (run me first)"):
        out.append(nbformat.v4.new_code_cell(PATCH))
nb.cells = out
nbformat.write(nb, sys.argv[2])
