# %% [markdown]
# # Object Detection with YOLO11
# ## Fine-Tuning a Pretrained Detector and Reading Its Metrics
#
# Simone Bonato & Susanne Suter, FHNW
#
# Object detection means finding the individual objects in an image and marking each one with a class label and a bounding box. In this notebook, we fine-tune a pretrained YOLO11 model on a small dataset from outside the everyday world of COCO, look at what it gets right and wrong, and learn to read the metrics that every detection paper reports.
#
# The notebook complements *Know Your Images*, where you explored a detection dataset before training. Here we train.
#
# *The best practices of Andrej Karpathy's [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) apply here as well, even if this notebook does not follow them step by step.*

# %% [markdown]
# **A note about this notebook**
#
# This notebook was developed with AI assistance and subsequently reviewed, edited, and extended by the authors. While we have made every effort to ensure its accuracy, not every detail could be thoroughly verified.
#
# If you encounter errors, unexpected behavior, or anything that seems incorrect or confusing, please let us know.

# %%
# @title Learning Objectives
from IPython.display import HTML

HTML(r"""
<div class="callout callout-banner"
     style="background:#E5E7EB;color:#374151;border-left:6px solid #6B7280;
            border-radius:6px;padding:12px 14px;line-height:1.5;
            font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;
            font-size:14px;width:100%;box-sizing:border-box;
            white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-size:22px;font-weight:bold;line-height:1.3;">Learning Objectives: Object Detection with YOLO11</div>
<p style="margin:6px 0 0 0;"><strong>At the end of this notebook you will be able to:</strong></p>
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;">Fine-tune a pretrained YOLO model on a custom dataset and explain the main training settings.</li>
<li style="margin:2px 0;">Classify detections as true positives, false positives and false negatives, and explain how the
confidence and NMS thresholds change them.</li>
<li style="margin:2px 0;">Interpret a precision–recall curve, AP, mAP@50 and mAP@50–95, and compare two model sizes with them.</li>
</ol>
<p style="margin:8px 0 0 0;">Tasks are marked beginner 🟢, intermediate 🟡, advanced 🔴 (optional).</p>
</div>
""")

# %% [markdown]
# **What Do You Need to Know for the Exam and Your Project?**
#
# Focus on the concepts and decisions: what fine-tuning changes, what makes a detection a true or false positive, how the two thresholds trade precision against recall, and what each metric rewards. You do **not** need to memorize the numbers of this dataset or the Ultralytics function arguments.

# %%
# @title How to Use This Notebook
from IPython.display import HTML

HTML(r"""
<div class="callout" style="background:#F5F5F5;color:#374151;border-left:6px solid #9CA3AF;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">How to Use This Notebook</div>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;">Make a copy (File → Save a copy in Drive), then select a GPU: Runtime → Change runtime type → T4 GPU.</li>
<li style="margin:2px 0;">Run the cells in order. Training the default model takes a few minutes on a T4; keep reading while it runs.</li>
<li style="margin:2px 0;">Answer each task before you open its 💡 Sample Answer.</li>
<li style="margin:2px 0;">The 🔴 experiments at the end are optional and train two more models.</li>
</ul>
</div>
""")

# %% [markdown]
# ### Setup
#
# The next cell installs Ultralytics (the library that ships YOLO11), checks for a GPU, and loads the helpers the notebook uses. It takes about a minute the first time.

# %%
# @title Setup (run me first) { display-mode: "form" }
# The common imports are made here, once (later cells use these names).
import base64
import os
import random
import sys
import time
from io import BytesIO
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
SMOKE = os.environ.get("NB_SMOKE") == "1"  # automated tests: tiny data, CPU, seconds. Never set this by hand.
SEED = 42

if IN_COLAB:  # Jupytext turns the commented magic below into a real `%pip install` (N-021)
    pass
    # %pip install -q ultralytics==8.4.208

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import HTML, clear_output, display
from IPython.display import Image as IPythonImage
from PIL import Image
from ultralytics import YOLO

random.seed(SEED)
np.random.seed(SEED)

DEVICE = 0 if torch.cuda.is_available() else "cpu"  # 0 = the first GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. Training will be very slow: Runtime → Change runtime type → T4 GPU, then run again.")

RUNS_DIR = Path("runs")  # Ultralytics writes weights and plots here

# --- Theme (SPEC L-004): one meaning per color in every figure ---
TP_COLOR, FP_COLOR, FN_COLOR = "#2E7D32", "#EF6C00", "#C62828"  # found · invented · missed
GT_COLOR = "#FFFFFF"  # ground truth: white dashed outline
INK = "#21434A"
mpl.rcParams.update({"figure.facecolor": "white", "axes.edgecolor": "#B0B0B0", "axes.grid": True,
                     "grid.color": "#E5E7EB", "axes.titlecolor": INK, "axes.labelcolor": INK})


def callout(kind: str, title: str, body_html: str) -> HTML:
    """One box as HTML. kind: "answer" (💡 a gray toggle, closed until opened), "theory" (📘 blue),
    "recipe" (🧾 yellow) or "deeper" (🔍 gray)."""
    background, ink, bar = {
        "answer": ("#F5F5F5", "#374151", "#9CA3AF"),
        "theory": ("#E3F2FD", "#0D47A1", "#0D47A1"),
        "recipe": ("#FFF8D6", "#333333", "#F5C400"),
        "deeper": ("#F3F4F6", "#4B5563", "#4B5563"),
    }[kind]
    style = (f"background:{background};color:{ink};border-left:6px solid {bar};border-radius:6px;"
             "padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;"
             "font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;"
             "margin:2px 0;")
    if kind == "answer":
        inner = (f'<details style="margin:0;">\n<summary style="cursor:pointer;font-weight:bold;">{title}</summary>'
                 f"\n{body_html}\n</details>")
    else:
        inner = f'<div style="font-weight:bold;">{title}</div>\n{body_html}'
    return HTML(f'<div class="callout callout-{kind}" style="{style}">\n{inner}\n</div>')


def show(fig) -> None:
    """Draw a figure as a PNG in the output (the same in Colab, Jupyter and the automated tests)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(IPythonImage(data=buf.getvalue()))


def show_live(fig) -> None:
    """Draw a figure inside a slider or Previous / Next view (Colab renders it as an image)."""
    display(fig)
    plt.close(fig)


def fig_to_html(fig, width: int = 100) -> str:
    """A matplotlib figure as an inline <img> (used inside Sample Answers)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    png = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{png}" style="width:{width}%;height:auto;margin:8px 0;">'


print(f"Colab: {IN_COLAB} · smoke mode: {SMOKE} · device: {DEVICE}")
