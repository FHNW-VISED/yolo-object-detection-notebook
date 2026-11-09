"""Write spec.json for shots.py: each shot finds its cells by a piece of source text (no hard-coded indices)."""
import json
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
ORIG = Path(sys.argv[1])  # her original, with outputs (rendered as before.html)
NB = {"before.html": nbformat.read(ORIG, 4), "after.html": nbformat.read(HERE / "after.ipynb", 4)}


def idx(html, text, after=0):
    for i, c in enumerate(NB[html].cells):
        if i >= after and text in c.source:
            return i
    sys.exit(f"not found in {html}: {text!r}")


def shot(html, out, first, last=None, marks=(), max_h=1600):
    a = idx(html, first)
    b = idx(html, last, a) if last else a
    ms = [{"cell": idx(html, m[0], a), "text": m[1], "label": m[2], "open": len(m) > 3} for m in marks]
    return {"html": html, "cells": [a, b], "out": out, "marks": ms, "max_h": max_h}


B, A = "before.html", "after.html"
SPECS = [
    shot(B, "b-setup.png", "The current configuration runs only on Python 3.11", "pip install torch==2.5.0",
         [("The current configuration", "2025.07", "1"), ("pip install torch==2.5.0", "torch==2.5.0", "2")]),
    shot(B, "b-clone.png", "#### Download YOLO Repository", "git clone https://github.com/WongKinYiu",
         [("git clone https://github.com/WongKinYiu", "git clone", "3")]),
    shot(A, "a-setup.png", "### Setup", "# @title Setup (run me first)", [("### Setup", "installs Ultralytics", "1")]),
    shot(A, "a-models.png", "YOLO11 comes in five sizes", "YOLO11 comes in five sizes", [("YOLO11 comes in five sizes", "YOLO11n", "3")]),
    shot(B, "b-data.png", "For this exercise, we are working with the cattle dataset", "For this exercise, we are working",
         [("For this exercise, we are working", "cattle dataset", "1")]),
    shot(A, "a-data.png", "A drone flies low over a field", "A drone flies low over a field",
         [("A drone flies low over a field", "spray only those spots", "1")]),
    shot(A, "a-gt.png", "# @title 🔎 Browse Training Images", "# @title 🔎 Browse Training Images", max_h=700),
    shot(A, "a-coco.png", "### 🟢 Before Fine-Tuning: What Does a COCO Model See?", "# @title 🔎 Predict with the COCO",
         [("### 🟢 Before Fine-Tuning", "What Does a COCO Model See?", "2")], max_h=1300),
    shot(B, "b-explore.png", "### Analyze the Data Qualitatively", "### Analyze the Data Qualitatively",
         [("### Analyze the Data Qualitatively", "Analyze the images and labels by visual inspection", "1")]),
    shot(B, "b-explore2.png", "### Analyze the Data Quantitatively", "### Analyze the Data Quantitatively",
         [("### Analyze the Data Quantitatively", "Analyze the images and labels using code", "2")]),
    shot(A, "a-explore.png", "A drone flies low over a field", "A drone flies low over a field",
         [("A drone flies low over a field", "Here we only look briefly", "1")]),
    shot(B, "b-train.png", "# Task: choose training params", "--name yolov7x-cattle\n",
         [("# Task: choose training params", "batch_size", "1"), ("!python train.py", "train.py", "2")]),
    shot(A, "a-train.png", "### 🟢 Training Settings: How Long and How Big?", "# ✏️ YOUR TURN: training settings",
         [("### 🟢 Training Settings", "How Long and How Big?", "1")]),
    shot(A, "a-curves.png", "# @title 🔎 Plot the Training Curves", "# @title 🔎 Plot the Training Curves",
         [("# @title 🔎 Plot the Training Curves", None, "2")], max_h=700),
    shot(B, "b-iou.png", "Check the labels and predictions of the sample image.", "def calculate_iou(box1, box2):\n\n",
         [("Check the labels and predictions", "Calculate the IOU", "1"), ("#TODO: continue here", "TODO", "2")]),
    shot(A, "a-tpfp.png", "### 🟢 True Positives, False Positives, False Negatives: What Went Wrong?",
         "# @title 🔎 Browse Predictions vs. Ground Truth",
         [("### 🟢 True Positives, False Positives", "Find one example of each", "1")], max_h=1500),
    shot(A, "a-pr.png", "### 🟡 The Confidence Threshold: Precision or Recall?", "# @title 🔎 Explore the Confidence Threshold",
         [("### 🟡 The Confidence Threshold", "Precision or Recall?", "1")], max_h=1700),
    shot(A, "a-ap.png", "# @title 🔎 Compute AP per Class", "# @title 🔎 AP at Ten IoU Thresholds",
         [("# @title 🔎 Compute AP per Class", None, "2"), ("# @title 🔎 AP at Ten IoU Thresholds", None, "3")], max_h=1700),
    shot(A, "a-nms.png", "# @title 🔎 Explore the NMS Threshold", "# @title 🔎 Explore the NMS Threshold", max_h=1900),
    shot(B, "b-tiny.png", "### Optional Exercise: Download and Train Tiny YOLO Model", "!python train.py --device 0 --workers 8 --batch-size 16 --epochs 200",
         [("### Optional Exercise: Download and Train Tiny", "Tiny YOLO", "1")]),
    shot(A, "a-ns.png", "### 🔴 YOLO11n vs. YOLO11s", "# @title 🔎 Train YOLO11s and Compare",
         [("### 🔴 YOLO11n vs. YOLO11s", "Is the Larger Model Worth It?", "1")], max_h=1500),
]
(HERE / "img").mkdir(exist_ok=True)
(HERE / "spec.json").write_text(json.dumps(SPECS, indent=1))
print(len(SPECS), "shots")
