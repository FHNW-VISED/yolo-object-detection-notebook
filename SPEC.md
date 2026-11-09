# SPEC — Object Detection with YOLO

**With:** Susanne Suter

The single source of truth for **what** this notebook is. `AGENTS.md` says how to work,
`RULES.md` says what every notebook must respect, and this file says what *this* one must do.
Written 2026-11-09 from Susanne's written brief (paraphrased below) and her older YOLOv7 workbook, kept
without outputs in `reference/original_yolov7.py`. Agents never change it, except the `Status` column of §6.

## 1. Purpose
A Colab lab in Susanne Suter's deep-learning / image-processing course. Students **fine-tune a pretrained
YOLO detector on a small non-COCO dataset, look at what it gets right and wrong, and learn to read the
standard detection metrics**. It modernizes her YOLOv7 workbook: same storyline (setup → data → train →
predict → TP/FP/FN → optional experiments), same kind of student tasks with sample solutions, on the current
Ultralytics Python API. It is the companion of the EDA notebook *Know Your Images* and does **not** repeat
its dataset exploration (that notebook has the qualitative and quantitative inspection and the optional IoU task).
The single question it answers: **"How do I fine-tune a detector, and how do I tell whether its detections are good?"**

The brief, paraphrased: replace YOLOv7 with Ultralytics YOLO11 (or a better-suited model), simplified setup with
the Python API, Colab-compatible · a public dataset of ~500 images, few classes, not COCO-like, with boxes ·
minimal dataset exploration · keep the explanations and the TP/FP/FN tasks; the IoU implementation task can go ·
add a short interactive precision–recall exercise (confidence threshold, AP, mAP@50 vs mAP@50–95) · optionally
compare YOLO11n with YOLO11s · a slider for the NMS threshold if possible · runs top to bottom in a fresh Colab
runtime, American English, sample solutions kept, short training times · **keep the notebook short**.

## 2. Audience and level
BSc computer science students who know Python and machine learning, have seen her lectures on CNNs and object
detection, and have done the EDA notebook. They have not trained a detector themselves.
**Freedom (N-028): low to medium.** Mostly run-and-interpret; ✏️ change-a-value cells for the training
settings, the thresholds and the sliders. No ✍️ write-code cells on the main path (the IoU implementation was cut).

## 2b. Prerequisites
A Google account (Colab) with a GPU runtime (T4 is enough). Her lecture on object detection (IoU, NMS, AP).

## 3. Learning outcomes
These appear in the Learning Objectives banner of `00_setup.py`.
1. Fine-tune a pretrained YOLO model on a custom dataset and explain the main training settings.
2. Classify detections as true positives, false positives and false negatives, and explain how the confidence and NMS thresholds change them.
3. Interpret a precision–recall curve, AP, mAP@50 and mAP@50–95, and compare two model sizes with them.

## 4. Use case and data
Decided per **variant** (see §11). Each variant names: the dataset, its source URL and licence, splits and
classes, how it is fetched in Colab (no login), and its SMOKE subset (a few train and val images, `imgsz` 160,
one epoch on CPU). The dataset is never committed (N-019); `data/` holds only small manifests if any.

**Variant weed-crop-aerial** (drone camera, agriculture; story: a drone maps weeds for targeted spraying).
- Dataset: Roboflow 100 "weed-crop-aerial" via https://huggingface.co/datasets/LibreYOLO/weed-crop-aerial, CC BY 4.0, YOLO txt, 640 × 640 px.
- Classes: crop, weed (crop is about 5 % of the boxes, about 2 % in the validation subset). Boxes are small: median about 17 × 27 px, √area about 21 px. Training at 640 px; larger `imgsz` would help the smallest weeds but costs time.
- Splits in the source: 823 train / 235 valid / 118 test. We use a fixed random sample (seed 42) of 500 train and 150 valid images; the test split is not used.
- Fetch in Colab (no login): `huggingface_hub.snapshot_download` of only the sampled images and labels (about 53 MB, 1 to 2 minutes), then a local `data.yaml`. Idempotent: a finished download is reused.
- SMOKE subset: 8 train and 4 valid images through the same code, `imgsz` 160, one epoch on CPU.

## 5. Environment
| Item | Value |
|---|---|
| Runs on | Google Colab (web). Local Jupyter only for development |
| Colab runtime | T4 GPU (CPU works with the smoke-sized settings, but training is slow) |
| Python | 3.12 |
| Packages not preinstalled in Colab | `ultralytics` (pinned, see `pyproject.toml`) |
| Smoke run | `NB_SMOKE=1`, CPU, budget 900 s |
| Full-run budget | training of the default path ≤ ~10 min on a T4; the whole notebook ≤ ~25 min of compute |

## 6. Sections
Time = minutes of student time. Status: todo · in-progress · done · blocked.

| # | File | Title | Objective (one line) | Level | Time | Status |
|---|---|---|---|---|---|---|
| 0 | sections/00_setup.py | Setup | Title, objectives, GPU check, install, helpers | 🟢 | 5 | done |
| 1 | sections/01_yolo.py | YOLO: A One-Stage Detector | What YOLO predicts, model sizes, why YOLO11 | 🟢 | 5 | done |
| 2 | sections/02_data.py | The Dataset | Download, YOLO label format, a short look at boxes | 🟢 | 10 | done |
| 3 | sections/03_train.py | Fine-Tune YOLO11n | Choose batch size and epochs, train, read the curves | 🟢 | 15 | done |
| 4 | sections/04_predict.py | Predictions: TP, FP and FN | Predict on validation images and classify every box | 🟢 | 15 | done |
| 5 | sections/05_pr_curve.py | Precision, Recall and AP | Confidence slider, PR curve, AP, mAP@50 vs mAP@50–95 | 🟡 | 15 | done |
| 6 | sections/06_nms.py | Non-Maximum Suppression | NMS IoU slider on a crowded image | 🟡 | 10 | done |
| 7 | sections/07_experiments.py | Optional Experiments | YOLO11n vs YOLO11s; freezing the backbone | 🔴 | 20 | done |

## 6b. Pitfalls and extensions
- No GPU selected: training is very slow. Setup says so and the training settings name a CPU fallback.
- Ultralytics numbers its run folders (`train`, `train2`, …). The notebook uses fixed `project`/`name` with
  `exist_ok=True`, and always reads results from the returned objects, so no folder hunting.
- Confidence threshold for *metrics* (Ultralytics uses 0.001 in `val`) is not the threshold for *display* (0.25).
  Section 5 makes this difference explicit.

## 7. Out of scope / limitations
Dataset exploration and the IoU implementation (both in the EDA notebook) · Weights & Biases (a go-deeper
mention only) · Google Drive (optional tip) · export / deployment · YOLO26 (one go-deeper note: it is NMS-free).

## 8. Notebook-specific rules
- L-001 — **Short.** ≤ ~95 student minutes on the default path; one idea per section; prefer one good figure over three. — her brief ("keep the notebook short").
- L-002 — Her SN-rules govern form (Tasks with level dots and Title-Case headings, no Q-numbers, objectives as plain markdown, predict → explore → interpret, 💡 Sample Answer toggles, her box colors). Overrides N-005/N-006/N-007's Q-numbering and N-003's `**Objective:**` box. — SN-002, SN-003, SN-012…SN-015.
- L-003 — Results come from the Python API's returned objects (`model.train`, `model.val`, `model.predict`), never from parsing run folders. — modernization brief + the original's experiment-folder tasks, which broke between runs.
- L-004 — One color meaning everywhere: ground truth = white dashed outline in figures with predictions; TP = green, FP = orange, FN = red; per-class colors only where no TP/FP/FN is shown. — N-027, N-061.
- L-005 — N-033 (facilitation guide), N-066 (state-of-the-art reference) and N-068 (result cards) are satisfied with the smallest version: a one-line SOTA note in section 7 and plain small tables. — L-001.

## 9. Acceptance (definition of done for the whole notebook)
- `just check` passes, including the content assertions in `tests/test_content.py`.
- Runs top to bottom on a fresh kernel; no hidden state.
- A full (non-smoke) run in Colab on a T4 by Simone finishes with no errors; default-path training ≤ ~10 min.
- Every Task keeps a 💡 Sample Answer, written from a real full run of the chosen variant.

## 10. Open questions
- Which dataset variant goes to Susanne (§11).
- Fine-tuned fallback weights for students without a GPU: where to host them (the original used her Drive).

## 11. Variants
Built in parallel git worktrees, one branch per dataset (`variant/<dataset>`), from the same base sections.
Only `02_data.py`, the dataset-dependent texts and Sample Answers, and the default settings differ.
The comparison and the choice are recorded in the vault page `wiki/work/projects/yolo-object-detection-notebook.md`.
