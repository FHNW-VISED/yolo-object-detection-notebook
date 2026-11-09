"""Content assertions on the executed smoke notebook: does it TEACH what SPEC.md promises?

"It runs" is not "it teaches" — and an agent will satisfy a weak test without doing the work.
These assertions are written by Simone (or with him, during /notebook grill), never by the
night agent: scripts/night.sh reverts any agent edit to tests/.

Runs after tests/test_smoke.py, which writes build/smoke-executed.ipynb.
Helpers below: cells_of(section), outputs_text(section), images_in(section), source_of(section).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

EXECUTED = Path(__file__).resolve().parent.parent / "build" / "smoke-executed.ipynb"


@pytest.fixture(scope="module")
def nb():
    if not EXECUTED.exists():
        pytest.skip("no executed notebook — run tests/test_smoke.py first (just smoke)")
    return nbformat.read(EXECUTED, as_version=4)


def cells_of(nb, section: str):
    return [c for c in nb.cells if c.metadata.get("section", "").startswith(section)]


def source_of(nb, section: str) -> str:
    return "\n".join(c.source for c in cells_of(nb, section))


def outputs_text(nb, section: str) -> str:
    parts = []
    for c in cells_of(nb, section):
        for o in c.get("outputs", []):
            parts.append(o.get("text", "") or str(o.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


def images_in(nb, section: str) -> int:
    return sum(
        1 for c in cells_of(nb, section) for o in c.get("outputs", []) if "image/png" in o.get("data", {})
    )


def test_setup_states_learning_outcomes(nb):
    src = source_of(nb, "00_")
    assert "At the end of this notebook you will be able to" in src


# Examples of the assertions to write per section during grilling:
# def test_class_distribution_is_plotted(nb):
#     assert images_in(nb, "03_") >= 1
# def test_planted_duplicates_are_found(nb):
#     assert "duplicates found: 5" in outputs_text(nb, "05_")


# ---- YOLO notebook (SPEC §1, §3, §6; written 2026-11-09 from Susanne's brief) ------------------------------


def html_text(nb, section: str) -> str:
    return "\n".join(
        str(o.get("data", {}).get("text/html", ""))
        for c in cells_of(nb, section)
        for o in c.get("outputs", [])
    )


def widgets_in(nb, section: str) -> int:
    """Slider and Previous / Next views (their figures render in Colab, not in the headless run)."""
    return sum(
        1
        for c in cells_of(nb, section)
        for o in c.get("outputs", [])
        if "application/vnd.jupyter.widget-view+json" in o.get("data", {})
    )


def test_no_variant_placeholders_left(nb):
    """Every variant fills the dataset-specific texts; the base marks them with VARIANT."""
    for section in ("00_", "01_", "02_", "03_", "04_", "05_", "06_", "07_"):
        assert "VARIANT" not in source_of(nb, section), f"{section}: VARIANT placeholder left"


def test_dataset_is_loaded_and_shown(nb):
    out = outputs_text(nb, "02_")
    assert "training images" in out and "validation images" in out and "classes:" in out
    assert "class_idx x_center y_center width height" in source_of(nb, "02_")
    assert widgets_in(nb, "02_") >= 1 and images_in(nb, "02_") >= 1  # box browser, COCO-model predictions


def test_training_runs_and_curves_are_plotted(nb):
    assert "Training took" in outputs_text(nb, "03_")
    assert images_in(nb, "03_") >= 1
    src = source_of(nb, "03_")
    assert "EPOCHS" in src and "BATCH_SIZE" in src and "Sample Answer" in src  # her batch/epochs task kept


def test_tp_fp_fn_task_is_kept(nb):
    src = source_of(nb, "04_")
    for word in ("True Positive", "False Positive", "False Negative", "Sample Answer"):
        assert word.lower() in src.lower()
    assert widgets_in(nb, "04_") >= 1  # the Previous / Next browser


def test_iou_implementation_task_is_gone(nb):
    """Her brief: the IoU calculation exercise can be deleted (the EDA notebook has it)."""
    for section in ("04_", "05_", "06_"):
        assert "calculate_iou" not in source_of(nb, section)


def test_precision_recall_part(nb):
    src, out = source_of(nb, "05_"), outputs_text(nb, "05_")
    assert "FloatSlider" in src and "confidence" in src  # the interactive threshold
    assert "mAP@50, Ultralytics val" in out and "mAP@50–95, our computation" in out
    assert widgets_in(nb, "05_") >= 1 and images_in(nb, "05_") >= 2  # slider; AP areas, AP per IoU


def test_nms_slider(nb):
    src = source_of(nb, "06_")
    assert "FloatSlider" in src and "iou=nms_iou" in src
    assert widgets_in(nb, "06_") >= 1


def test_model_comparison_n_vs_s(nb):
    html = html_text(nb, "07_")
    assert "YOLO11n" in html and "YOLO11s" in html and "frozen" in html


def test_american_english(nb):
    """Her brief: American English (and SN-016)."""
    text = "\n".join(c.source for c in nb.cells)
    british_words = ("colour", "behaviour", "normalis", "visualis", "organis", "analyse", "labelled", "centre")
    for british in british_words:
        assert british not in text.lower(), f"British spelling: {british}"
