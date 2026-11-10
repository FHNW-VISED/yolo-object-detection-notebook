"""Lay out the annotated before/after screenshots as an A4-landscape PDF for Susanne."""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

from checklist import html as checklist_html

HERE = Path(__file__).parent
CHROME = ("/Users/simone.bonato/Library/Caches/ms-playwright/chromium_headless_shell-1234/"
          "chrome-headless-shell-mac-arm64/chrome-headless-shell")

PAGES = [
    # (title, text, before image(s) or None, after image(s), after label)
    ("1 · Setup: from version pins to one install", "Your workbook had to pin an old Colab runtime (Python 3.11, "
     "2025.07), reinstall torch 2.5.0 and clone the YOLOv7 repository. Now one cell installs Ultralytics (pinned "
     "8.4.208) on Colab's current runtime, and training, validation and prediction are each one Python call. "
     "Drive and Weights & Biases are no longer needed; the training curves are drawn in the notebook.",
     ["b-setup.png", "b-clone.png"], ["a-setup.png", "a-models.png"], None),
    ("2 · A new dataset, from outside COCO", "Cattle from a drone became grape bunches in a vineyard (Embrapa WGISD, CC BY-NC 4.0, so free for teaching): "
     "242 training and 58 validation images, one class. The bunches are large and hang side by side, which the NMS part "
     "needs. The images come straight from the dataset's GitHub page (about 300 MB, no login). A new first task shows what the COCO model predicts "
     "before fine-tuning, which motivates the training.",
     ["b-data.png"], ["a-data.png", "a-gt.png", "a-coco.png"], None),
    ("3 · Dataset exploration: one short look", "Your qualitative and quantitative analysis tasks are covered by "
     "Know Your Images, so here they are replaced by one sentence and one image browser.",
     ["b-explore.png", "b-explore2.png"], ["a-explore.png"], None),
    ("4 · Training: your batch-size and epochs task kept", "Same task, now as two sliders with a Sample Answer, "
     "followed by the training curves (loss and validation mAP per epoch) and a task on reading them.",
     ["b-train.png"], ["a-train.png", "a-curves.png"], None),
    ("5 · TP, FP, FN kept; the IoU implementation removed", "Your TP/FP/FN task stays, now with every box colored "
     "by the matching rule (green found, orange invented, red missed) and Previous / Next through the validation "
     "images. The IoU coding task is gone, as you suggested; Know Your Images has the optional IoU task.",
     ["b-iou.png"], ["a-tpfp.png"], None),
    ("6 · New: confidence threshold and precision–recall", "A slider moves the confidence threshold; one image "
     "shows the TP/FP/FN at that threshold and the precision–recall curve shows where the threshold puts you. "
     "Shown here at 0.70.", None, ["a-pr.png"], "New in this version"),
    ("7 · New: AP, mAP@50 and mAP@50–95", "AP drawn as the shaded area under each class's curve, checked against "
     "Ultralytics' own value; then AP at the ten IoU thresholds whose mean is mAP@50–95, with a task on what each "
     "metric rewards.", None, ["a-ap.png"], "New in this version"),
    ("8 · New: the NMS slider", "The validation image with the most objects (27 grape bunches), at the default NMS IoU of 0.7 "
     "next to the student's setting. Shown at 0.95: duplicates appear (false positives 2 → 58). At 0.3 the two false "
     "positives go and no real bunch is lost, so the 'lost neighbors' side of NMS is explained in the Sample Answer rather "
     "than seen.",
     None, ["a-nms.png"], "New in this version"),
    ("9 · YOLOv7-tiny became YOLO11n vs YOLO11s (optional)", "Both optional experiments stay behind one checkbox, "
     "so Run all stays fast. In our run YOLO11s was clearly better (mAP@50 0.90 vs 0.86, mAP@50–95 0.59 vs 0.54) "
     "and took about 1.5 times as long; freezing the backbone lowered mAP@50 to 0.81 and trained fastest.",
     ["b-tiny.png"], ["a-ns.png"], None),
]
SUMMARY = """
<p>Your YOLOv7 workbook, modernized as you asked: the same storyline (setup, data, training, predictions,
TP/FP/FN, optional experiments) and your tasks with Sample Answers, on YOLO11 and the Ultralytics Python API, with
a new dataset and two new interactive parts. Each page shows your version on the left and the new one on the
right; the red numbers mark the exact place.</p>
<p><b>Is it shorter?</b> In cells and commands, yes: 66 cells instead of 90, 1 shell command instead of 23. In reading, it is longer: about 2,300 visible words instead of 1,400 (1.6 times), because the precision–recall, AP and NMS parts are new and every task keeps your full structure (Before running the code, Explore, Observe and interpret). The Sample Answers are hidden in toggles.</p>
<p><b>Tested:</b> runs top to bottom headless (CPU smoke test, every check passing) and once in full on a Mac:
YOLO11n trained 20 epochs in about 4 minutes there, mAP@50 0.86, mAP@50–95 0.54. A full run in Colab on a T4 is still to do.
Why YOLO11 and not the newer YOLO26: on Ultralytics' own 37-dataset fine-tuning benchmark YOLO11n scores slightly
higher (46.4 vs 44.4 mAP@50–95), and its classic pipeline shows the NMS step the slider explores.</p>
"""


def page(title, text, before, afters, after_label):
    def fig(src, label):
        return f'<figure><figcaption>{label}</figcaption><img src="img/{src}"></figure>'
    cols = []
    if before:
        befores = before if isinstance(before, list) else [before]
        cols.append('<div class="col">' + "".join(fig(b, "Your YOLOv7 workbook") for b in befores) + "</div>")
    cols.append('<div class="col">' + "".join(fig(a, after_label or "New") for a in afters) + "</div>")
    return f'<section><h2>{title}</h2><p class="why">{text}</p><div class="row">{"".join(cols)}</div></section>'


HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
@page { size: A4 landscape; margin: 12mm; }
body { font-family: Arial, sans-serif; color: #1f2937; }
section { page-break-after: always; }
h1 { border-left: 8px solid #FDE70E; padding-left: 8px; font-size: 22px; margin: 0 0 6px; } h2 { font-size: 18px; margin: 0 0 4px; }
p { font-size: 12.5px; line-height: 1.45; margin: 4px 0 8px; max-width: 250mm; }
.why { color: #374151; }
.row { display: flex; gap: 8mm; align-items: flex-start; }
.col { flex: 1; min-width: 0; }
figure { margin: 0 0 4mm; } figcaption { font-size: 11px; font-weight: bold; color: #D7263D; margin-bottom: 2px; }
table { border-collapse: collapse; font-size: 11px; width: 100%; } th { background: #FDE70E; text-align: left; }
td, th { border: 1px solid #d1d5db; padding: 3px 6px; vertical-align: top; } td.st { white-space: nowrap; }
img { width: 100%; max-height: 150mm; object-fit: contain; object-position: top left;
      border: 1px solid #d1d5db; border-radius: 4px; }
</style></head><body>
<section><h1>Object Detection with YOLO11: what changed</h1>
<p style="color:#6b7280">Simone Bonato · 9 November 2026</p>{summary}</section>
{checklist}
{pages}</body></html>"""


async def main(out: Path):
    html = HTML.replace("{summary}", SUMMARY).replace("{checklist}", checklist_html()).replace("{pages}", "".join(page(*p) for p in PAGES))
    src = HERE / "report.html"
    src.write_text(html)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME)
        pg = await browser.new_page()
        await pg.goto("file://" + str(src))
        await pg.wait_for_timeout(500)
        await pg.pdf(path=str(out), format="A4", landscape=True, print_background=True)
        await browser.close()
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
