"""Her brief, point by point, and her original exercises, one by one (pages 2-3 of the hand-back PDF)."""
OK, PART, NO = "✅ done", "⚠️ partly", "❌ not done"
BRIEF = [
    ("Update to YOLO11 (or another one you suggest)", OK,
     "YOLO11n through Ultralytics, pinned to 8.4.208. YOLO26 is newer, but YOLO11n scores slightly higher on Ultralytics' own 37-dataset fine-tuning benchmark (46.4 vs 44.4 mAP@50–95) and keeps the classic NMS step the slider shows.", "1"),
    ("Simplify the setup using the Python API", OK,
     "Training, validation and prediction are one Python call each. Shell commands: 23 in your version, 1 now (the install). No runtime pin, no torch reinstall, no git clone, no Drive, no W&amp;B login.", "1"),
    ("Ensure compatibility with Google Colab", PART,
     "Not run in Colab yet: Simone does a T4 run next. Tested instead with a headless run on CPU (every check passing) and one full run on a Mac GPU (45 min, 0 errors). Colab now ships Python 3.13; the tests ran on 3.12.", "–"),
    ("A public dataset, about 500 images, a few classes, not COCO-like, with boxes, suitable for fine-tuning", OK,
     "Grape bunches in a vineyard (Embrapa WGISD, CC BY-NC 4.0, non-commercial teaching allowed): 1 class, 242 training and 58 validation images. Fine-tuned YOLO11n reaches mAP@50 0.86.", "2"),
    ("Download from Kaggle or so", PART,
     "From the dataset's own GitHub page instead. Kaggle needs an account and an API key inside Colab; GitHub works without a login (about 300 MB, one to two minutes).", "2"),
    ("Keep dataset exploration minimal", OK,
     "One image browser and the label-format recap. Your two analysis tasks are left to Know Your Images.", "3"),
    ("Keep explanations, student tasks and the IoU / TP / FP / FN exercises; avoid restructuring", OK,
     "Same order as yours: setup, model, data, training, predictions, TP/FP/FN, optional experiments. The TP/FP/FN task, the training-settings task, the label-format recap and both optional experiments are kept. Every one of your exercises is on the next page.", "4–5, 9"),
    ("The IoU calculation can be deleted", OK,
     "Deleted. IoU is defined in one sentence and points to the optional IoU task in Know Your Images.", "5"),
    ("Interactive precision–recall exercise with YOLO predictions", OK,
     "Confidence slider: one image with its TP/FP/FN and the precision–recall curve of each class with the current point.", "6"),
    ("… understand AP", OK, "AP drawn as the shaded area under each class's curve, checked against Ultralytics' value.", "7"),
    ("… interpret mAP@50 vs mAP@50–95", OK,
     "AP at the ten IoU thresholds; their mean is mAP@50–95. A task asks what each rewards (finding vs tight boxes).", "7"),
    ("Optionally compare YOLO11n and YOLO11s", OK,
     "Optional, behind one checkbox. In our run YOLO11s was clearly better (mAP@50 0.90 vs 0.86) and took about 1.5 times as long.", "9"),
    ("All cells run sequentially in a fresh runtime", PART,
     "Yes on a fresh local kernel (CPU smoke test and full Mac run). Colab still to do (see above).", "–"),
    ("American English", OK, "An automatic check for British spellings runs with every test; headings in Title Case.", "–"),
    ("Retain sample solutions", OK, "9 tasks, 9 Sample Answers, written from what the real runs showed.", "4–9"),
    ("Keep training times manageable", PART,
     "Default model: about 4 min for 20 epochs on a Mac GPU. Not yet measured on a Colab T4.", "–"),
    ("If possible, a slider for the NMS threshold", PART,
     "Built. Duplicates show clearly at 0.95 (false positives 2 → 58). Lost neighbors at low values do not show on this dataset (no bunch is lost at 0.3); the Sample Answer explains it.", "8"),
    ("Keep the notebook short", PART,
     "Fewer cells (66 vs 90) and 1 shell command vs 23, but more to read: about 2,300 visible words vs 1,400 (1.6×), because the new precision–recall, AP and NMS parts are explained and every task keeps your full structure. The Sample Answers are hidden in toggles.", "–"),
    ("Complement the exploration notebook without duplicating it", OK,
     "No exploration tasks and no IoU implementation here; both are in Know Your Images.", "3"),
]
EXERCISES = [
    ("Analyze the data qualitatively (visual inspection)", "Cut: Know Your Images covers it. One short image browser remains."),
    ("Analyze the data quantitatively (with code)", "Cut: Know Your Images covers it."),
    ("Choose training parameters (batch size, epochs)", "Kept: two sliders and a Sample Answer, then a new task on reading the training curves."),
    ("Evaluate the training in wandb.ai", "Replaced: the training curves are drawn in the notebook (new task \"Is the Model Still Learning?\")."),
    ("Check individual predictions; set the weights path and confidence threshold", "Kept in spirit: predictions are browsed image by image; the threshold became the confidence slider."),
    ("Find the experiment folder of your run", "Dropped: the Python API returns the results, so no folder hunting is needed."),
    ("TP, FP, FN in the sample image", "Kept, with every box colored by the matching rule."),
    ("Calculate the IoU (coding)", "Deleted, as you suggested."),
    ("Optional: freeze layers during training", "Kept as an optional task (freeze=10, the backbone)."),
    ("Optional: train the tiny model and compare in wandb", "Became YOLO11n vs YOLO11s, compared in a table in the notebook."),
    ("(new) What does the COCO model see before fine-tuning?", "New first task: motivates fine-tuning."),
    ("(new) Confidence threshold, AP, mAP@50 vs mAP@50–95, NMS threshold", "New tasks from your brief."),
]


def html() -> str:
    rows = "".join(f"<tr><td>{r}</td><td class='st'>{s}</td><td>{w}</td><td>{p}</td></tr>" for r, s, w, p in BRIEF)
    done = sum(s == OK for _, s, _, _ in BRIEF)
    part = sum(s == PART for _, s, _, _ in BRIEF)
    no = sum(s == NO for _, s, _, _ in BRIEF)
    ex = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in EXERCISES)
    return (f"<section><h2>Your brief, point by point</h2><p class='why'>{len(BRIEF)} points: {done} done, {part} partly, "
            f"{no} not done. 'See' names the numbered before/after page that shows it.</p>"
            f"<table><tr><th>You asked</th><th>Status</th><th>What was done</th><th>See</th></tr>{rows}</table></section>"
            f"<section><h2>Your exercises, one by one</h2><table><tr><th>Exercise in your YOLOv7 workbook</th>"
            f"<th>In the new notebook</th></tr>{ex}</table></section>")
