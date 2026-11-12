# %% [markdown]
# ## Predictions: TP, FP and FN
#
# Predict on the validation images and classify every box as a true positive, false positive or false negative.
#
# A metric is only a summary. Before trusting one, look at the predictions: what the model finds, invents and misses.

# %%
# @title 📘 True Positives, False Positives, False Negatives
callout("theory", "📘 True Positives, False Positives, False Negatives", r"""
<p style="margin:6px 0 0 0;">A predicted box is compared with the ground-truth boxes of the same class using the
<strong>intersection over union (IoU)</strong>: the overlap area divided by the area the two boxes cover together
(0 = no overlap, 1 = identical).</p>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><span style="color:#2E7D32;font-weight:bold;">True positive (TP)</span>: a prediction that overlaps a
ground-truth box of the same class with IoU ≥ 0.5. Each ground-truth box is matched once, by the most
confident prediction.</li>
<li><span style="color:#EF6C00;font-weight:bold;">False positive (FP)</span>: a prediction with no such match: a
duplicate, wrong class, badly placed box, or nonexistent object.</li>
<li><span style="color:#C62828;font-weight:bold;">False negative (FN)</span>: a ground-truth box no
prediction matched: a missed object.</li>
</ul>
<p style="margin:6px 0 0 0;">Detection has no true negatives: "background" is not countable.</p>
""")

# %%
# @title Helpers: matching and drawing { display-mode: "form" }
MATCH_IOU = 0.5  # IoU a prediction needs to count as a true positive


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N×4) with every box in b (M×4), boxes as x0, y0, x1, y1 in pixels."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match(pred: dict, gt: dict, iou_thr: float = MATCH_IOU) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matching, most confident prediction first. Returns (is_tp per prediction, is_found per GT box)."""
    is_tp = np.zeros(len(pred["conf"]), bool)
    found = np.zeros(len(gt["cls"]), bool)
    ious = box_iou(pred["xyxy"], gt["xyxy"])
    for i in np.argsort(-pred["conf"]):
        candidates = np.where((gt["cls"] == pred["cls"][i]) & ~found & (ious[i] >= iou_thr))[0]
        if len(candidates):
            best = candidates[np.argmax(ious[i, candidates])]
            found[best] = is_tp[i] = True
    return is_tp, found


def predict_all(weights, images, conf: float, iou: float = 0.7) -> dict:
    """Predictions for every image path: {path: {"xyxy", "conf", "cls"}} as numpy arrays."""
    detector = YOLO(str(weights))
    out = {}
    for path in images:
        r = detector.predict(str(path), conf=conf, iou=iou, imgsz=160 if SMOKE else IMG_SIZE,
                             device="cpu" if SMOKE else DEVICE, verbose=False)[0]
        out[path] = {"xyxy": r.boxes.xyxy.cpu().numpy(), "conf": r.boxes.conf.cpu().numpy(),
                     "cls": r.boxes.cls.cpu().numpy().astype(int)}
    return out


def keep_above(pred: dict, conf: float) -> dict:
    """The predictions with confidence ≥ conf."""
    keep = pred["conf"] >= conf
    return {k: v[keep] for k, v in pred.items()}


def outline(ax, box, color, dashed=False, label=None):
    """One box on an image axis; a thin dark edge keeps light colors visible on bright images."""
    x0, y0, x1, y1 = box
    rect = mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.8, edgecolor=color,
                                 linestyle="--" if dashed else "-")
    rect.set_path_effects([path_effects.Stroke(linewidth=3, foreground="#00000088"), path_effects.Normal()])
    ax.add_patch(rect)
    if label:
        ax.text(x0, y0 - 2, label, color="white", fontsize=7, va="bottom",
                bbox=dict(facecolor=color, edgecolor="none", pad=1))


def draw_matches(ax, path, pred: dict, title: str = "", show_conf: bool = True) -> tuple[int, int, int]:
    """Image with GT (white dashed), TP (green), FP (orange) and missed GT (red). Returns (TP, FP, FN)."""
    gt = read_gt(path)
    is_tp, found = match(pred, gt)
    ax.imshow(Image.open(path).convert("RGB"))
    ax.axis("off")
    for box, hit in zip(gt["xyxy"], found, strict=True):
        outline(ax, box, GT_COLOR if hit else FN_COLOR, dashed=True)
    for box, c, k, tp in zip(pred["xyxy"], pred["conf"], pred["cls"], is_tp, strict=True):
        label = f"{CLASS_NAMES[k]} {c:.2f}" if show_conf else None
        outline(ax, box, TP_COLOR if tp else FP_COLOR, label=label)
    counts = int(is_tp.sum()), int((~is_tp).sum()), int((~found).sum())
    ax.set_title(f"{title}TP {counts[0]} · FP {counts[1]} · FN {counts[2]}", fontsize=10)
    return counts


def match_legend(fig):
    """The legend every TP/FP/FN figure carries (N-059)."""
    handles = [mpl.lines.Line2D([], [], color=TP_COLOR, lw=2, label="true positive (found)"),
               mpl.lines.Line2D([], [], color=FP_COLOR, lw=2, label="false positive (invented)"),
               mpl.lines.Line2D([], [], color=FN_COLOR, lw=2, ls="--", label="false negative (missed)"),
               mpl.lines.Line2D([], [], color="#9CA3AF", lw=2, ls="--", label="ground truth, matched (white)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))


# %% [markdown]
# ### Predict on the Validation Images
#
# The next cell runs the fine-tuned model on every validation image and keeps every box with confidence of at least 0.25, the Ultralytics default. Later sections lower it.

# %%
# @title Predict on the validation images { display-mode: "form" }
val_predictions = predict_all(best_weights, VAL_IMAGES, conf=0.25)
n_boxes = sum(len(p["conf"]) for p in val_predictions.values())
n_gt = sum(len(read_gt(p)["cls"]) for p in VAL_IMAGES)
print(f"{len(VAL_IMAGES)} validation images: {n_boxes} predicted boxes, {n_gt} ground-truth boxes.")

# %% [markdown]
# ### 🟢 True Positives, False Positives, False Negatives: What Went Wrong?
#
# Each image shows the ground truth as dashed outlines and the predictions as solid boxes with class and confidence, colored by the rule above.
#
# **Before running the code:**
# - Which mistake do you expect more often: invented bunches (FP) or missed bunches (FN)?
#
# **Explore:**
# - Browse at least ten validation images (Previous / Next).
#
# **Observe and interpret:**
# - Find one example of each: a true positive, a false positive, a false negative.
# - Why might the model have made each FP and FN? Consider object size, occlusion, the image edge, and whether the ground truth is right.

# %%
# @title 🔎 Browse predictions vs. ground truth { display-mode: "form" }
browse_index = 0
browse_output = widgets.Output()
browse_prev = widgets.Button(description="← Previous")
browse_next = widgets.Button(description="Next →")
browse_counter = widgets.HTML()


def browse_show():
    with browse_output:
        clear_output(wait=True)
        path = VAL_IMAGES[browse_index]
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_matches(ax, path, val_predictions[path], title=f"{Path(path).name} · ")
        match_legend(fig)
        show_live(fig)
    browse_counter.value = f"Image {browse_index + 1} of {len(VAL_IMAGES)}"
    browse_prev.disabled = browse_index == 0
    browse_next.disabled = browse_index >= len(VAL_IMAGES) - 1


def browse_step(step):
    global browse_index
    browse_index = min(max(browse_index + step, 0), len(VAL_IMAGES) - 1)
    browse_show()


browse_prev.on_click(lambda _: browse_step(-1))
browse_next.on_click(lambda _: browse_step(1))
display(widgets.HBox([browse_prev, browse_next, browse_counter]), browse_output)
browse_show()

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>True positives</strong> are bunches that hang clearly in view: pale green, fairly large, with leaves
only at their edge. At confidence 0.25 the model finds roughly five out of six of the labeled bunches (recall
about 0.8–0.85).</p>
<p><strong>False positives are more frequent than false negatives</strong>, about twice as many. Almost all of
them sit on or right next to a labeled bunch, not on bare background: a second box on the same bunch, a box that
covers only half of it, or one box around two touching bunches. Some may be real bunches the annotators left
unlabeled, deep in the background; only a look at the image can tell.</p>
<p><strong>False negatives</strong> are mostly bunches half hidden behind leaves, small bunches (about 40 pixels
across, against about 47 for all of them) and bunches cut by the image border. They are not mostly bunches that
touch a neighbor.</p>
""")
