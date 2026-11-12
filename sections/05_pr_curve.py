# %% [markdown]
# ## Precision, Recall and AP
#
# Explore how the confidence threshold trades precision against recall, and how AP, mAP@50 and mAP@50–95 summarize it.
#
# Every box comes with a confidence. Which boxes we keep depends on a threshold, and each threshold gives different TP, FP and FN counts. Two numbers summarize them:
#
# - **Precision** = TP / (TP + FP): the share of shown boxes that are real objects.
# - **Recall** = TP / (TP + FN): the share of real objects the model finds.

# %% [markdown]
# The next cell predicts again, keeping every box down to confidence 0.001, as Ultralytics does for metrics. Each threshold below then just filters these boxes.

# %%
# @title Predict with a very low threshold { display-mode: "form" }
METRIC_CONF = 0.001  # Ultralytics' threshold for metrics: keep (almost) everything, then sweep
all_predictions = predict_all(best_weights, VAL_IMAGES, conf=METRIC_CONF)
gt_all = {p: read_gt(p) for p in VAL_IMAGES}
print(f"{sum(len(p['conf']) for p in all_predictions.values())} candidate boxes on {len(VAL_IMAGES)} images.")

# %%
# @title Helpers: precision-recall curves and AP { display-mode: "form" }
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz


def pr_curves(predictions: dict, iou_thr: float = MATCH_IOU) -> dict:
    """Per class: confidences (high → low) with the precision and recall when keeping boxes down to each one."""
    confs, tps, classes, n_gt = [], [], [], np.zeros(len(CLASS_NAMES), int)
    for path, pred in predictions.items():
        is_tp, _ = match(pred, gt_all[path], iou_thr)
        confs.append(pred["conf"])
        tps.append(is_tp)
        classes.append(pred["cls"])
        n_gt += np.bincount(gt_all[path]["cls"], minlength=len(CLASS_NAMES))
    confs, tps, classes = (np.concatenate(x) if x else np.zeros(0) for x in (confs, tps, classes))
    curves = {}
    for k in range(len(CLASS_NAMES)):
        order = np.argsort(-confs[classes == k])
        tp = tps[classes == k][order].astype(float)
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
        curves[k] = {"conf": confs[classes == k][order],
                     "precision": cum_tp / np.maximum(cum_tp + cum_fp, 1),
                     "recall": cum_tp / max(n_gt[k], 1), "n_gt": int(n_gt[k])}
    return curves


def average_precision(curve: dict) -> float:
    """Area under the precision–recall curve, made monotonic and sampled at 101 recall points (COCO style)."""
    recall = np.concatenate(([0.0], curve["recall"], [1.0]))
    precision = np.concatenate(([1.0], curve["precision"], [0.0]))
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))  # the best precision at this recall or more
    x = np.linspace(0, 1, 101)
    return float(trapezoid(np.interp(x, recall, precision), x))


def at_threshold(curve: dict, conf: float) -> tuple[float, float]:
    """(precision, recall) when keeping the boxes with confidence ≥ conf."""
    n = int((curve["conf"] >= conf).sum())
    if n == 0:
        return 1.0, 0.0
    return float(curve["precision"][n - 1]), float(curve["recall"][n - 1])


CLASS_LINE_COLORS = ["#3D405B", "#E07A5F", "#81B29A", "#9C89B8", "#F2CC8F", "#2A9D8F"]
curves_50 = pr_curves(all_predictions)

# %% [markdown]
# ### 🟡 The Confidence Threshold: Precision or Recall?
#
# The vineyard counts bunches to estimate the harvest. An invented box inflates the estimate; a missed bunch deflates it.
#
# **Before running the code:**
# - Raising the confidence threshold from 0.1 to 0.8: what happens to precision and recall? Write down your guess.
#
# **Explore:**
# 1. Move the `confidence` slider from low to high. The left panel shows one image at that threshold; the right shows each class's precision–recall curve, with a dot at the threshold.
# 2. Repeat on two or three other images with the `image` slider.
#
# **Observe and interpret:**
# - What happens to the orange (invented) and the red (missed) boxes as the threshold rises?
# - Which threshold would you choose for the harvest estimate, and why is none best for both precision and recall?

# %%
# @title 🔎 Explore the confidence threshold { display-mode: "form" }
def show_threshold(conf: float, image: int):
    path = VAL_IMAGES[image]
    fig, (ax_img, ax_pr) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    draw_matches(ax_img, path, keep_above(all_predictions[path], conf), title=f"confidence ≥ {conf:.2f} · ")
    for k, curve in curves_50.items():
        color = CLASS_LINE_COLORS[k % len(CLASS_LINE_COLORS)]
        ax_pr.plot(curve["recall"], curve["precision"], color=color, lw=1.8, label=CLASS_NAMES[k])
        p, r = at_threshold(curve, conf)
        ax_pr.plot(r, p, "o", color=color, ms=10, mec="black")
        ax_pr.annotate(f"P {p:.2f} · R {r:.2f}", (r, p), textcoords="offset points", xytext=(8, -14),
                       fontsize=9, color=color)
    ax_pr.set(xlim=(0, 1.02), ylim=(0, 1.05), xlabel="recall: share of real objects found",
              ylabel="precision: share of boxes that are real", title="Precision–recall curve per class (IoU ≥ 0.5)")
    ax_pr.legend(loc="lower left", title="class (dot = current threshold)", fontsize=8, title_fontsize=8)
    match_legend(fig)
    show_live(fig)


_ = widgets.interact(show_threshold,
                     conf=widgets.FloatSlider(value=0.25, min=0.05, max=0.95, step=0.05, continuous_update=False,
                                              description="confidence"),
                     image=widgets.IntSlider(value=0, min=0, max=len(VAL_IMAGES) - 1, continuous_update=False,
                                             description="image"))

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>Raising the threshold increases precision and lowers recall.</strong> Low-confidence boxes are mostly
false positives, so removing them makes the shown boxes more trustworthy (fewer orange boxes), but some real
objects also had low confidence and are now missed (more red boxes). The dot slides left and up along the curve.
Over the validation images, precision rises from roughly 0.5 at a threshold of 0.1 to 0.9 at 0.5 and nearly 1.0
at 0.8, while recall falls from roughly 0.9 to 0.65 at 0.5 and about 0.2 at 0.8. The model is rarely very
confident about a bunch partly hidden by leaves, so recall drops steadily and ends low.</p>
<p><strong>For a harvest estimate</strong>, invented and missed boxes both bias the count, so a middle threshold
of about 0.3–0.5 is defensible: it keeps precision around 0.75–0.9 and recall around 0.65–0.8, so the errors
roughly cancel in the total. If the bunches are also to be picked or measured one by one, a wrong box costs more
and a higher threshold fits. That is a decision about the vineyard, not about the model. No threshold is best for
both, because the same low-confidence boxes contain both the invented bunches and some of the real ones.</p>
""")

# %% [markdown]
# ### AP: The Whole Curve in One Number
#
# Choosing a threshold is a deployment decision. To compare *models*, we want one number that ignores it: the **average precision (AP)**, the area under the precision–recall curve. A perfect detector has AP = 1. **mAP** is the mean AP over all classes. Ultralytics computes it too, as the next cell shows.

# %%
# @title 🔎 Compute AP per class and compare with Ultralytics { display-mode: "form" }
val_metrics = YOLO(str(best_weights)).val(data=str(DATA_YAML), imgsz=160 if SMOKE else IMG_SIZE, batch=4 if SMOKE else 16,
                                          device="cpu" if SMOKE else DEVICE, plots=False, verbose=False)
ap_50 = {k: average_precision(c) for k, c in curves_50.items()}

fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(4.2 * len(CLASS_NAMES), 3.6), squeeze=False)
for ax, (k, curve) in zip(axes.ravel(), curves_50.items(), strict=True):
    recall = np.concatenate(([0.0], curve["recall"]))
    precision = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], curve["precision"])))))
    color = CLASS_LINE_COLORS[k % len(CLASS_LINE_COLORS)]
    ax.fill_between(recall, precision, step="post", color=color, alpha=0.25, label=f"area = AP = {ap_50[k]:.2f}")
    ax.plot(curve["recall"], curve["precision"], color=color, lw=1.5, label="precision–recall curve")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.05), xlabel="recall", ylabel="precision",
           title=f"{CLASS_NAMES[k]} ({curve['n_gt']} objects)")
    ax.legend(loc="lower left", fontsize=8)
fig.suptitle("AP is the shaded area under each class's curve (IoU ≥ 0.5)", color=INK)
fig.tight_layout()
show(fig)
print(f"mAP@50, our computation: {np.mean(list(ap_50.values())):.3f} (mean of the class APs above)")
print(f"mAP@50, Ultralytics val:  {val_metrics.box.map50:.3f} (small differences come from how boxes are matched)")

# %% [markdown]
# ### 🟡 mAP@50 vs. mAP@50–95: What Does Each Reward?
#
# So far a prediction counted as correct at IoU ≥ 0.5, which a box covering two-thirds of an object can pass. **mAP@50–95** averages the AP at ten IoU thresholds, 0.50, 0.55, …, 0.95. It is the main COCO metric, used in the model table at the top.
#
# **Before running the code:**
# - Will AP go up or down as the required IoU rises from 0.50 to 0.95?
#
# **Explore:**
# - Run the cell. Each bar is the mAP at one IoU threshold; the dashed line is their mean, mAP@50–95.
#
# **Observe and interpret:**
# - How large is the gap between mAP@50 and mAP@50–95? What does it tell you about the boxes?
# - Name one application where mAP@50 is the number to watch, and one where mAP@50–95 matters more.

# %%
# @title 🔎 AP at ten IoU thresholds { display-mode: "form" }
IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)
map_per_iou = [np.mean([average_precision(c) for c in pr_curves(all_predictions, t).values()]) for t in IOU_THRESHOLDS]

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar([f"{t:.2f}" for t in IOU_THRESHOLDS], map_per_iou, color=["#2E7D32"] + ["#81B29A"] * 9)
ax.axhline(np.mean(map_per_iou), ls="--", color=INK, label=f"mean = mAP@50–95 = {np.mean(map_per_iou):.2f}")
ax.set(xlabel="IoU a prediction needs to count as a true positive", ylabel="mAP", ylim=(0, 1),
       title=f"mAP@50 = {map_per_iou[0]:.2f} (dark bar) · stricter thresholds reward tighter boxes")
ax.legend()
show(fig)
print(f"mAP@50–95, our computation: {np.mean(map_per_iou):.3f} · Ultralytics val: {val_metrics.box.map:.3f}")

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>AP falls as the required IoU rises, and here it falls fast.</strong> mAP@50 is roughly 0.85–0.9, but
the bars drop to about 0.6 at IoU 0.75 and to about 0.1 at 0.9, so mAP@50–95 is only about 0.5–0.55. The gap of
about 0.3 means the model <em>finds</em> the bunches, but its boxes are not always tight. This is partly because
bunch edges are fuzzy: for a bunch 45 pixels wide, a box shifted by just 5 pixels in both directions already has an
IoU of only about 0.65.</p>
<p><strong>mAP@50</strong> is the right number when finding is what matters: is there a bunch at this spot, so
that it is counted for the harvest estimate. <strong>mAP@50–95</strong> matters when the box itself is used:
estimating each bunch's size or weight from its box, or guiding a harvesting robot's gripper.</p>
""")
