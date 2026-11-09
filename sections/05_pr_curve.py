# %% [markdown]
# ## Precision, Recall and AP
#
# Explore how the confidence threshold trades precision against recall, and see how AP, mAP@50 and mAP@50–95 summarize that trade-off.
#
# Every predicted box comes with a confidence. Which boxes we keep depends on a threshold, and every threshold gives a different number of TPs, FPs and FNs. Two numbers summarize them:
#
# - **Precision** = TP / (TP + FP): of the boxes the model shows, the share that are real objects.
# - **Recall** = TP / (TP + FN): of the real objects, the share the model finds.

# %% [markdown]
# The next cell predicts once more on all validation images, this time keeping every box down to a confidence of 0.001, as Ultralytics does when it computes metrics. Each threshold below then just filters these boxes.

# %%
# @title Predict with a Very Low Threshold { display-mode: "form" }
METRIC_CONF = 0.001  # Ultralytics' threshold for metrics: keep (almost) everything, then sweep
all_predictions = predict_all(best_weights, VAL_IMAGES, conf=METRIC_CONF)
gt_all = {p: read_gt(p) for p in VAL_IMAGES}
print(f"{sum(len(p['conf']) for p in all_predictions.values())} candidate boxes on {len(VAL_IMAGES)} images.")

# %%
# @title Helpers: Precision–Recall Curves and AP { display-mode: "form" }
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
# Imagine the detector is used to count how many animals visit a feeder each night: every invented box makes the count too high, and every missed animal makes it too low.
#
# **Before running the code:**
# - If you raise the confidence threshold from 0.1 to 0.8, what happens to precision and to recall? Write down your guess.
#
# **Explore:**
# 1. Move the `confidence` slider from low to high. The left panel shows one image at that threshold; the right panel shows the precision–recall curve of each class, with a dot where the threshold puts you.
# 2. Pick two or three other images with the `image` slider and repeat.
#
# **Observe and interpret:**
# - What happens to the orange (invented) and the red (missed) boxes as the threshold rises?
# - Which threshold would you choose for counting, and why is there no threshold that is best for both precision and recall?

# %%
# @title 🔎 Explore the Confidence Threshold { display-mode: "form" }
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
animals also had low confidence and are now missed (more red boxes). The dot slides left and up along the curve.
Over all validation images, precision rises from roughly 0.7 at confidence 0.1 to 0.99 at 0.8, while recall falls
from about 0.9 to 0.5: the move is gentle up to about 0.5 and steep beyond 0.6.</p>
<p><strong>For counting</strong>, only the difference between invented and missed boxes matters, so a threshold
where the two are about equally frequent works best. For this model that is around 0.3, where each error kind
occurs about 30 to 35 times on the 150 validation images and the errors cancel in the total. No threshold is
best for both precision and recall, because the same low-confidence boxes contain both the invented animals and
some of the real ones.</p>
""")

# %% [markdown]
# ### AP: The Whole Curve in One Number
#
# Choosing a threshold is a deployment decision. To compare *models*, we want one number that does not depend on it: the **average precision (AP)**, the area under the precision–recall curve. A perfect detector has AP = 1: it finds every object before it shows a single wrong box. **mAP** is the mean of the AP over all classes. Ultralytics also computes it when it validates a model, as the next cell shows.

# %%
# @title 🔎 Compute AP per Class and Compare with Ultralytics { display-mode: "form" }
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
# So far a prediction counted as correct if its IoU with the ground truth was at least 0.5. A box that covers only two-thirds of an object can pass that bar. **mAP@50–95** repeats the AP computation at ten stricter IoU thresholds, 0.50, 0.55, …, 0.95, and averages them. It is the main COCO metric and the one in the model table at the top.
#
# **Before running the code:**
# - Will AP go up or down as the required IoU rises from 0.50 to 0.95?
#
# **Explore:**
# - Run the cell. Each bar is the mAP at one IoU threshold; the dashed line is their mean, the mAP@50–95.
#
# **Observe and interpret:**
# - How large is the gap between mAP@50 and mAP@50–95 for our model? What does the gap tell you about the boxes it draws?
# - Name one application where mAP@50 is the number to look at, and one where mAP@50–95 matters more.

# %%
# @title 🔎 AP at Ten IoU Thresholds { display-mode: "form" }
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
<p><strong>AP falls as the required IoU rises.</strong> At 0.5 almost every found animal counts; at 0.9 only
boxes that fit the animal very tightly do. For our model mAP@50 is about 0.9 to 0.95 and mAP@50–95 about 0.6 to
0.65, a gap of roughly 0.3. The bars stay high up to about IoU 0.75 and then drop fast (around 0.2 at 0.9, close
to 0 at 0.95): the model <em>finds</em> the animals reliably, but its boxes are only moderately tight. Part of that
is the labels: a box around a hog that is half cut off by the image edge or hidden in grass is a matter of
judgment, and two annotators would not agree to the last pixel.</p>
<p><strong>mAP@50</strong> is the right number when finding and counting is what matters (how many hogs came,
is any animal there at all). <strong>mAP@50–95</strong> matters when the box itself is used: measuring an animal's
size, cropping it for a second model that tells young from adult, or pointing a second camera at it.</p>
""")
