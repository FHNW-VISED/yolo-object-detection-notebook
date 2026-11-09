# %% [markdown]
# ## Non-Maximum Suppression
#
# See how the NMS threshold decides between duplicate boxes and missed neighbors.
#
# Before the confidence threshold is applied, the network proposes thousands of candidate boxes, and a well-visible object usually gets several overlapping ones. **Non-maximum suppression (NMS)** keeps one box per object:
#
# 1. Sort the boxes of one class by confidence.
# 2. Keep the most confident box.
# 3. Delete every remaining box whose IoU with the kept box is larger than the **NMS IoU threshold**.
# 4. Repeat with the next box that is still left.
#
# Ultralytics uses an NMS IoU threshold of 0.7 by default (the `iou` argument of `predict`).

# %% [markdown]
# ### 🟡 NMS Threshold: Duplicates or Missed Neighbors?
#
# The image below is the validation image with the most annotated objects, where objects stand close together.
#
# **Before running the code:**
# - A *low* NMS threshold deletes boxes that overlap only a little. What could go wrong when two real objects stand side by side?
# - A threshold of 1.0 deletes nothing. What will you see?
#
# **Explore:**
# 1. Move the `NMS IoU` slider from 0.1 to 1.0. The left panel always shows the default 0.7, the right panel your setting.
# 2. Watch the counts of true positives, false positives and false negatives in the titles.
#
# **Observe and interpret:**
# - At which values do duplicates appear (orange boxes on top of a green one)?
# - At which values do real objects get deleted (red dashed outlines next to a green box)?
# - Why does the best value depend on how crowded the scenes are?

# %%
# @title 🔎 Explore the NMS Threshold { display-mode: "form" }
NMS_CONF = 0.25  # the display threshold of section Predictions: TP, FP and FN
crowded_image = max(VAL_IMAGES, key=lambda p: len(read_gt(p)["cls"]))
nms_detector = YOLO(str(best_weights))


def predict_with_nms(nms_iou: float) -> dict:
    r = nms_detector.predict(str(crowded_image), conf=NMS_CONF, iou=nms_iou, imgsz=160 if SMOKE else IMG_SIZE,
                             device="cpu" if SMOKE else DEVICE, verbose=False)[0]
    return {"xyxy": r.boxes.xyxy.cpu().numpy(), "conf": r.boxes.conf.cpu().numpy(),
            "cls": r.boxes.cls.cpu().numpy().astype(int)}


nms_default = predict_with_nms(0.7)


def show_nms(nms_iou: float):
    fig, (ax_default, ax_yours) = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_matches(ax_default, crowded_image, nms_default, title="default NMS IoU 0.70 · ", show_conf=False)
    draw_matches(ax_yours, crowded_image, predict_with_nms(nms_iou), title=f"your NMS IoU {nms_iou:.2f} · ",
                 show_conf=False)
    match_legend(fig)
    show_live(fig)


_ = widgets.interact(show_nms, nms_iou=widgets.FloatSlider(value=0.7, min=0.1, max=1.0, step=0.05,
                                                           continuous_update=False, description="NMS IoU"))

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>Close to 1.0, duplicates appear:</strong> almost no box is deleted, so one object is covered by several
boxes; only one of them can be the true positive, the others count as false positives.</p>
<p><strong>At low values, neighbors disappear:</strong> two real objects that overlap a little are treated as one,
the less confident box is deleted, and its object becomes a false negative.</p>
<p><strong>The best value depends on crowding:</strong> where objects never overlap, a low threshold is safe; in
dense scenes (herds, shelves, crowds) real objects overlap, and a higher threshold is needed to keep them.</p>
""")
