# %% [markdown]
# ## Non-Maximum Suppression
#
# See how the NMS threshold trades duplicate boxes against missed neighbors.
#
# The network proposes thousands of candidate boxes, and a clearly visible object usually gets several overlapping ones. **Non-maximum suppression (NMS)** keeps one box per object:
#
# 1. Sort the boxes of one class by confidence.
# 2. Keep the most confident box.
# 3. Delete every remaining box whose IoU with the kept box exceeds the **NMS IoU threshold**.
# 4. Repeat with the next box still left.
#
# The threshold is the `iou` argument of `predict`. We use 0.7, the value Ultralytics uses for metrics (its plain-prediction default is 0.5).

# %% [markdown]
# ### 🟡 NMS Threshold: Duplicates or Missed Neighbors?
#
# The image below is the validation image with the most annotated objects: about thirty cows.
#
# **Before running the code:**
# - A *low* NMS threshold deletes even slightly overlapping boxes. What could go wrong with two real objects side by side?
# - A threshold of 1.0 deletes nothing. What will you see?
#
# **Explore:**
# 1. Move the `NMS IoU` slider from 0.1 to 1.0. The left panel shows the default 0.7, the right your setting.
# 2. Watch the TP, FP and FN counts in the titles.
#
# **Observe and interpret:**
# - At which values do duplicates appear (orange boxes on a green one)?
# - At which values, if any, do real objects get deleted (red dashed outlines beside a green box)?
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
<p><strong>Close to 1.0, duplicates appear:</strong> almost no box is deleted, so one cow is covered by several
boxes; only one of them can be the true positive, the others count as false positives (orange boxes stacked on a
green one). On this image the false positives are about 4 at 0.3, 7 at the default 0.7, 15 at 0.85 and about 60 at
0.95, while the true positives (21 of 32 cows) do not change.</p>
<p><strong>At low values the effect is weak, but not zero:</strong> from about 0.2 to 0.5 the picture is the same,
and only at the very bottom of the slider (0.1) does NMS start to delete real neighbors: two true positives turn into
red dashed misses. Even in this herd, true cows' boxes never overlap by more than 0.25 IoU, so a low threshold costs
little here. In a denser scene it would delete many more. NMS does not explain every error: the 11 missed cows in the middle of the herd and the
two orange boxes on trees at the top left stay at every value, because they come from the model.</p>
<p><strong>The best value depends on crowding:</strong> where objects rarely overlap, any value from about 0.3 to 0.6 is
safe, as here; in dense scenes (herds, shelves, crowds) real objects overlap, and a higher threshold is needed to
keep them.</p>
""")
