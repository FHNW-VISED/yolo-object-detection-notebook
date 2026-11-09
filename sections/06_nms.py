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
# The image below is the validation image with the most annotated objects, about twenty weeds spread over it.
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
<p><strong>Close to 1.0, duplicates appear:</strong> almost no box is deleted, so one object is covered by several
boxes; only one of them can be the true positive, the others count as false positives. On this image the false
positives stay at about 4 up to 0.75, then grow to over 10 at 0.85 and to more than 50 at 0.95, while the true
positives do not change.</p>
<p><strong>At low values nothing happens here:</strong> the red outlines and the green boxes stay the same from 0.1
to 0.75. NMS can only delete a real neighbor when two real objects overlap, and the weeds in these images are tiny
and stand apart, so their boxes almost never touch. In a different dataset the low end would show lost neighbors.</p>
<p><strong>The best value depends on crowding:</strong> where objects never overlap, any value from low to about
0.7 is safe, as here; in dense scenes (herds, shelves, crowds, weeds growing in a clump) real objects overlap, and a
higher threshold is needed to keep them.</p>
""")
