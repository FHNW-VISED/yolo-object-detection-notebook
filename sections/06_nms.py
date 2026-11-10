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
# The image below is the validation image with the most annotated objects, 27 grape bunches, many of them touching.
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
positives are 0 to 2 up to 0.7, then grow to about 15 at 0.85, to about 60 at 0.95 and to over 150 at 1.0, while
the true positives do not change (about 24 of 27): the orange boxes pile up on top of the green ones.</p>
<p><strong>At the lowest values a real neighbor is deleted:</strong> at 0.1 one more bunch turns red than at 0.3,
because two touching bunches overlap enough for NMS to treat the second as a duplicate of the first. The effect is
small here (one bunch), since most bunches overlap their neighbors only a little; between about 0.3 and 0.5 this image
is clean, with no orange box and only the same three missed bunches.</p>
<p><strong>The best value depends on crowding:</strong> where objects never overlap, a low value is safe; in dense
scenes (bunches in a clump, herds, shelves, crowds) real objects overlap, and a higher threshold is needed to keep
them, at the price of some duplicates. For these vines a value around 0.5–0.7 is a good compromise.</p>
""")
