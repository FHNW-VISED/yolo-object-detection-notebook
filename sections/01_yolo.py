# %% [markdown]
# ## YOLO: A One-Stage Detector
#
# YOLO ("You Only Look Once") is a **one-stage detector**: one convolutional network looks at the whole image once and directly outputs boxes and class scores. Two-stage detectors such as Faster R-CNN first propose regions and then classify each one, which is slower. We use **YOLO11** through the Ultralytics Python API: training, validation and prediction are one line each.

# %%
# @title 📘 What a YOLO Model Outputs
callout("theory", "📘 What a YOLO Model Outputs", r"""
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Thousands of candidate boxes</strong>, each with class scores (8,400 for a 640×640 image).</li>
<li><strong>A confidence threshold</strong> drops the uncertain ones.</li>
<li><strong>Non-maximum suppression (NMS)</strong> removes duplicates of the same object.</li>
</ol>
<p style="margin:6px 0 0 0;">You will move both thresholds with a slider later.</p>
""")

# %% [markdown]
# YOLO11 comes in five sizes; larger models are more accurate but slower (Ultralytics' results on COCO at 640 px):
#
# | Model | Parameters | COCO mAP@50–95 | Typical target |
# |---|---|---|---|
# | YOLO11n | 2.6 M | 39.5 | phones, edge devices |
# | YOLO11s | 9.4 M | 47.0 | small GPUs |
# | YOLO11m | 20.1 M | 51.5 | consumer GPUs |
# | YOLO11l | 25.3 M | 53.4 | cloud GPUs |
# | YOLO11x | 56.9 M | 54.7 | high-end cloud GPUs |
#
# We fine-tune **YOLO11n**, the smallest, so training takes minutes.

