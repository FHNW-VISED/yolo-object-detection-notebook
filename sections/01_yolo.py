# %% [markdown]
# ## YOLO: A One-Stage Detector
#
# Learn what a YOLO model predicts for one image and which model size we train.
#
# YOLO ("You Only Look Once") is a **one-stage detector**: one convolutional network looks at the whole image once and outputs boxes, class scores and a confidence per box. Two-stage detectors such as Faster R-CNN first propose regions, then classify each, which is slower.
#
# We use **YOLO11** through the Ultralytics Python API, where training, validation and prediction are each one line.

# %%
# @title 📘 What a YOLO Model Outputs
callout("theory", "📘 What a YOLO Model Outputs", r"""
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Many candidate boxes.</strong> For every location of a coarse grid over the image (at three scales),
the network predicts a box, class scores and a confidence. A 640×640 input gives 8,400 candidates.</li>
<li><strong>A confidence threshold</strong> drops candidates the model is unsure about.</li>
<li><strong>Non-maximum suppression (NMS)</strong> removes duplicates: of several boxes on the same object, only
the most confident is kept.</li>
</ol>
<p style="margin:6px 0 0 0;">You will move both thresholds with a slider later.</p>
""")

# %% [markdown]
# YOLO11 comes in five sizes: larger models are more accurate but need more memory and time. These are Ultralytics' COCO results at 640 px (mAP@50–95 is explained later).
#
# | Model | Parameters | Compute (GFLOPs) | COCO mAP@50–95 | Typical target |
# |---|---|---|---|---|
# | YOLO11n | 2.6 M | 6.5 | 39.5 | phones, edge devices |
# | YOLO11s | 9.4 M | 21.5 | 47.0 | small GPUs |
# | YOLO11m | 20.1 M | 68.0 | 51.5 | consumer GPUs |
# | YOLO11l | 25.3 M | 86.9 | 53.4 | cloud GPUs |
# | YOLO11x | 56.9 M | 194.9 | 54.7 | high-end cloud GPUs |
#
# We fine-tune **YOLO11n**, the smallest, so training takes minutes. The optional experiments compare it with YOLO11s.

# %%
# @title 🔍 Go Deeper: Why YOLO11 and Not the Newest YOLO?
callout("deeper", "🔍 Go Deeper: Why YOLO11 and Not the Newest YOLO?", r"""
<p style="margin:6px 0 0 0;">Ultralytics' newest family, YOLO26 (2026), can also run <em>NMS-free</em>: a second
output head is trained to give one box per object, so the duplicate-removal step can be skipped at inference time
(<code>nms=False</code>). On small fine-tuning datasets like ours, YOLO11 performs about as well, and its classic
pipeline shows every step you explore in this notebook. Swapping in YOLO26 is a one-word change:
<code>YOLO("yolo26n.pt")</code>.</p>
""")
