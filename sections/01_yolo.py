# %% [markdown]
# ## YOLO: A One-Stage Detector
#
# Learn what a YOLO model predicts for one image and which model size we train.
#
# YOLO ("You Only Look Once") is a **one-stage detector**: a single convolutional network looks at the whole image once and directly outputs boxes, class scores and a confidence per box. Two-stage detectors such as Faster R-CNN first propose regions and then classify each one, which is slower. Because it needs only one pass, YOLO is fast enough for video and runs on small devices.
#
# Since the first YOLO in 2015, the family has been developed continuously, and Ultralytics releases new versions regularly. In this notebook, we use **YOLO11** through the Ultralytics Python API: one `pip install`, and training, validation and prediction are each one line of Python.

# %%
# @title 📘 What a YOLO Model Outputs
callout("theory", "📘 What a YOLO Model Outputs", r"""
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Many candidate boxes.</strong> For every location of a coarse grid over the image (at three scales),
the network predicts a box, a score for each class and, from these, a confidence. A 640×640 input gives
8,400 candidates.</li>
<li><strong>A confidence threshold</strong> drops the candidates the model is unsure about.</li>
<li><strong>Non-maximum suppression (NMS)</strong> removes duplicates: when several remaining boxes overlap the
same object, only the most confident one is kept.</li>
</ol>
<p style="margin:6px 0 0 0;">You will move both thresholds with a slider later in this notebook.</p>
""")

# %% [markdown]
# YOLO11 comes in five sizes. Larger models detect more accurately but need more memory and time. The numbers below are Ultralytics' published results on COCO at 640 px (mAP@50–95 is explained in the section *Precision, Recall and AP*).
#
# | Model | Parameters | Compute (GFLOPs) | COCO mAP@50–95 | Typical target |
# |---|---|---|---|---|
# | YOLO11n | 2.6 M | 6.5 | 39.5 | phones, edge devices |
# | YOLO11s | 9.4 M | 21.5 | 47.0 | small GPUs |
# | YOLO11m | 20.1 M | 68.0 | 51.5 | consumer GPUs |
# | YOLO11l | 25.3 M | 86.9 | 53.4 | cloud GPUs |
# | YOLO11x | 56.9 M | 194.9 | 54.7 | high-end cloud GPUs |
#
# We fine-tune **YOLO11n**, the smallest, so that training takes minutes. The optional experiments at the end compare it with YOLO11s.

# %%
# @title 🔍 Go Deeper: Why YOLO11 and Not the Newest YOLO?
callout("deeper", "🔍 Go Deeper: Why YOLO11 and Not the Newest YOLO?", r"""
<p style="margin:6px 0 0 0;">Ultralytics' newest family, YOLO26 (2026), can also run <em>NMS-free</em>: a second
output head is trained to give one box per object, so the duplicate-removal step can be skipped at inference time
(<code>nms=False</code>). On small fine-tuning datasets like ours, YOLO11 performs about as well, and its classic
pipeline shows every step you explore in this notebook. Swapping in YOLO26 is a one-word change:
<code>YOLO("yolo26n.pt")</code>.</p>
""")
