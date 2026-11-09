# %% [markdown]
# ## Optional Experiments
#
# Each experiment trains another model (a few minutes on a T4). Tick the box to run them; otherwise the cells skip themselves.

# %%
# ✏️ YOUR TURN: run the optional experiments?
RUN_EXPERIMENTS = False  # @param {type:"boolean"}

# %% [markdown]
# ### 🔴 YOLO11n vs. YOLO11s: Is the Larger Model Worth It? (Optional)
#
# YOLO11s has about 3.6 times as many parameters.
#
# **Before running the code:** how much better and slower do you expect it to be on 500 images?
#
# **Observe and interpret:** compare mAP, training time and prediction time. When would you choose each?

# %%
# @title 🔎 Train YOLO11s and Compare { display-mode: "form" }
def evaluate(weights) -> dict:
    """mAP and speed of one trained model on the validation images."""
    detector = YOLO(str(weights))
    m = detector.val(data=str(DATA_YAML), imgsz=160 if SMOKE else IMG_SIZE, batch=4 if SMOKE else 16,
                     device="cpu" if SMOKE else DEVICE, plots=False, verbose=False)
    n_params = sum(p.numel() for p in detector.model.parameters())
    return {"parameters (millions)": round(n_params / 1e6, 1), "mAP@50": round(float(m.box.map50), 3),
            "mAP@50–95": round(float(m.box.map), 3), "prediction ms / image": round(m.speed["inference"], 1)}


if RUN_EXPERIMENTS or SMOKE:  # smoke mode runs the same code with tiny settings
    model_s = YOLO("yolo11s.pt")
    start = time.time()
    model_s.train(name="yolo11s", **train_settings)
    train_minutes_s = (time.time() - start) / 60
    comparison = pd.DataFrame({
        "YOLO11n": {**evaluate(best_weights), "training minutes": round(train_minutes_n, 1)},
        "YOLO11s": {**evaluate(model_s.trainer.best), "training minutes": round(train_minutes_s, 1)},
    })
    display(comparison)
else:
    print("Skipped: tick RUN_EXPERIMENTS above and run this cell again.")

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>On this small dataset, YOLO11s is not clearly better.</strong> In our test run both models reached about
the same mAP@50 (roughly 0.75–0.8) and mAP@50–95 (roughly 0.45–0.5), and YOLO11s was sometimes a little lower,
while it took two to three times as long to train and was slower per image.</p>
<p><strong>Why:</strong> with a few hundred images, the data, not the model size, limits the result; the larger
model has more to learn from too little. <strong>Choose YOLO11n</strong> for a drone or a small chip, or when
many images per second are needed. A larger model pays off with more data, or when every missed object counts.
Always compare on the validation images, never on the training images.</p>
""")

# %% [markdown]
# ### 🔴 Freezing the Backbone: Faster, but How Good? (Optional)
#
# `freeze=10` keeps the first ten layers (the backbone, which extracts general features) fixed and trains only the rest.
#
# **Before running the code:** faster or slower? Higher or lower mAP?
#
# **Observe and interpret:** when would freezing be a good choice?

# %%
# @title 🔎 Train YOLO11n with a Frozen Backbone { display-mode: "form" }
if RUN_EXPERIMENTS or SMOKE:
    model_frozen = YOLO("yolo11n.pt")
    start = time.time()
    model_frozen.train(name="yolo11n-frozen", freeze=10, **train_settings)
    train_minutes_frozen = (time.time() - start) / 60
    comparison["YOLO11n, frozen backbone"] = {**evaluate(model_frozen.trainer.best),
                                              "training minutes": round(train_minutes_frozen, 1)}
    display(comparison)
else:
    print("Skipped: tick RUN_EXPERIMENTS above and run this cell again.")

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>The mAP is lower</strong> (in our run mAP@50 dropped from about 0.78 to about 0.72), because the
features stay those learned for COCO's everyday photos, and aerial images of plants look nothing like them.
<strong>Training is not faster here</strong>, unlike what you might expect: for such a small model the frozen
layers still compute their outputs, only their gradients are skipped, and the saving is lost in the rest of the
work (in our test runs the frozen model even took longer).</p>
<p>Freezing is a good choice with very few training images (it reduces overfitting), with little GPU memory, or
when the new images look like the pretraining images. The further our images are from COCO, the more it helps to
fine-tune the backbone as well.</p>
""")

# %%
# @title 📘 What Matters for Object Detection?
callout("theory", "📘 What Matters for Object Detection?", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Start from pretrained weights:</strong> a few hundred images give a usable detector.</li>
<li><strong>Look at the errors</strong>, not only the metric: FPs and FNs show what the model, and sometimes the labels, get wrong.</li>
<li><strong>Thresholds are decisions:</strong> confidence trades precision for recall, NMS trades duplicates for missed neighbors.</li>
<li><strong>Report AP:</strong> mAP@50 says whether objects are found, mAP@50–95 also how tight the boxes are.</li>
</ul>
""")

