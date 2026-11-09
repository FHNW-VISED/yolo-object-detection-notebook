# %% [markdown]
# ## Optional Experiments
#
# Compare a larger model with the one we trained, and see what freezing the pretrained backbone changes.
#
# Both experiments train another model with the same settings as before, so each takes a few minutes on a T4. Tick the box in the next cell to run them; with the box unticked, the cells skip themselves and "Run all" stays fast.

# %%
# ✏️ YOUR TURN: run the optional experiments?
RUN_EXPERIMENTS = False  # @param {type:"boolean"}

# %% [markdown]
# ### 🔴 YOLO11n vs. YOLO11s: Is the Larger Model Worth It? (Optional)
#
# YOLO11s has about 3.6 times as many parameters as YOLO11n.
#
# **Before running the code:**
# - How much better do you expect YOLO11s to be on our small dataset, and how much slower?
#
# **Explore:**
# - Run the cell. It trains YOLO11s with your settings and compares both models on the validation images.
#
# **Observe and interpret:**
# - Compare mAP@50, mAP@50–95, training time and prediction time per image.
# - For which use would you choose each model?

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
<p><strong>YOLO11s is usually a few points of mAP better</strong>, with the larger gain in mAP@50–95 (tighter
boxes), and it takes roughly one and a half to two times as long to train and to predict.</p>
<p><strong>Choose YOLO11n</strong> for a phone, a drone or a camera with a small chip, or when many images per
second are needed. <strong>Choose YOLO11s</strong> (or larger) when a GPU is available and every missed object
counts. On a small dataset the larger model can also overfit sooner, so compare on the validation images, never on
the training images.</p>
""")

# %% [markdown]
# ### 🔴 Freezing the Backbone: Faster, but How Good? (Optional)
#
# The first layers of a YOLO model, the *backbone*, extract general image features. We can **freeze** them, so that only the later layers are trained. In YOLO11, `freeze=10` freezes the first ten layers, which is the whole backbone.
#
# **Before running the code:**
# - Will training be faster or slower? Will the mAP be higher or lower than without freezing?
#
# **Explore:**
# - Run the cell. It retrains YOLO11n with the backbone frozen and adds it to the comparison.
#
# **Observe and interpret:**
# - When would freezing be a good choice?

# %%
# @title 🔎 Train YOLO11n with a Frozen Backbone { display-mode: "form" }
if RUN_EXPERIMENTS or SMOKE:
    model_frozen = YOLO("yolo11n.pt")
    start = time.time()
    model_frozen.train(name="yolo11n-frozen", freeze=10, **train_settings)
    comparison["YOLO11n, frozen backbone"] = {**evaluate(model_frozen.trainer.best),
                                              "training minutes": round((time.time() - start) / 60, 1)}
    display(comparison)
else:
    print("Skipped: tick RUN_EXPERIMENTS above and run this cell again.")

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>Training is faster</strong>, because no gradients are computed for the frozen layers, <strong>and the
mAP is usually somewhat lower</strong>, because the features stay those learned for COCO's everyday photos.</p>
<p>Freezing is a good choice with very few training images (it reduces overfitting), with little compute, or when
the new images look like the pretraining images. The further our images are from COCO, the more it helps to
fine-tune the backbone as well.</p>
""")

# %%
# @title 📘 What Matters for Object Detection?
callout("theory", "📘 What Matters for Object Detection?", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Start from pretrained weights.</strong> A few hundred images and a few minutes of fine-tuning give a
usable detector.</li>
<li><strong>Look at the errors, not only at the metric.</strong> False positives and false negatives show what the
model, and sometimes the labels, get wrong.</li>
<li><strong>Thresholds are decisions.</strong> The confidence threshold trades precision for recall; the NMS
threshold trades duplicates for missed neighbors. Choose them for the application.</li>
<li><strong>Report AP, not one threshold.</strong> mAP@50 says whether objects are found; mAP@50–95 also says how
tight the boxes are.</li>
</ul>
""")
