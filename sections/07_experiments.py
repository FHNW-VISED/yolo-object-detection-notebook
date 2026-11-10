# %% [markdown]
# ## Optional Experiments
#
# Compare a larger model with ours, and see what freezing the backbone changes.
#
# Each experiment trains another model with the same settings, taking a few minutes on a T4. Tick the box below to run them; unticked, the cells skip themselves and "Run all" stays fast.

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
<p><strong>On this dataset, YOLO11s is a little better, but not by much.</strong> In our run it reached a
mAP@50 of roughly 0.88–0.91 against 0.84–0.87 for YOLO11n, and a mAP@50–95 of roughly 0.57–0.61 against
0.52–0.55. It took about one and a half times as long to train and about twice as long per image on a CPU.</p>
<p><strong>Why:</strong> the gain is larger for mAP@50–95 than for mAP@50: the extra capacity mostly makes boxes
tighter, since YOLO11n already finds most bunches. <strong>Choose YOLO11n</strong> for a camera on a tractor or a
small chip, or when many images per second are needed. <strong>Choose YOLO11s</strong> when box quality matters
and the time per image does not. Always compare on the validation images, never on the training images.</p>
""")

# %% [markdown]
# ### 🔴 Freezing the Backbone: Faster, but How Good? (Optional)
#
# The first layers of a YOLO model, the *backbone*, extract general image features. We can **freeze** them so only later layers train. In YOLO11, `freeze=10` freezes the whole backbone.
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
    train_minutes_frozen = (time.time() - start) / 60
    comparison["YOLO11n, frozen backbone"] = {**evaluate(model_frozen.trainer.best),
                                              "training minutes": round(train_minutes_frozen, 1)}
    display(comparison)
else:
    print("Skipped: tick RUN_EXPERIMENTS above and run this cell again.")

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>The mAP is lower, but only a little</strong> (in our run mAP@50 dropped from about 0.86 to about
0.81, and mAP@50–95 from about 0.54 to about 0.49), because the backbone keeps the features learned for COCO's
everyday photos instead of adapting them to vine leaves and pale bunches. That a frozen backbone still gets so
close shows that those general features (edges, textures, blobs) transfer well.
<strong>Training is faster</strong>: about 3 minutes against about 4 in our run, because the frozen layers need no
gradients. The saving is modest and depends on your hardware.</p>
<p>Freezing is a good choice with very few training images (it reduces overfitting), with little GPU memory, or
when time is short. With enough images, fine-tuning the backbone as well gives the better mAP.</p>
""")

# %%
# @title 📘 What Matters for Object Detection?
callout("theory", "📘 What Matters for Object Detection?", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Start from pretrained weights.</strong> A few hundred images and minutes of fine-tuning give a usable
detector.</li>
<li><strong>Look at the errors, not only the metric.</strong> FPs and FNs show what the model, and sometimes the
labels, get wrong.</li>
<li><strong>Thresholds are decisions.</strong> Confidence trades precision for recall; NMS trades duplicates for
missed neighbors. Choose for the application.</li>
<li><strong>Report AP, not one threshold.</strong> mAP@50 says whether objects are found; mAP@50–95 also how tight
the boxes are.</li>
</ul>
""")
