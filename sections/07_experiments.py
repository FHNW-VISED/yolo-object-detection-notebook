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
<p><strong>On this dataset, YOLO11s is clearly better.</strong> In our test run YOLO11s reached about 0.70 mAP@50
against 0.65 for YOLO11n, and about 0.33 mAP@50–95 against 0.27: roughly five points on both. It took about 1.5 times
as long to train (11 minutes against 7 in our run) and needs more computation per image.</p>
<p><strong>Why:</strong> the cows are only a few pixels wide, so the extra capacity helps to tell a cow from a
bush or a shadow, and the 500 images hold thousands of cows. <strong>Choose YOLO11n</strong> for a drone or a small
chip, or when many images per second are needed; <strong>choose YOLO11s</strong> when the count must be as
accurate as possible and the hardware allows it. The size of the gap can change with another seed, so always compare
on the validation images, never on the training images.</p>
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
<p><strong>The mAP is slightly lower</strong> (in our run mAP@50 dropped from about 0.65 to about 0.62 and
mAP@50–95 from about 0.27 to about 0.25), because the features stay those learned for COCO's everyday photos, and
aerial images of cows look different. The drop is small: edges and textures carry over, and a difference of two
points is within what two runs with another seed can show. <strong>Training was faster</strong>: about 5 minutes
against 7 in our run, because the frozen layers need no gradients, so the backward pass is shorter.</p>
<p>Freezing is a good choice with very few training images (it reduces overfitting), with little GPU memory, when
speed matters, or when the new images look like the pretraining images. The further the images are from COCO, the more it
helps to fine-tune the backbone as well.</p>
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
