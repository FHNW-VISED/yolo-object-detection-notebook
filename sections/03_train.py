# %% [markdown]
# ## Fine-Tune YOLO11n
#
# Choose the training settings, fine-tune a pretrained YOLO11n on our dataset, and read its training curves.
#
# We do not train from scratch. The model starts from weights learned on COCO, so it already knows edges, textures and object shapes; fine-tuning adapts it to our classes. This is why a few hundred images and a few minutes are enough.

# %%
# @title 📘 The Main Training Settings
callout("theory", "📘 The Main Training Settings", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>epochs</strong>: how many times the model sees every training image. More epochs learn more, until the
model starts to memorize the training set.</li>
<li><strong>batch</strong>: how many images are processed together in one step. Larger batches give smoother
gradients but need more GPU memory.</li>
<li><strong>imgsz</strong>: images are resized so that their longer side has this many pixels. Small objects need
a larger size.</li>
<li>Ultralytics applies <strong>data augmentation</strong> by default (for example <em>mosaic</em>, which stitches
four training images into one), so the model rarely sees the same picture twice.</li>
</ul>
""")

# %% [markdown]
# ### 🟢 Training Settings: How Long and How Big?
#
# The original YOLO paper trained for hundreds of epochs on COCO's 118,000 images. We have 500 training images, a pretrained model, and a free Colab GPU with 15 GB of memory.
#
# **Before running the code:**
# - How many epochs do you expect to need, more or fewer than for COCO? Why?
#
# **Explore:**
# 1. Set `EPOCHS` and `BATCH_SIZE` in the cell below. `EPOCHS` can be 5 to 50; `BATCH_SIZE` 8, 16 or 32 (a T4 handles 16 at 640 px for YOLO11n). Each extra 10 epochs costs roughly three to four minutes on a T4.
# 2. Run the training cell after it.
#
# **Observe and interpret:**
# - Look at the curves after training. Did the model stop improving before the last epoch, or would more epochs help?

# %%
# ✏️ YOUR TURN: training settings
EPOCHS = 20  # @param {type:"slider", min:5, max:50, step:5}
BATCH_SIZE = 16  # @param [8, 16, 32] {type:"raw"}
IMG_SIZE = 640  # pixels, the size YOLO11 was pretrained at

if BATCH_SIZE not in (8, 16, 32) or not 1 <= EPOCHS <= 100:
    raise ValueError("Choose BATCH_SIZE from 8, 16, 32 and EPOCHS between 1 and 100.")
print(f"Training for {EPOCHS} epochs with batches of {BATCH_SIZE} images at {IMG_SIZE} px.")

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>Far fewer epochs than for COCO.</strong> The model starts from pretrained weights and our dataset is
small, so 20 epochs already give a good detector, with a validation mAP@50 around 0.9 or higher. More epochs would
still add a little, mostly in mAP@50–95, but the gain per epoch is small and each one costs about 20 seconds on a T4.</p>
<p><strong>Batch size 16</strong> fits comfortably in a T4's memory for YOLO11n at 640 px. Smaller batches also
work but each epoch takes longer; 32 may run out of memory for the larger YOLO11s.</p>
""")

# %% [markdown]
# ### Train
#
# The next cell fine-tunes YOLO11n. Ultralytics prints one progress line per epoch: the losses on the training images, then precision, recall and mAP on the validation images. **The first epoch is the slowest**, because the images are read and cached. On a T4 the whole run with the default settings takes about 6 to 8 minutes.

# %%
# @title Train YOLO11n { display-mode: "form" }
# In smoke mode (automated tests only) the same call runs with tiny settings on the CPU.
train_settings = dict(
    data=str(DATA_YAML),
    epochs=1 if SMOKE else EPOCHS,
    batch=4 if SMOKE else BATCH_SIZE,
    imgsz=160 if SMOKE else IMG_SIZE,
    device="cpu" if SMOKE else DEVICE,
    workers=0 if SMOKE else 2,
    seed=SEED,
    deterministic=True,
    project=str(RUNS_DIR.resolve()),
    exist_ok=True,  # re-running overwrites this run instead of creating train2, train3, ...
    plots=not SMOKE,
)

model = YOLO("yolo11n.pt")  # COCO-pretrained weights, downloaded on first use
start = time.time()
model.train(name="yolo11n", **train_settings)
train_minutes_n = (time.time() - start) / 60
best_weights = Path(model.trainer.best)  # the epoch with the best validation fitness
print(f"\nTraining took {train_minutes_n:.1f} min. Best weights: {best_weights}")

# %% [markdown]
# ### 🟢 Training Curves: Is the Model Still Learning?
#
# The left plot shows the box loss (how far the predicted boxes are from the true ones) on the training and on the validation images, after each epoch. The right plot shows the validation mAP, a score from 0 to 1 that we explain in detail later: higher is better.
#
# **Observe and interpret:**
# - Do both losses still go down at the last epoch?
# - Is the validation mAP still rising, or has it leveled off?
# - Would you train longer, shorter, or keep your setting?

# %%
# @title 🔎 Plot the Training Curves { display-mode: "form" }
history = pd.read_csv(Path(model.trainer.save_dir) / "results.csv")
history.columns = history.columns.str.strip()

fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(11, 3.6))
ax_loss.plot(history["epoch"], history["train/box_loss"], "o-", color=INK, label="training images")
ax_loss.plot(history["epoch"], history["val/box_loss"], "o--", color="#E07A5F", label="validation images")
ax_loss.set(title="Box loss (lower is better)", xlabel="epoch", ylabel="loss")
ax_loss.legend()
ax_map.plot(history["epoch"], history["metrics/mAP50(B)"], "o-", color=TP_COLOR, label="mAP@50")
ax_map.plot(history["epoch"], history["metrics/mAP50-95(B)"], "o--", color="#3D405B", label="mAP@50–95")
ax_map.set(title="Validation mAP (higher is better)", xlabel="epoch", ylabel="mAP", ylim=(0, 1))
ax_map.legend()
fig.tight_layout()
show(fig)

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>Both losses fall steadily, and neither has flattened completely at epoch 20.</strong> The validation mAP
is jumpy in the first epochs (it can even drop while the new classes are being learned), then climbs quickly and
rises only slowly in the last few epochs. So the model is still learning a little: a longer run would help
slightly, a shorter one would cost accuracy. Because the validation loss falls together with the training loss,
there is no sign of overfitting yet.</p>
<p><strong>A warning sign would be</strong> a training loss that keeps falling while the validation loss rises:
the model then memorizes the training images (overfitting), and fewer epochs or more augmentation would help.</p>
""")

# %%
# @title 🔍 Go Deeper: More Training Plots and Experiment Tracking
callout("deeper", "🔍 Go Deeper: More Training Plots and Experiment Tracking", r"""
<p style="margin:6px 0 0 0;">Ultralytics saves more plots in the run folder (<code>runs/yolo11n/</code> in the
Colab file browser): all losses, example training batches with their augmentations, and a confusion matrix.
For comparing many runs, it can log to tools such as Weights &amp; Biases or MLflow; see the
<a href="https://docs.ultralytics.com/integrations/">Ultralytics integrations</a>. To keep your weights after the
Colab session ends, copy <code>best.pt</code> to your Google Drive.</p>
""")
