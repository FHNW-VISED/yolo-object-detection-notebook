# %% [markdown]
# ## Fine-Tune YOLO11n
#
# Choose training settings, fine-tune YOLO11n on our dataset, and read its training curves.
#
# We do not train from scratch. The model starts from COCO weights, so it already knows edges, textures and shapes; fine-tuning adapts it to our classes. A few hundred images and minutes suffice.

# %%
# @title 📘 The Main Training Settings
callout("theory", "📘 The Main Training Settings", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>epochs</strong>: how many times the model sees every training image. More epochs learn more, until it
memorizes the training set.</li>
<li><strong>batch</strong>: images processed together in one step. Larger batches give smoother gradients but
need more GPU memory.</li>
<li><strong>imgsz</strong>: the pixel size of an image's longer side. Small objects need a larger size.</li>
<li><strong>Data augmentation</strong> is on by default (for example <em>mosaic</em>, which stitches four images
into one), so the model rarely sees the same picture twice.</li>
</ul>
""")

# %% [markdown]
# ### 🟢 Training Settings: How Long and How Big?
#
# YOLO was trained for hundreds of epochs on COCO's 118,000 images. We have a few hundred images, a pretrained model, and a free Colab GPU.
#
# **Before running the code:**
# - Will you need more or fewer epochs than for COCO? Why?
#
# **Explore:**
# 1. Set `EPOCHS` (5 to 50) and `BATCH_SIZE` (8, 16 or 32; a T4 handles 16) in the cell below. Each extra 10 epochs costs about three minutes on a T4.
# 2. Run the training cell after it.
#
# **Observe and interpret:**
# - Did the model stop improving before the last epoch, or would more epochs help?

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
<p><strong>Far fewer epochs than for COCO.</strong> The model starts from pretrained weights and has only 500
training images, so 20 epochs take a few minutes and already give a usable detector. For these tiny objects the
model is still improving slowly at epoch 20, so 30 epochs may add a little mAP, at the price of about three more
minutes on a T4.</p>
<p><strong>Batch size 16</strong> fits comfortably in a T4's memory for YOLO11n at 640 px. Smaller batches also
work but each epoch takes longer; 32 may run out of memory for the larger YOLO11s.</p>
""")

# %% [markdown]
# ### Train
#
# The next cell fine-tunes YOLO11n and prints one line per epoch: training losses, then precision, recall and mAP on the validation images. **The first epoch is the slowest**, because the images are cached. On a T4 the default run takes 6 to 8 minutes.

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
# The left plot shows the box loss (how far predicted boxes are from true ones) per epoch. The right plot shows the validation mAP, a score from 0 to 1 explained later: higher is better.
#
# **Observe and interpret:**
# - Do both losses still go down at the last epoch?
# - Is the validation mAP still rising or leveled off?
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
<p><strong>Both losses fall steadily and are still falling at the end.</strong> The training and validation loss
move together, with no gap that grows, so the model is not overfitting yet.</p>
<p><strong>The validation mAP is noisy at first</strong> and can jump up or down by 0.1 or more between epochs,
because the validation set is small. It settles in the last five epochs at roughly 0.7–0.8 (mAP@50) and 0.4–0.5 (mAP@50–95)
and is still creeping up: 30 epochs would probably add a few points, but not change the picture.</p>
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
