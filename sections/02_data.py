# %% [markdown]
# ## The Dataset
#
# Download the dataset, recall YOLO's label format, and take a short look at the images and their boxes.
#
# Steel cables on cranes, lifts and cable cars are inspected regularly, and a snapped strand that nobody notices can end in an accident. A detector that scans close-up photos and flags the damaged spots lets the inspector look only at those. Our images show steel cables with two kinds of damage, a **break** (a snapped strand) and a **thunderbolt** (a burnt, melted spot), small marks far from COCO's everyday photos.
#
# You explored a detection dataset in depth in *Know Your Images*. Here we only look briefly, enough to know what the model has to learn.

# %% [markdown]
# The next cell downloads the dataset (about 35 MB) the first time it runs; later runs reuse it. The full dataset has 919 training and 265 validation images. To keep training short, we use a fixed random subset of 500 training and 150 validation images.

# %%
# @title Download the Dataset { display-mode: "form" }
import random

import yaml
from huggingface_hub import list_repo_files, snapshot_download
from ultralytics.data.utils import check_det_dataset

HF_REPO = "LibreYOLO/cable-damage"  # on the Hugging Face Hub, public: no login needed
N_TRAIN, N_VAL = (8, 4) if SMOKE else (500, 150)  # a fixed random subset keeps training short
DATA_ROOT = Path("data/smoke" if SMOKE else "data/datasets/cable-damage").resolve()
CLASS_NAMES = ["break", "thunderbolt"]  # a broken strand · a burn mark where electricity melted the wires


def label_path_for(image: Path) -> Path:
    """YOLO's convention: the label file sits in a 'labels' folder parallel to 'images', same file name."""
    parts = list(Path(image).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")


def download_subset(n_train: int, n_val: int) -> Path:
    """Download a fixed random subset (images + labels) and return the path of its data.yaml.
    Safe to run again: files that are already there are skipped, and an interrupted download continues."""
    repo_files = list_repo_files(HF_REPO, repo_type="dataset")
    chosen = {}
    for split, n in (("train", n_train), ("valid", n_val)):
        images = sorted(f for f in repo_files if f.startswith(f"{split}/images/"))
        chosen[split] = random.Random(SEED).sample(images, n)
    wanted = [f for images in chosen.values() for image in images
              for f in (image, label_path_for(Path(image)).as_posix())]
    snapshot_download(HF_REPO, repo_type="dataset", token=False, local_dir=DATA_ROOT, allow_patterns=wanted)
    for split, images in chosen.items():  # the lists tell Ultralytics exactly which images belong to a split
        (DATA_ROOT / f"{split}.txt").write_text("".join(f"{DATA_ROOT / image}\n" for image in sorted(images)))
    yaml_path = DATA_ROOT / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(DATA_ROOT), "train": "train.txt", "val": "valid.txt",
                                         "names": dict(enumerate(CLASS_NAMES))}))
    return yaml_path


DATA_YAML = download_subset(N_TRAIN, N_VAL)
data_info = check_det_dataset(str(DATA_YAML))  # checks that every listed image exists
TRAIN_IMAGES = [Path(line) for line in (DATA_ROOT / "train.txt").read_text().splitlines()]
VAL_IMAGES = [Path(line) for line in (DATA_ROOT / "valid.txt").read_text().splitlines()]
print(f"{len(TRAIN_IMAGES)} training images · {len(VAL_IMAGES)} validation images · classes: {', '.join(CLASS_NAMES)}")

# %% [markdown]
# ### Data Format Recap
#
# YOLO expects one text file per image, with the same file name, in a `labels` folder next to the `images` folder. Each line describes one object: its class index and its bounding box, **normalized** by the image width and height (so all values lie between 0 and 1):
#
# ```
# class_idx x_center y_center width height
# ```
#
# A file `data.yaml` tells Ultralytics where the training and validation images are and what the classes are called:
#
# ```
# path: /content/data/datasets/cable-damage
# train: train.txt   # a list of image files; a folder of images works too
# val: valid.txt
# names:
#   0: break
#   1: thunderbolt
# ```
#
# The next cell prints the first lines of one label file and its image size.

# %%
# @title 🔎 Read One Label File { display-mode: "form" }
example_image = TRAIN_IMAGES[0]
with Image.open(example_image) as im:
    print(f"{example_image.name}: {im.width} × {im.height} pixels")
print(f"{label_path_for(example_image).name}:")
print("\n".join(label_path_for(example_image).read_text().splitlines()[:5]))


# %%
# @title Helpers: Ground-Truth Boxes { display-mode: "form" }
def read_gt(image) -> dict:
    """The ground-truth boxes of one image: {"xyxy": N×4 pixel corners, "cls": N class indices}."""
    label = label_path_for(Path(image))
    rows = np.loadtxt(label, ndmin=2) if label.exists() and label.stat().st_size else np.zeros((0, 5))
    with Image.open(image) as im:
        w, h = im.size
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return {"xyxy": xyxy, "cls": rows[:, 0].astype(int)}


CLASS_COLORS = ["#E07A5F", "#3D405B", "#81B29A", "#9C89B8", "#F2CC8F", "#2A9D8F"]  # one per class, in order


def show_gt(ax, image, title=""):
    gt = read_gt(image)
    ax.imshow(Image.open(image).convert("RGB"))
    ax.axis("off")
    for box, k in zip(gt["xyxy"], gt["cls"], strict=True):
        x0, y0, x1, y1 = box
        ax.add_patch(mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.8,
                                           edgecolor=CLASS_COLORS[k % len(CLASS_COLORS)]))
    ax.set_title(f"{title}{len(gt['cls'])} boxes", fontsize=9)


# %% [markdown]
# ### A Short Look at the Training Images
#
# Each page shows four training images with their ground-truth boxes, one color per class. Browse a few pages to get a feeling for the objects: their size, how many appear together, and how they differ from everyday photos.

# %%
# @title 🔎 Browse Training Images with Their Boxes { display-mode: "form" }
gt_page = 0
gt_output = widgets.Output()
gt_prev, gt_next = widgets.Button(description="← Previous"), widgets.Button(description="Next →")
gt_counter = widgets.HTML()
GT_PAGE_SIZE = 4


def gt_show():
    with gt_output:
        clear_output(wait=True)
        page = TRAIN_IMAGES[gt_page * GT_PAGE_SIZE:(gt_page + 1) * GT_PAGE_SIZE]
        fig, axes = plt.subplots(1, GT_PAGE_SIZE, figsize=(16, 4), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, image in zip(axes.ravel(), page, strict=False):
            show_gt(ax, image)
        handles = [mpl.patches.Patch(edgecolor=CLASS_COLORS[k % len(CLASS_COLORS)], fill=False, lw=2, label=name)
                   for k, name in enumerate(CLASS_NAMES)]
        fig.legend(handles=handles, loc="lower center", ncol=len(CLASS_NAMES), frameon=False,
                   bbox_to_anchor=(0.5, -0.06))
        show_live(fig)
    n_pages = -(-len(TRAIN_IMAGES) // GT_PAGE_SIZE)
    gt_counter.value = f"Page {gt_page + 1} of {n_pages}"
    gt_prev.disabled, gt_next.disabled = gt_page == 0, gt_page >= n_pages - 1


def gt_step(step):
    global gt_page
    gt_page += step
    gt_show()


gt_prev.on_click(lambda _: gt_step(-1))
gt_next.on_click(lambda _: gt_step(1))
display(widgets.HBox([gt_prev, gt_next, gt_counter]), gt_output)
gt_show()

# %% [markdown]
# ### 🟢 Before Fine-Tuning: What Does a COCO Model See?
#
# The pretrained YOLO11n has learned COCO's 80 everyday classes: people, cars, dogs, cups, and so on. None of them is one of our classes.
#
# **Before running the code:**
# - What will the pretrained model predict on our images: nothing, wrong classes, or our objects under another name?
#
# **Explore:**
# - Run the cell. It shows the predictions of the COCO model on four validation images.
#
# **Observe and interpret:**
# - Which COCO classes does the model use, if any? What does this tell you about why we need to fine-tune?

# %%
# @title 🔎 Predict with the COCO-Pretrained Model { display-mode: "form" }
coco_model = YOLO("yolo11n.pt")  # downloads the COCO weights on first use
fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for ax, image in zip(axes.ravel(), VAL_IMAGES[:4], strict=False):
    r = coco_model.predict(str(image), conf=0.25, device="cpu" if SMOKE else DEVICE, verbose=False)[0]
    ax.imshow(r.plot(line_width=2)[..., ::-1])  # r.plot() draws the COCO class names; BGR → RGB
    found = sorted({coco_model.names[int(k)] for k in r.boxes.cls})
    ax.set_title(f"COCO model: {', '.join(found) if found else 'nothing found'}", fontsize=9)
fig.suptitle("Predictions of the COCO-pretrained YOLO11n before fine-tuning (confidence ≥ 0.25)", color=INK)
show(fig)

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p>VARIANT: what the COCO model actually predicts on these images, read from a full run.</p>
<p><strong>Why fine-tune:</strong> a detector can only output the classes it was trained on. Its features (edges,
textures, shapes) are still useful, so we keep them and teach the last layers our classes.</p>
""")
