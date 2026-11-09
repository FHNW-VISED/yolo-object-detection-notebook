# %% [markdown]
# ## The Dataset
#
# Download the dataset, recall YOLO's label format, and look at some images and boxes.
#
# A drone flies low over a field and takes photos. A farmer who knows where the weeds are can spray only those spots, saving herbicide. A detector must find every weed and tell it from the crop.
#
# Our dataset has photos taken from above, with boxes around crop plants and weeds: two classes, almost all weeds. The objects are tiny, around 20 pixels wide in a 640 × 640 image, and look nothing like COCO's photos. We only look briefly here.

# %% [markdown]
# The next cell downloads the images (about 55 MB, one to two minutes the first time; later runs reuse them). To keep training short, we use a fixed random subset: 500 training and 150 validation images.

# %%
# @title Download the Dataset { display-mode: "form" }
import shutil

import yaml
from huggingface_hub import list_repo_files, snapshot_download
from ultralytics.data.utils import check_det_dataset

REPO_ID = "LibreYOLO/weed-crop-aerial"  # Roboflow 100 "weed-crop-aerial" in YOLO format, CC BY 4.0, no login needed
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def images_in(folder) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for(image: Path) -> Path:
    """YOLO's convention: the label file sits in a 'labels' folder parallel to 'images', same file name."""
    parts = list(Path(image).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")


def download_subset(n_train: int, n_val: int) -> Path:
    """Download a fixed random sample of images and labels, and return the path of its data.yaml.

    Safe to run again: a finished download is reused, an unfinished one is restarted (N-019).
    """
    root = Path(f"data/datasets/weed-crop-aerial-{n_train}-{n_val}").resolve()
    yaml_path = root / "data.yaml"
    if yaml_path.exists():  # written last, so its presence means the download finished
        return yaml_path
    shutil.rmtree(root, ignore_errors=True)
    files = list_repo_files(REPO_ID, repo_type="dataset")
    chosen = []
    for folder, n in (("train", n_train), ("valid", n_val)):
        names = sorted(f for f in files if f.startswith(f"{folder}/images/"))
        for image in random.Random(SEED).sample(names, n):
            chosen += [image, str(Path(image.replace("/images/", "/labels/")).with_suffix(".txt"))]
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=root, allow_patterns=chosen, token=False)
    yaml_path.write_text(yaml.safe_dump({"path": str(root), "train": "train/images", "val": "valid/images",
                                         "names": dict(enumerate(["crop", "weed"]))}))
    return yaml_path


# In smoke mode (automated tests only) the same code downloads a handful of images instead.
data_info = check_det_dataset(str(download_subset(8, 4) if SMOKE else download_subset(500, 150)))
CLASS_NAMES = [data_info["names"][k] for k in sorted(data_info["names"])]
DATA_YAML = Path(data_info["yaml_file"])
TRAIN_IMAGES = images_in(data_info["train"])
VAL_IMAGES = images_in(data_info["val"])
print(f"{len(TRAIN_IMAGES)} training images · {len(VAL_IMAGES)} validation images · classes: {', '.join(CLASS_NAMES)}")

# %% [markdown]
# ### Data Format Recap
#
# YOLO expects one text file per image, with the same name, in a `labels` folder next to `images`. Each line is one object: class index and box, **normalized** by image width and height (0 to 1):
#
# ```
# class_idx x_center y_center width height
# ```
#
# A file `data.yaml` tells Ultralytics where the images are and what the classes are called:
#
# ```
# path: /content/data/datasets/weed-crop-aerial-500-150
# train: train/images
# val: valid/images
# names:
#   0: crop
#   1: weed
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
# Each page shows four training images with their ground-truth boxes, one color per class. Browse a few pages to see the objects' size, how many appear together, and how they differ from everyday photos.

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
# The pretrained YOLO11n knows COCO's 80 everyday classes: people, cars, dogs, cups. None is one of ours.
#
# **Before running the code:**
# - What will it predict on our images: nothing, wrong classes, or our objects under another name?
#
# **Explore:**
# - Run the cell to see its predictions on four validation images.
#
# **Observe and interpret:**
# - Which COCO classes does it use, if any? Why do we need to fine-tune?

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
<p><strong>It predicts almost nothing.</strong> On the four images the COCO model finds no box at all. Over all
150 validation images it finds a box in only about one image in ten, with the wrong kind of label: a "bird", a
"frisbee" or a "potted plant" on a patch of soil or a leaf. None of COCO's 80 classes is a weed or a crop plant, and
the objects are far smaller than the people and cars it knows.</p>
<p><strong>Why fine-tune:</strong> a detector can only output the classes it was trained on. Its features (edges,
textures, shapes) are still useful, so we keep them and teach the last layers our classes.</p>
""")
