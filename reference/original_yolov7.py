# %% [markdown]
# # Workbook Object Detection in Images Using Deep Learning
# Susanne Suter, HSI FHNW
#
# In this Jupyter notebook, we train a model for object segmentation of images. Object detection means that we detect individual objects in an image and mark them by a bounding box around the image.
#
# While there are many object detection architectures and models around, we are working with the YOLOv7 model in this workbook, as it is one of the widely used and performant models. It is even available for varying target devices.
#
# Comprehensive related articels:
# * [The Ultimate Guide to Object Detection](https://www.v7labs.com/blog/object-detection-guide)
# * [YOLO: Algorithm for Object Detection Explained [+Examples]](https://www.v7labs.com/blog/yolo-object-detection)
#
# Note: Even though this workbook is not strictly setup according to the practices of [Karpathy:19](http://karpathy.github.io/2019/04/25/recipe/), the best practices should be applied here as well.

# %% [markdown]
# ## Before You Start
#
# * Make a copy of this notebook to your google drive
# * Change runtime to a GPU and connect to google drive
#
# ### Google Colab GPUs - some need a pro abo
# from - 2024-10-13
# - *A100 GPU*: The A100 GPU is is one of the top GPU options available in Google Colab.
# - *V100 GPU*: The V100 GPU is well-suited for workloads that require high memory and processing power.
# - *T4 GPU*: The T4 GPU is a more budget-friendly GPU option that still offers good performance for machine learning tasks, although it’s not as powerful as the A100 or V100.
# - *TPU (Tensor Processing Unit)*: TPUs are custom-designed by Google for accelerating machine learning workloads, particularly those that involve neural networks and large-scale data. They are highly specialized and can outperform GPUs in certain scenarios.
#

# %%
# gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

# %% [markdown]
# ### Adjusting Python and Pytorch Versions

# %%
import sys
print(sys.version)

# %% [markdown]
# The current configuration runs only on Python 3.11. Therefore, we need to select an older runtime in Colab. Runtime → Change runtime type → Runtime version → 2025.07 → Save

# %%
import torch
print(torch.__version__)

# %%
# !pip uninstall -y torch torchvision torchaudio

# %%
# !pip install torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0

# %% [markdown]
# ### Setup Google Drive

# %%
import os

from google.colab import drive
mount_base_path = os.path.join('/','content', 'drive')
drive.mount(mount_base_path)

# %%
od_path = os.path.join('MyDrive', 'cv2', 'od')
work_dir_path = os.path.join(mount_base_path, od_path)

os.makedirs(work_dir_path, exist_ok=True)
# %cd {work_dir_path}

# %% [markdown]
# ### ML OPs - Tool Login - Weights and Bias

# %%
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

print(f"In colab: {IN_COLAB}")

if IN_COLAB:
  # !pip install wandb


  # %%
  # click on promted https://wandb.ai/authorize, login and past the auth-code into the field below; hit enter
  # !wandb login


# %% [markdown]
# ## YOLO - One-Stage-Detector

# %% [markdown]
# Setup the object detection model and load the data in the required format.
#
# #### YOLO Model Versions
# During this exercise, we work with [YOLOv7](https://github.com/WongKinYiu/yolov7.git) as object detection model, to work with an open-source baseline version of YOLO ([published](https://arxiv.org/abs/2207.02696)). YOLO (You only look once) is an end-to-end single-stage object detector (SSD) using CNNs frequently in practice.  Since its release, YOLO was continously developed. Ultralytics regularly releases new updates. SSDs are well suited for real-time applications as they are more performant (e.g., videos). Most YOLO versions include data augmentation techiques such as mosaic. YOLO is available at different sizes depending on the computing requirements in the target environment.
#
# YOLOv7 model variations:
# * YOLOv7-Tiny: edge GPU
# * YOLOv7: normal (consumer) GPU
# * YOLOv7-W6: cloud GPU
# * YOLOv7-E6, YOLOv7-D6, YOLOv7-E6E: high-end cloud GPUs only
#
# YOLOv7 introduces the *focal loss*. Previous YOLO versions used cross-entropy loss function, which is less successful at detecting small objects. Focal loss down-weightes the loss for well-classified examples and focuses the objects hard to detect.

# %% [markdown]
# #### Download YOLO Repository

# %%
# only needs to be executed once or in case an update is desired/required
# this is the official YOLOv7
# !git clone https://github.com/WongKinYiu/yolov7.git

# %% [markdown]
# ### Load and Format Data
#

# %% [markdown]
# #### Dataset

# %% [markdown]
# For this exercise, we are working with the cattle dataset ([data](cattle), [official website](http://bird.nae-lab.org/cattle/)). The cattle dataset was created to detect and count cattle using *unmanned aerial vehicle* (UAV) images. For this exercise, cattle dataset1 (653 images) was used as training data, and dataset2 (14 images) as validation data. To keep the training times shorter during the deep dive, we downsized the images to 800x600px.
#
# Further image preprocessing steps such as scaling of the pixel values could be applied, as illustrated in the segmentation workbook. For simplicity, we are not performing these preparation steps here.

# %% [markdown]
# #### Data Format Recap

# %% [markdown]
# As annotations, [YOLO's data format](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) expects normalized bounding boxes for objects given their classes in `*.txt` files. There is one annotation file per image with the same filename. For each object, the object class index and the position of the bounding box are required, while one object is annotated in a single line:
#
# ```
# class_idx x_center y_center width height
# ```
# The bounding box is normalized according to the images size. One annoated line looks as follows, e.g., for the cattle data:
#
# ```
# 0 0.565250 0.760333 0.012000 0.034000
# ```
#
# YOLO expects the data in two folders for training and validation data. In each folder, sub folders for images and labels are expected. The custom folder names can be configured in `yolov7/data/cattle.yaml`.
#
# ```
# train: ./data/cattle/train
# val: ./data/cattle/val
# ```
#
# In the cattle dataset, we only detect one object class, which is configured in the file `yolov7/data/cattle.yaml` as well:
#
# ```
# # number of classes
# nc: 1
#
# # class names
# names: [ 'cattle' ]
# ```

# %% [markdown]
# #### Define Relevant Directories

# %%
import os

yolo_dir         = os.path.join(work_dir_path, 'yolov7')
data_dir         = os.path.join(yolo_dir, 'data')

train_dir = os.path.join(data_dir, 'cattle', 'train')
train_img_dir = os.path.join(train_dir, 'images')
train_labels_dir = os.path.join(train_dir, 'labels')

val_dir = os.path.join(data_dir, 'cattle', 'val')
val_img_dir = os.path.join(val_dir, 'images')
val_labels_dir = os.path.join(val_dir, 'labels')

# %% [markdown]
# #### Download Data

# %%
# !pip install -q gdown

# %%
cattle_zip_path  = os.path.join(data_dir, 'cattle.zip')
# !gdown 1TPc-hs8Rq6UjaWjdEh35eSl_NYHAb2dW -O {data_dir}/hyp.scratch.cattle.yaml
# !gdown 1RnOJnLsPWOSrPtccMMxroKKM4mtIRs9_ -O {data_dir}/cattle.yaml
# !gdown 1GBeS5ujc-yVuIMvNhgHDtyFCIoAKXIS2 -O {cattle_zip_path}

# %%
# !unzip -q -o {cattle_zip_path} -d {data_dir}

# %%
# %ls -la {data_dir}/*cattle*

# %% [markdown]
# ### Analyze the Data Qualitatively
#
# #### <font color='green'>Exercise </font>
# 1. Analyze the images and labels by visual inspection and report the problems.
# 2. Get a qualitative sense of the data and try to find any sort of patterns that are visible to the human brain
#

# %% [markdown]
# #### Verify Labels Visually

# %% [markdown]
# ##### Helper Functions Visualization

# %%
# helpers gt
import os
from PIL import Image, ImageDraw
from PIL.ImageFile import ImageFile

def get_shape(line, w_, h_):
  elems = line.split(' ')
  id = elems[0]
  x_centre = float(elems[1])
  y_centre = float(elems[2])
  w = float(elems[3])
  h = float(elems[4])

  x0 = int((x_centre - (w / 2))* w_)
  y0 = int((y_centre - (h / 2))* h_)
  x1 = int((x_centre + (w / 2))* w_)
  y1 = int((y_centre + (h / 2))* h_)

  shape = [(x0, y0), (x1,y1)]
  return shape

def display_gt_one_image(img_path, label_path, prediction_path=None):
    image = Image.open(img_path)
    with open(label_path, "r", encoding="utf8") as label_file:
        lines = label_file.readlines()
        for line in lines:
            shape = get_shape(line, image.width, image.height)
            img1 = ImageDraw.Draw(image)
            img1.rectangle(shape,  outline="red")

    if prediction_path is not None:
        with open(prediction_path, "r", encoding="utf8") as label_file:
            lines = label_file.readlines()
            for line in lines:
                shape = get_shape(line, image.width, image.height)
                img1 = ImageDraw.Draw(image)
                img1.rectangle(shape,  outline="white")

    return image
from PIL import Image
from IPython.display import HTML, display
import io, base64

def display_images_with_scroll(images, width=200, height=200, orientation="horizontal", max_height=500):
    """
    Display a list of PIL images in a scrollable container (no temp files needed).
    orientation: "horizontal" or "vertical"
    """
    img_tags = []
    for img in images:
        # ensure consistent format
        if img.mode in ("P", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        img_tags.append(
            f'<img src="data:image/png;base64,{b64}" '
            f'width="{width}" height="{height}" style="margin-right:10px;margin-bottom:10px;">'
        )

    if orientation == "vertical":
        container_style = (
            f"display:flex; flex-direction:column; gap:10px; "
            f"overflow-y:auto; overflow-x:hidden; "
            f"max-height:{max_height}px; border:1px solid #ccc; padding:10px;"
        )
    else:  # horizontal
        container_style = (
            "display:flex; flex-direction:row; gap:10px; "
            "overflow-x:auto; overflow-y:hidden; "
            "border:1px solid #ccc; padding:10px;"
        )

    html = f'<div style="{container_style}">{"".join(img_tags)}</div>'
    display(HTML(html))



# %% [markdown]
# ##### Visualize Sample Images with GT Boxes

# %%
import glob

all_images = sorted(glob.glob(os.path.join(train_img_dir, "*.JPG")))
all_labels = sorted(glob.glob(os.path.join(train_labels_dir, "*.txt")))
n_images = len(all_images)

# pick every (step)th image, up to 10 total
step = 5   # e.g. indices 0, 5, 10, ...
selected = all_images[0:n_images:5][:20]
print(selected)

images_to_show = []
for img_path in selected:
    basename = os.path.basename(img_path).replace(".JPG", ".txt")
    label_path = os.path.join(train_labels_dir, basename)
    if os.path.exists(label_path):
        img = display_gt_one_image(img_path, label_path)
        images_to_show.append(img)

display_images_with_scroll(images_to_show, width=400, height=300)

# %% [markdown]
# ### Analyze the Data Quantitatively
#
# #### <font color='green'>Exercise </font>
# Analyze the images and labels using code. E.g, count number of images, images shapes and their frequencies, classes per image, numbers and sizes of object boxes, average +/- std of number of pixels, pixel value ranges and distributions, imbalances, bias, and other quantitative measures that you can think of.

# %% [markdown]
# ### Train YOLO Model

# %% [markdown]
# Note: in case you experience problems with the training, delete the caches in the data folders.

# %%
# %cd {yolo_dir}

# %% [markdown]
# #### Download the Model

# %%
# download pre-trained weights (MS-COCO dataset)
# only needed once or upon updates
# !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt

# %% [markdown]
# #### Training

# %% [markdown]
#
# Selected training parameters:
# * `--weights`: type=str, default='yolo7.pt', help='initial weights path'
# * `--cfg`: type=str, default='', help='model.yaml path'
# * `--data`: type=str, default='data/coco.yaml', help='data.yaml path'
# * `--hyp`: type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path'
# * `--epochs`: type=int, default=300
# * `--batch-size`: type=int, default=16, help='total batch size for all GPUs'
# * `--img-size`: type=int, default=[640, 640], help='[train, test] image sizes'
# * `--device`: default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
# * `--workers`: type=int, default=8, help='maximum number of dataloader workers'
# * `--name`: default='exp', help='save to project/name'
# * see more parameters in train.py file of YOLOv7, e.g., data augmentation related parameters

# %% [markdown]
# ##### <font color='green'>Exercise </font>

# %%
# Task: choose training params (see below)
# Task: choose YOLO config params, e.g., augmention, IOU threshold, see file hyp.scratch.cattle.yaml
#### BEGIN SOLUTION
batch_size = ...
epochs = ...
### END SOLUTION

# %% [markdown]
# ##### <font color='orange'>Sample Solution </font>

# %%
#### BEGIN SOLUTION
batch_size = 16
epochs = 10
### END SOLUTION

# %% [markdown]
# ##### Train

# %%
# adjust epochs to get a satisfying result
# !python train.py \
#   --device 0 \
#   --workers 8 \
#   --batch-size $batch_size \
#   --epochs $epochs \
#   --img 640 640 \
#   --data {data_dir}/cattle.yaml \
#   --hyp {data_dir}/hyp.scratch.cattle.yaml \
#   --cfg cfg/training/yolov7x.yaml \
#   --weights yolov7x.pt \
#   --name yolov7x-cattle


# %% [markdown]
# #### Evaluate Training
#
# Object detectors are typicall evaluated using *average precision*, which is the area under the precision-recall curve for different class probability thresholds.  
#
# ##### <font color='green'>Exercise </font>
#
# Evaluate the trained model using wandb.ai: E.g., https://wandb.ai/cv2/YOLOR (adjust path depending on your setup)

# %% [markdown]
# #### Detect Objects (Inference)

# %% [markdown]
# ##### <font color='green'>Exercise </font>
# Check individual predictions. The results can be found in the `runs/detect/...` folder of `YOLO`.
#
# Selected detect parameters:
# * `--weights`: type=str, default='yolov7.pt', help='trained weights of a specific model.pt path(s)'
# * `--source`: type=str, default='inference/images', help='source image(s) file/folder name'  # file/folder, 0 for webcam
# * `--img-size`: type=int, default=640, help='inference size (pixels)'
# * `--conf-thres`: type=float, default=0.25, help='object confidence threshold'
# * `--no-trace`: action='store_true', help='don`t trace model'
# * `--save-txt`: if set stores label files
# * see further parameters in [detect.py](./detect.py) file of YOLOv7
#

# %% [markdown]
# Optionally upload and use weights from Susanne.
# https://drive.google.com/drive/folders/1icTtAkdu60uIxz4kjZ5MWKwRRMVRbSKw?usp=drive_link

# %%
import os

tmp_dir = os.path.join(yolo_dir,'tmp')
os.makedirs(tmp_dir, exist_ok=True)
weights_file = 'yolo7-weights.pt'

# %%
# !gdown 1BUFarx96Et8_w1Do7l5JZmFjREZL2eav -O {tmp_dir}/{weights_file}
# !gdown 11jVFMljRPnhbR5wxZdpqcm4lNWhtOqwz -O {tmp_dir}/DJI_0028_test-img.jpg

# %%
# define variables
weights_path = "..."
conf_thresh = ...
#...

# %% [markdown]
# ##### <font color='orange'>Sample Solution </font>

# %%
cattle_run = "cattle4"
trained = False
if trained:
  weights_path = f"{yolo_dir}/runs/train/yolov7x-{cattle_run}/weights/best.pt"
else:
  weights_path = f"{tmp_dir}/{weights_file}"
conf_thresh = 0.5
print(weights_path)

# %% [markdown]
# ##### Detect

# %%
# !python detect.py \
#   --weights {weights_path} \
#   --conf-thres {conf_thresh} \
#   --img-size 640 \
#   --source {tmp_dir}/DJI_0028_test-img.jpg \
#   --no-trace \
#   --save-txt

# %% [markdown]
# Predict for all images in validation set:

# %%
# !python detect.py \
#   --weights {weights_path} \
#   --conf-thres {conf_thresh} \
#   --img-size 640 \
#   --source {os.path.join(data_dir, 'cattle', 'val', 'images')} \
#   --no-trace \
#   --save-txt

# %% [markdown]
# ##### <font color='green'>Exercise</font>

# %%
# adjust experiment folder to current run
experiment = 'expx'

# %% [markdown]
# ##### <font color='orange'>Sample Solution </font>

# %%
experiment = 'exp17'

# %% [markdown]
# ##### Visualize

# %%
import os, glob
import matplotlib.pyplot as plt

out_dir = os.path.join(yolo_dir, 'runs', 'detect')
exp_dir = os.path.join(out_dir, experiment)

input_imgs = sorted(
    glob.glob(os.path.join(val_img_dir, "*.jpg")) +
    glob.glob(os.path.join(val_img_dir, "*.JPG"))
)

pred_labels_dir = os.path.join(exp_dir, "labels")

images_to_show = []
for img_path in input_imgs:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(pred_labels_dir, stem + ".txt")
    if os.path.exists(label_path):
        img = display_gt_one_image(img_path, label_path)
        images_to_show.append(img)

display_images_with_scroll(images_to_show, width=400, height=300)


# %% [markdown]
# #### Intersection over Union (IOU)

# %% [markdown]
# ##### Visualize Predictions vs. Ground Truth Boxes

# %%
import os, glob
import matplotlib.pyplot as plt

out_dir = os.path.join(yolo_dir, 'runs', 'detect')
exp_dir = os.path.join(out_dir, experiment)

input_imgs = sorted(
    glob.glob(os.path.join(val_img_dir, "*.jpg")) +
    glob.glob(os.path.join(val_img_dir, "*.JPG"))
)

pred_labels_dir = os.path.join(exp_dir, "labels")

images_to_show = []
for img_path in input_imgs:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(val_labels_dir, stem + ".txt")
    pred_path = os.path.join(pred_labels_dir, stem + ".txt")

    if os.path.exists(label_path):
        img = display_gt_one_image(img_path, label_path, pred_path)
        images_to_show.append(img)

display_images_with_scroll(images_to_show, width=400, height=300)


# %% [markdown]
# ##### <font color='green'>Exercise </font>

# %% [markdown]
# Check the labels and predictions of the sample image.
# - What are true positives, false positives, and false negatives?
# - Calculate the IOU and print it for the boxes that match with an IOU of at least 0.5

# %%
def calculate_iou(box1, box2):

    x_min1, y_min1 = box1[0]
    x_max1, y_max1 = box1[1]
    x_min2, y_min2 = box2[0]
    x_max2, y_max2 = box2[1]

    #TODO: continue here
    iou = 0

    return iou



# %% [markdown]
# ##### <font color='orange'>Sample Solution </font>

# %%
def calculate_iou(box1, box2):
    x_min1, y_min1 = box1[0]
    x_max1, y_max1 = box1[1]
    x_min2, y_min2 = box2[0]
    x_max2, y_max2 = box2[1]

    # Intersection rectangle
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Compute intersection area (check for overlap)
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # Areas of each box
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Union area
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou



# %% [markdown]
# Calculate IOU

# %%
def match_boxes(img_path, label_path, prediction_path):
    image = Image.open(img_path)

    with open(label_path, "r", encoding="utf8") as label_file:
        lines = label_file.readlines()
        with open(prediction_path, "r", encoding="utf8") as label_file:
            lines_pred = label_file.readlines()

            for line in lines:
                box1 = get_shape(line, image.width, image.height)

                for line_pred in lines_pred:
                    box2 = get_shape(line_pred, image.width, image.height)
                    iou = calculate_iou(box1, box2)
                    if iou > 0.5:
                      print('box1 and box2:', box1, box2)
                      print('IOU box1 and box2=', iou)


# example, you'll have to put your paths
test_img = os.path.join(exp_dir, 'DJI_0030.JPG')
test_pred = os.path.join(pred_labels_dir, 'DJI_0030.txt')
test_label = os.path.join(val_labels_dir, 'DJI_0030.txt')

match_boxes(test_img, test_label, test_pred)


# %% [markdown]
# ### Optional Exercise: Freeze Layers during Training

# %% [markdown]
# Yolo can be trained for all weights or selected weights. Usually we input the network, what layers we do not want do retrain. This is called *freezing*.
#
#
# ##### <font color='green'>Exercise </font>
# Retrain your network by freezing certain weights.
# * `--freeze`, nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2'
#

# %%
# !python train.py \
#   --device 0 \
#   --workers 8 \
#   --batch-size $batch_size \
#   --epochs $epochs \
#   --img 640 640 \
#   --data {data_dir}/cattle.yaml \
#   --hyp {data_dir}/hyp.scratch.cattle.yaml \
#   --cfg cfg/training/yolov7x.yaml \
#   --weights yolov7x.pt \
#   --name yolov7x-cattle \
#   --freeze 50

# %% [markdown]
# ### Optional Exercise: Download and Train Tiny YOLO Model

# %%
# download trained weights, only needed once or upon updates
# !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

# %% [markdown]
# ##### <font color='green'>Exercise </font>
# Check individual predictions. The results can be found in the `runs/detect/...` folder of `YOLO`.
#

# %%
# !python train.py --device 0 --workers 8 --batch-size 16 --epochs 200 \
#  --img 640 640 --data data/cattle/cattle.yaml \
#  --hyp data/cattle/hyp.scratch.cattle.yaml \
#  --cfg cfg/training/yolov7-tiny.yaml \
#  --weights yolov7-tiny.pt --name yolov7-tiny-cattle

# %%
# remember to adjust the folder with the current run
# !python detect.py --weights runs/train/yolov7-tiny-cattle1/weights/best.pt --conf-thres 0.5 --img-size 640 --source ../DJI_0028_test-img.jpg --no-trace --save-txt

# %%
# display a second example with more objects and bounding boxes
out_dir = os.path.join('runs/detect')
test_img = os.path.join(out_dir, 'exp9','DJI_0028_test-img.jpg')
test_label = os.path.join(out_dir, 'exp9','labels','DJI_0028_test-img.txt')
image_test1 = display_gt_one_image(test_img, test_label)

_ = plt.imshow(image_test1)

# %% [markdown]
# ##### <font color='green'>Exercise </font>
#
# Evaluate the training using wandb.ai: E.g., https://wandb.ai/dlbs/YOLOR (adjust path depending on your setup). What differences do you notice in the training results from the two YOLOv7 models?
#
