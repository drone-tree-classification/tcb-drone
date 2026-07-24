#!/usr/bin/env python3

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from os import walk

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

if tf is not None:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv2D, Reshape
    import tensorflow.keras.applications as applications

PROGRAM_NAME=str(sys.argv[0].lstrip('.').lstrip('/'))

if tf is not None:
    print(tf.__version__)

    # Configure GPU to use memory dynamically if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{PROGRAM_NAME}: Info: Found {len(gpus)} GPU(s). Enabled dynamic memory growth.", file=sys.stderr)
        except RuntimeError as e:
            print(f"{PROGRAM_NAME}: Warning: GPU configuration failed: {e}", file=sys.stderr)
    else:
        print(f"{PROGRAM_NAME}: Info: No GPU detected. Training will run on CPU.", file=sys.stderr)

parser = argparse.ArgumentParser(description='Train a ResNet-based object detection model for 1 class (trees) on PASCAL VOC labelled images.')
parser.add_argument('-m', '--model', type=str, default='ResNet50', help='Name of the ResNet model to use from tf.keras.applications (e.g., ResNet50). Default is ResNet50.')
parser.add_argument('-d', '--datasets', type=str, nargs='+', required=True, help='Paths to the directories containing the dataset.')
parser.add_argument('-o', '--output', type=str, default='FineTunedResNetOD.keras', help='Path to save the fine-tuned model.')
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train for.')
parser.add_argument('-c', '--classes', type=str, default='classes.txt', help='File to read/write class labels.')

args = parser.parse_args()

if tf is None or cv2 is None or np is None:
    print("Required libraries not found. Ensure tensorflow, opencv, and numpy are installed.", file=sys.stderr)
    sys.exit(1)

# Retrieve the correct ResNet model function
try:
    resnet_constructor = getattr(applications, args.model)
except AttributeError:
    print(f"Error: {args.model} is not a valid model in tf.keras.applications.", file=sys.stderr)
    sys.exit(1)

# A fixed input size (e.g. 512x512) typically results in a 16x16 feature map for ResNet50
trainImageHeight = 512
trainImageWidth = 512
GRID_SIZE = 16  # For 512x512 with 32x downsampling (standard ResNet) -> 16x16 grid

print(f"Loading {args.model} base model with ImageNet weights...")
base_model = resnet_constructor(
    weights='imagenet',
    include_top=False,
    input_shape=(trainImageHeight, trainImageWidth, 3)
)

# Freeze the backbone
base_model.trainable = False

# Build a simple grid-based detection head (YOLO-lite for 1 class)
# Output channels = 5 (object confidence, center_x, center_y, width, height)
x = base_model.output
# Additional conv layer to process features
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
# Final conv layer to output the grid predictions with a sigmoid activation
# so that all values (confidence and normalized box coords) are between 0 and 1
predictions = Conv2D(5, (1, 1), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

TrainingSetPathList = args.datasets
checkpoint_path = args.output
indexRecord = args.classes
epochs = args.epochs

classesArray = []
try:
    with open(indexRecord, 'r') as file:
        for line in file:
            c = line.strip()
            if c:
                classesArray.append(c)
except FileNotFoundError:
    print(PROGRAM_NAME + ": Warning: classes file: " + indexRecord + " not found. Starting fresh.", file=sys.stderr)

def parsePascalVocXml(path):
    success = True
    width = 0
    height = 0

    try:
        root = ET.parse(path).getroot()
    except FileNotFoundError:
        success = False

    returnList = []

    if success:
        if len(root.findall('size')) > 0:
            size_tag = root.findall('size')[0]
            try:
                width = int(size_tag.find('width').text)
                height = int(size_tag.find('height').text)
            except (IndexError, AttributeError, TypeError):
                pass

        for obj_tag in root.findall('object'):
            bndbox = obj_tag.find('bndbox')
            if bndbox is not None:
                try:
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    returnList.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
                except (IndexError, AttributeError, TypeError):
                    continue

    return success, width, height, returnList

## 1. Produce list of files
textFileList = []
jpgFileList = []
for TrainingSetPath in TrainingSetPathList:
    for (dirpath, dirnames, filenames) in walk(TrainingSetPath):
        for filename in filenames:
            lastPeriod = filename.rfind('.')
            if lastPeriod > 0:
                ext = filename[lastPeriod:].lower()
                if ext == ".xml":
                    textFileList.append(os.path.join(dirpath, filename))
                elif ext == ".jpg":
                    jpgFileList.append(os.path.join(dirpath, filename))

## 2. Associate images with xml files
xyFileList = []
jpg_dict = {f[:f.rfind('.')]: f for f in jpgFileList}
for x in textFileList:
    key = x[:x.rfind('.')]
    if key in jpg_dict:
        xyFileList.append({'image': jpg_dict[key], 'text': x, 'parsed': False})

parsed_data = []

## 3. Parse XML and Map Bounding Boxes to Grid
for x in xyFileList:
    success, orig_width, orig_height, parseList = parsePascalVocXml(x["text"])
    if not success:
        print(PROGRAM_NAME + ": Warning: Issue parsing: " + str(x["text"]) + " skipping", file=sys.stderr)
        continue

    if orig_width == 0 or orig_height == 0:
        # Fallback to reading image to get dims
        img_temp = cv2.imread(x["image"])
        if img_temp is not None:
            orig_height, orig_width, _ = img_temp.shape
        else:
            continue

    boxes = []

    scale_x = trainImageWidth / float(orig_width)
    scale_y = trainImageHeight / float(orig_height)

    for i in parseList:
        # Scale bounding boxes to new image dimensions
        xmin = np.clip(i['xmin'] * scale_x, 0, trainImageWidth)
        xmax = np.clip(i['xmax'] * scale_x, 0, trainImageWidth)
        ymin = np.clip(i['ymin'] * scale_y, 0, trainImageHeight)
        ymax = np.clip(i['ymax'] * scale_y, 0, trainImageHeight)

        # Convert to center_x, center_y, width, height
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + (w / 2.0)
        cy = ymin + (h / 2.0)

        # Normalize to [0, 1] across the entire image
        norm_cx = cx / trainImageWidth
        norm_cy = cy / trainImageHeight
        norm_w = w / trainImageWidth
        norm_h = h / trainImageHeight

        boxes.append([norm_cx, norm_cy, norm_w, norm_h])

    parsed_data.append({
        'image_path': x['image'],
        'boxes': boxes
    })


num_images = len(parsed_data)
if num_images == 0:
    print(PROGRAM_NAME + ": Error: No valid annotations found in datasets.", file=sys.stderr)
    sys.exit(1)

x_train = np.zeros(shape=(num_images, trainImageHeight, trainImageWidth, 3), dtype=np.float32)
# y_train shape is (num_images, GRID_SIZE, GRID_SIZE, 5)
y_train = np.zeros(shape=(num_images, GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)

for idx, data in enumerate(parsed_data):
    # Load and resize image
    im = cv2.imread(data['image_path'])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_resized = cv2.resize(im, (trainImageWidth, trainImageHeight))
    # Standard normalization, or use specific ResNet preprocess_input
    x_train[idx] = im_resized / 255.0

    for box in data['boxes']:
        # box = [norm_cx, norm_cy, norm_w, norm_h]
        # Determine which grid cell this box falls into
        grid_x = int(box[0] * GRID_SIZE)
        grid_y = int(box[1] * GRID_SIZE)

        # Clip to ensure valid grid indices
        grid_x = np.clip(grid_x, 0, GRID_SIZE - 1)
        grid_y = np.clip(grid_y, 0, GRID_SIZE - 1)

        # If multiple objects fall into same cell, this simplistic approach overwrites.
        # Set object confidence to 1.0
        y_train[idx, grid_y, grid_x, 0] = 1.0
        # Set box coordinates (relative to whole image [0, 1])
        y_train[idx, grid_y, grid_x, 1:5] = box

def od_loss(y_true, y_pred):
    """
    Custom loss for single-class object detection.
    y_true/y_pred shape: (batch, GRID_SIZE, GRID_SIZE, 5)
    Index 0: object confidence
    Indices 1-4: cx, cy, w, h
    """
    # Extract confidence mask (1 if object exists in cell, 0 otherwise)
    obj_mask = y_true[..., 0]

    # Binary Crossentropy for object confidence (applies to all cells)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    conf_loss = bce(y_true[..., 0:1], y_pred[..., 0:1])

    # Huber loss for bounding boxes
    huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    # y_true boxes are already scaled 0-1
    box_loss = huber(y_true[..., 1:5], y_pred[..., 1:5])

    # Only apply box loss to cells that actually contain an object
    masked_box_loss = box_loss * obj_mask

    # Sum up loss over the grid and average over the batch
    total_conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2]))
    total_box_loss = tf.reduce_mean(tf.reduce_sum(masked_box_loss, axis=[1, 2]))

    # Weight the box loss slightly higher to emphasize coordinate precision
    return total_conf_loss + (total_box_loss * 5.0)

print("Compiling grid-based object detection model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=od_loss,
    # Adding MAE for boxes just as an observable metric (masked manually since metrics don't accept masks directly)
)

print(str(model.summary()))

# Train the model
print("Starting training...")
model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=8,
    validation_split=0.2 if num_images > 10 else 0.0
)

model.save(checkpoint_path)
print(f"Saved fine-tuned object detection model to {checkpoint_path}")
