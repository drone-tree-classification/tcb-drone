#!/usr/bin/env python3

import os
import sys
import argparse
import urllib.parse
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
    from tensorflow.keras.models import load_model
    import tensorflow_hub as hub # Sometimes off-the-shelf models are best loaded via hub, but we can also use keras get_file

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
            # Memory growth must be set before GPUs have been initialized
            print(f"{PROGRAM_NAME}: Warning: GPU configuration failed: {e}", file=sys.stderr)
    else:
        print(f"{PROGRAM_NAME}: Info: No GPU detected. Training will run on CPU.", file=sys.stderr)

parser = argparse.ArgumentParser(description='Fine-tune an object detection model on PASCAL VOC labelled images.')
parser.add_argument('-m', '--model', type=str, required=True, help='Path or URL to the input object detection model (SavedModel dir, .h5, .keras, or a downloadable archive URL).')
parser.add_argument('-d', '--datasets', type=str, nargs='+', required=True, help='Paths to the directories containing the dataset.')
parser.add_argument('-o', '--output', type=str, default='FineTunedODModel.keras', help='Path to save the fine-tuned model.')
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train for.')
parser.add_argument('-c', '--classes', type=str, default='classes.txt', help='File to read/write class labels.')
parser.add_argument('--max_boxes', type=int, default=20, help='Maximum number of bounding boxes per image.')

args = parser.parse_args()

if tf is None or cv2 is None or np is None:
    print("Required libraries not found. Ensure tensorflow, opencv, and numpy are installed.", file=sys.stderr)
    sys.exit(1)


def get_model_path(model_arg):
    # Check if URL
    parsed = urllib.parse.urlparse(model_arg)
    if parsed.scheme in ('http', 'https'):
        print(f"Downloading model from {model_arg}...")
        fname = os.path.basename(parsed.path)
        if not fname:
            fname = "downloaded_model"

        extract = False
        if fname.endswith(('.zip', '.tar.gz', '.tgz', '.tar')):
            extract = True

        download_path = tf.keras.utils.get_file(
            fname=fname,
            origin=model_arg,
            extract=extract,
            cache_subdir='models'
        )

        if extract:
            # If extracted, the actual model might be a subdirectory (SavedModel format)
            # Find the directory containing saved_model.pb
            cache_dir = os.path.dirname(download_path)
            for root, dirs, files in os.walk(cache_dir):
                if 'saved_model.pb' in files:
                    return root
                # Alternatively look for a .keras or .h5 inside
                for f in files:
                    if f.endswith(('.keras', '.h5')):
                        return os.path.join(root, f)

            # Fallback to the cache directory if nothing specific found (might just be a flat dir of weights)
            return cache_dir
        else:
            return download_path

    return model_arg

model_path = get_model_path(args.model)

# Load existing model
try:
    print(f"Loading model from {model_path}")
    # Compile false to prevent requiring custom loss functions on load
    model = load_model(model_path, compile=False)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

# Try to infer input shape
try:
    input_shape = model.input_shape
    trainImageHeight = input_shape[1]
    trainImageWidth = input_shape[2]
    print(f"Inferred input shape: {trainImageWidth}x{trainImageHeight}")
except Exception as e:
    print(f"Warning: Could not infer input shape: {e}. Defaulting to 512x512.", file=sys.stderr)
    trainImageHeight = 512
    trainImageWidth = 512

if trainImageWidth is None or trainImageHeight is None:
    trainImageHeight = 512
    trainImageWidth = 512

TrainingSetPathList = args.datasets
checkpoint_path = args.output
indexRecord = args.classes
epochs = args.epochs
MAX_BBOXES = args.max_boxes

classesArray = []
try:
    with open(indexRecord, 'r') as file:
        for line in file:
            c = line.strip()
            if c:
                classesArray.append(c)
except FileNotFoundError:
    print(PROGRAM_NAME + ": Warning: classes file: " + indexRecord + " not found. Starting fresh.", file=sys.stderr)
except Exception as e:
    print(PROGRAM_NAME + ": Warning: classes file: " + indexRecord + " error: " + str(e), file=sys.stderr)


def parsePascalVocXml(path, classes):
    success = True
    width = 0
    height = 0
    depth = 0

    try:
        root = ET.parse(path).getroot()
    except FileNotFoundError:
        success = False

    returnList = []

    if success:
        if len(root.findall('size')) > 0:
            type_tag = root.findall('size')
            try:
                width = int(type_tag[0].findall('width')[0].text)
                height = int(type_tag[0].findall('height')[0].text)
                depth = int(type_tag[0].findall('depth')[0].text)
            except IndexError:
                pass

        for type_tag in root.findall('object'):
            try:
                name = str(type_tag.findall('name')[0].text)
                if name in classes:
                    classid = classes.index(name)
                else:
                    classes.append(name)
                    classid = len(classes) - 1
            except IndexError:
                continue

            if len(type_tag.findall('bndbox')) > 0:
                type_tag2 = type_tag.findall('bndbox')
                try:
                    xmin = int(float(type_tag2[0].findall('xmin')[0].text))
                    ymin = int(float(type_tag2[0].findall('ymin')[0].text))
                    xmax = int(float(type_tag2[0].findall('xmax')[0].text))
                    ymax = int(float(type_tag2[0].findall('ymax')[0].text))

                    returnList.append({'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'classID': classid})
                except IndexError:
                    continue
            else:
                continue

    return success, width, height, depth, returnList

## 1. Produce list of files
textFileList = []
jpgFileList = []
for TrainingSetPath in TrainingSetPathList:
    for (dirpath, dirnames, filenames) in walk(TrainingSetPath):
        for filename in filenames:
            lastPeriod=filename.rfind('.')
            if lastPeriod > 0:
                if filename[lastPeriod:].lower() == ".xml":
                    textFileList.append(os.path.join(dirpath, filename))
                elif filename[lastPeriod:].lower() == ".jpg":
                    jpgFileList.append(os.path.join(dirpath, filename))
                else:
                    print(PROGRAM_NAME + ": Warning: Unidentified extension: " + str(filename[lastPeriod:]) + " found on file path, " + str(filename) + ", continuing.", file=sys.stderr)

xyFileList = []

## 2. Associate images with xml files
jpg_dict = {f[:f.rfind('.')]: f for f in jpgFileList}
for x in textFileList:
    xLastPeriod=x.rfind('.')
    key = x[:xLastPeriod]
    if key in jpg_dict:
        xyFileList.append({'image': jpg_dict[key], 'text':x, 'parsed': False})

parsed_data = []

## 3. Parse XML
for x in xyFileList:
    success, orig_width, orig_height, depth, parseList = parsePascalVocXml(x["text"], classesArray)
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
    classes = []

    scale_x = trainImageWidth / float(orig_width)
    scale_y = trainImageHeight / float(orig_height)

    for i in parseList:
        # Scale bounding boxes to new image dimensions
        xmin = np.clip(i['xmin'] * scale_x, 0, trainImageWidth)
        xmax = np.clip(i['xmax'] * scale_x, 0, trainImageWidth)
        ymin = np.clip(i['ymin'] * scale_y, 0, trainImageHeight)
        ymax = np.clip(i['ymax'] * scale_y, 0, trainImageHeight)

        # Format as [ymin, xmin, ymax, xmax] standard for TF OD
        boxes.append([ymin, xmin, ymax, xmax])
        classes.append(i['classID'])

    x['parsed'] = True
    parsed_data.append({
        'image_path': x['image'],
        'boxes': boxes,
        'classes': classes
    })


num_images = len(parsed_data)
numTreeTypes = len(classesArray)

if num_images == 0:
    print(PROGRAM_NAME + ": Error: No valid annotations found in datasets.", file=sys.stderr)
    sys.exit(1)

# Prepare numpy arrays
x_train = np.zeros(shape=(num_images, trainImageHeight, trainImageWidth, 3), dtype=np.float32)
y_train_boxes = np.zeros(shape=(num_images, MAX_BBOXES, 4), dtype=np.float32)
y_train_classes = np.zeros(shape=(num_images, MAX_BBOXES, numTreeTypes), dtype=np.float32)

for idx, data in enumerate(parsed_data):
    # Load and resize image
    im = cv2.imread(data['image_path'])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_resized = cv2.resize(im, (trainImageWidth, trainImageHeight))
    x_train[idx] = im_resized / 255.0  # Normalize

    # Pad boxes and classes
    num_boxes = min(len(data['boxes']), MAX_BBOXES)
    for b in range(num_boxes):
        y_train_boxes[idx, b] = data['boxes'][b]
        class_id = data['classes'][b]
        y_train_classes[idx, b, class_id] = 1.0


def box_regression_loss(y_true, y_pred):
    """Huber loss for bounding box regression."""
    huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    loss = huber(y_true, y_pred)
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)

def classification_loss(y_true, y_pred):
    """Categorical Crossentropy loss for object classes."""
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = cce(y_true, y_pred)
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)


# Check model output format
num_outputs = len(model.outputs)

print("Compiling model for object detection...")
if num_outputs == 2:
    # Typical OD model with 2 output heads [boxes, classes]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=[box_regression_loss, classification_loss]
    )
    targets = [y_train_boxes, y_train_classes]
else:
    # Fallback to train as a single output model if it outputs a combined tensor
    # e.g. [batch, boxes, 4 + classes]
    y_train_combined = np.concatenate([y_train_boxes, y_train_classes], axis=-1)

    def combined_loss(y_true, y_pred):
        boxes_true = y_true[..., :4]
        boxes_pred = y_pred[..., :4]
        classes_true = y_true[..., 4:]
        classes_pred = y_pred[..., 4:]

        b_loss = box_regression_loss(boxes_true, boxes_pred)
        c_loss = classification_loss(classes_true, classes_pred)
        return b_loss + c_loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss
    )
    targets = y_train_combined

print(str(model.summary()))

# Train the model
print("Starting training...")
model.fit(
    x_train,
    targets,
    epochs=epochs,
    batch_size=8,
    validation_split=0.2 if num_images > 10 else 0.0
)

model.save(checkpoint_path)
print(f"Saved fine-tuned object detection model to {checkpoint_path}")

try:
    with open(indexRecord, "w") as f:
        for entry in classesArray:
            f.write(str(entry) + "\n")
except Exception as e:
    print(PROGRAM_NAME + ": Error: writing to file: " + str(e), file=sys.stderr)
    sys.exit(1)
